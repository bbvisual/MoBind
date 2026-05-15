import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_contrastive_loss(logit_scale, cls_i, cls_m, t_i=None, t_m=None, global_weight=1.0, local_weight=0.5):
    cls_i = F.normalize(cls_i, dim=-1)
    cls_m = F.normalize(cls_m, dim=-1)

    logit_scale = torch.clamp(logit_scale, max=np.log(100)).exp()

    logits_per_imu = logit_scale * cls_i @ cls_m.t()
    logits_per_motion = logits_per_imu.t()

    batch_size = cls_i.shape[0]
    labels = torch.arange(batch_size, device=cls_i.device).long()
    loss_g = (
        F.cross_entropy(logits_per_imu, labels) +
        F.cross_entropy(logits_per_motion, labels)
    ) / 2

    loss_l = torch.tensor(0.0, device=cls_i.device)
    if t_i is not None and t_m is not None:
        if local_weight > 0:
            t_i = F.normalize(t_i, dim=-1)
            t_m = F.normalize(t_m, dim=-1)
            B, N, D = t_i.shape

            logits_per_imu = logit_scale * torch.bmm(t_i, t_m.transpose(1, 2))
            logits_per_motion = logits_per_imu.transpose(1, 2)

            labels = torch.arange(N, device=t_i.device).expand(B, N)

            loss_imu = F.cross_entropy(logits_per_imu.reshape(B * N, N), labels.reshape(B * N))
            loss_motion = F.cross_entropy(logits_per_motion.reshape(B * N, N), labels.reshape(B * N))
            loss_l = (loss_imu + loss_motion) / 2

    total_loss = loss_g * global_weight + loss_l * local_weight
    return total_loss, loss_g, loss_l


def compute_multi_pos_contrastive_loss(logit_scale, cls_m, local_i):
    cls_m = F.normalize(cls_m, dim=-1).detach()
    local_i = F.normalize(local_i, dim=-1)

    B, S, D = local_i.shape

    logit_scale = torch.clamp(logit_scale, max=np.log(100)).exp()

    logits_p2i_g = logit_scale * (cls_m @ local_i.reshape(B * S, -1).t())
    blocks_p2i_g = logits_p2i_g.reshape(B, B, S)
    pos_block_g  = blocks_p2i_g[torch.arange(B), torch.arange(B)]
    num_g   = torch.logsumexp(pos_block_g, dim=1) - np.log(S)
    denom_g = torch.logsumexp(logits_p2i_g, dim=1)
    loss_g_pose2imu = -(num_g - denom_g).mean()

    logits_i2p_g = logit_scale * (local_i.reshape(B * S, -1) @ cls_m.t())
    labels_i2p_g = torch.arange(B, device=local_i.device).repeat_interleave(S)
    loss_g_imu2pose = F.cross_entropy(logits_i2p_g, labels_i2p_g)

    loss_g = 0.5 * (loss_g_pose2imu + loss_g_imu2pose)
    return loss_g


class ContrastiveStage1Module(nn.Module):
    def __init__(self, imu_encoder, motion_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float))
        self.imu_encoder = imu_encoder
        self.motion_encoder = motion_encoder

    def forward(self, input, global_weight=1.0, local_weight=0.5, multi_weight=0.0, is_train=True):
        motion_input = input['motion']
        imu_input = input['imu']

        cls_m, token_m = self.motion_encoder(motion_input)
        cls_i, token_i = self.imu_encoder(imu_input)

        total_loss, loss_g, loss_l = compute_contrastive_loss(self.logit_scale, cls_i, cls_m, token_i, token_m, global_weight, local_weight)
        return {
            'total_loss': total_loss,
            'loss_g': loss_g,
            'loss_l': loss_l,
            'cls_i': cls_i,
            'cls_m': cls_m,
            'token_i': token_i,
            'token_m': token_m,
        }


class ContrastiveMAE(nn.Module):
    def __init__(
        self,
        imu_encoder, motion_encoder,
        num_limbs: int=6, global_emb_dim: int=256,
        num_heads: int=8, num_blocks: int=4, droppath: float=0.1,
        lambda_mae: float=0.3, mask_ratio: float=0.75
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float))
        self.motion_encoder = motion_encoder
        self.imu_encoder = imu_encoder

        self.num_limbs = num_limbs

        num_patches_i = self.imu_encoder.num_patches
        num_patches_m = self.motion_encoder.num_patches
        assert num_patches_i == num_patches_m, f"Expected equal number of patches, but got {num_patches_i} and {num_patches_m}."
        self.num_patches = num_patches_i

        emb_size_i = self.imu_encoder.embedding_size
        emb_size_m = self.motion_encoder.embedding_size
        assert emb_size_i == emb_size_m, f"Expected equal embedding sizes, but got {emb_size_i} and {emb_size_m}."
        self.embedding_size = emb_size_i

        self.local2global_imu = nn.Linear(global_emb_dim, global_emb_dim)

        self.imu_norm = nn.LayerNorm(self.embedding_size)
        self.motion_norm = nn.LayerNorm(self.embedding_size)
        self.imu_agg_proj = nn.Sequential(
            nn.Linear(num_limbs*emb_size_i, 2*global_emb_dim), nn.GELU(),
            nn.Linear(2*global_emb_dim, global_emb_dim)
        )
        self.motion_agg_proj = nn.Sequential(
            nn.Linear(num_limbs*emb_size_i, 2*global_emb_dim), nn.GELU(),
            nn.Linear(2*global_emb_dim, global_emb_dim)
        )

        self.sensor_id_emb = nn.Embedding(self.num_limbs, self.embedding_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_size))
        self.imu_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, self.embedding_size))

        self.imu_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads,
            dim_feedforward=self.embedding_size*4, dropout=droppath, batch_first=True
        ), num_layers=num_blocks)
        self.imu_pred = nn.Linear(self.embedding_size, self.embedding_size)
        self.pre_pred_norm = nn.LayerNorm(self.embedding_size)

        self.lambda_mae = lambda_mae
        self.mask_ratio = mask_ratio
        self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        nn.init.normal_(self.sensor_id_emb.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.imu_pos_emb, std=0.02)

    def forward(self, input, global_weight=1.0, local_weight=0.0, multi_weight=0.0, is_train=False):
        motion_input = input['motion']
        imu_input = input['imu']
        device = imu_input.device

        B, S, Cm, Tm = motion_input.shape
        B, S, Ci, Ti = imu_input.shape
        assert S == self.num_limbs, f"Expected {self.num_limbs} limbs, but got {S} limbs in the input sequence."

        imu_input = imu_input.reshape(B * S, Ci, Ti)
        motion_input = motion_input.reshape(B * S, Cm, Tm)
        local_i, token_i = self.imu_encoder(imu_input)
        local_m, token_m = self.motion_encoder(motion_input)

        token_i = token_i.reshape(B, S, self.num_patches, -1)
        token_m = token_m.reshape(B, S, self.num_patches, -1)

        local_i = self.imu_norm(local_i.view(B, S, -1))
        local_m = self.motion_norm(local_m.view(B, S, -1))

        projected_i = self.imu_agg_proj(local_i.reshape(B, S * local_i.size(-1)))
        projected_m = self.motion_agg_proj(local_m.reshape(B, S * local_m.size(-1)))

        total_loss, global_loss, local_loss = compute_contrastive_loss(
            self.logit_scale, projected_i, projected_m, global_weight=global_weight, local_weight=0)

        local_i_projected = self.local2global_imu(local_i.reshape(B*S, -1))
        local_i_projected = local_i_projected.reshape(B, S, -1)

        multi_pos_loss = compute_multi_pos_contrastive_loss(
            self.logit_scale, projected_m, local_i_projected)

        total_loss = total_loss + multi_weight * multi_pos_loss

        mae_loss = torch.tensor(0.0, device=device)
        if is_train:
            B, S, T, D = token_i.shape
            N = B * S
            x = token_i.view(N, T, D)

            L_keep = int(T * (1.0 - self.mask_ratio))
            noise = torch.rand(N, T, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :L_keep]
            x_keep = x.gather(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

            mask = torch.ones(N, T, device=x.device)
            mask[:, :L_keep] = 0
            mask = mask.gather(dim=1, index=ids_restore)
            mask_bool = mask.bool()

            mask_tokens = self.mask_token.expand(N, T - L_keep, D)
            x_ = torch.cat([x_keep, mask_tokens], dim=1)
            x_ = x_.gather(dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

            pos = self.imu_pos_emb.expand(1, T, D).repeat(N, 1, 1)
            x_ = x_ + pos
            ids_limb = torch.arange(S, device=x.device).repeat_interleave(B)
            x_ = x_ + self.sensor_id_emb(ids_limb)[:, None, :]

            x_ = self.imu_transformer(x_)
            x_ = self.pre_pred_norm(x_)
            pred = self.imu_pred(x_)

            tgt = x.detach()
            mae_loss = F.mse_loss(pred[mask_bool], tgt[mask_bool])
            total_loss = total_loss + self.lambda_mae * mae_loss

        return {
            'total_loss': total_loss,
            'loss_g': global_loss,
            'loss_l': local_loss,
            'loss_mse': mae_loss,
            'cls_i': projected_i,
            'cls_m': projected_m,
            'token_i': token_i,
            'token_m': token_m,
            'local_i': local_i,
            'local_m': local_m,
        }

    def forward_imu(self, imu_input):
        B, S, C, T = imu_input.shape
        imu_input = imu_input.reshape(B * S, C, T)
        local_i, token_i = self.imu_encoder(imu_input)
        token_i = token_i.reshape(B, S, self.num_patches, -1)

        local_i = local_i.view(B, S, -1)
        local_i = self.imu_norm(local_i)
        global_i = self.imu_agg_proj(local_i.reshape(B, S * local_i.size(-1)))
        return local_i, global_i
