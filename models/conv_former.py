import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=2, num_heads=4, dropout=0.1, num_patches=25, class_token=False):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(1, num_patches + 1, input_dim)) if class_token else nn.Parameter(torch.empty(1, num_patches, input_dim))
        self.use_cls_token = class_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.init_parameters()

    def forward(self, x):
        if self.use_cls_token:
            B, T, D = x.size()
            class_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([class_tokens, x], dim=1)  # (B, 1 + N, embed_dim)
        x = x + self.pos_emb
        x = self.transformer(x)  # (B, N, D)
        return x

    @torch.no_grad()
    def init_parameters(self):
        nn.init.normal_(self.pos_emb, std=0.01)
        nn.init.normal_(self.cls_token, std=0.01)


class ConvFormer(nn.Module):
    def __init__(self,
                 patch_sec=0.2,
                 window_sec=5,
                 sample_rate=50,
                 input_channels=3,
                 filter_size=5,
                 embedding_size=256,
                 num_layers=4,
                 num_heads=8,
                 class_token=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=filter_size, padding=filter_size // 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=filter_size, padding=filter_size // 2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=filter_size, padding=filter_size // 2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=filter_size, padding=filter_size // 2)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

        self.input_channels = input_channels
        self.patch_size = int(patch_sec * sample_rate)
        self.num_patches = int(window_sec / patch_sec)
        self.embedding_size = embedding_size
        self.use_cls_token = class_token

        self.pre_transformer = nn.Linear(self.patch_size * 128, embedding_size, bias=False)
        self.transformer = PatchTransformer(input_dim=embedding_size, num_layers=num_layers, num_heads=num_heads,
                                            num_patches=self.num_patches, class_token=class_token)
        self.projector = nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = x.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, num_patches, C * patch_size)

        x = self.pre_transformer(x)
        x = self.transformer(x)
        x = self.dropout(x)

        if self.use_cls_token:
            local_feats = x[:, 1:, :]
            global_feats = self.projector(x[:, 0, :])
        else:
            local_feats = x
            global_feats = self.projector(x.mean(dim=1))
        return global_feats, local_feats
