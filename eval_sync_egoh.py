"""
Synchronization evaluation for EgoHumans.

Person-level: each sample is one person. Uses cache_sync_action_{window}_{stride}.
Video-level:  each sample has P people. Uses cache_sync_action_multi_{window}_{stride}.
              Votes are aggregated across all persons' diagonal similarity blocks.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from omegaconf import OmegaConf
from builder import build_model
from datasets.utils import get_sync_annotations, normalize_pose, extract_limb_motion
from configs.config import DATASET_CONFIG, DATA_ROOT


def eval_person_level(model, data_list, sync_dir, dataset_name, limb_list,
                      wd_imu, stride_imu, wd_mot, stride_mot, sampled_stride_s, topk, device):
    gt_offsets, pred_offsets = [], []

    for data in tqdm(data_list, desc='Person-level sync'):
        file_idx, w_s, w_e, subject, action, offset = data
        gt_offsets.append(offset)

        imu      = np.load(os.path.join(sync_dir, 'imu',           f"{file_idx:05d}.npy"))
        pose2d   = np.load(os.path.join(sync_dir, 'pose2d',        f"{file_idx:05d}.npy"))
        pose2d_s = np.load(os.path.join(sync_dir, 'pose2d_shifted', f"{file_idx:05d}.npy"))

        pose2d_norm   = normalize_pose(pose2d,   dataset_name)
        pose2d_s_norm = normalize_pose(pose2d_s, dataset_name)

        # motion: (S, T, C, J) → (S, T, Cm)
        motion_list  = [extract_limb_motion(pose2d_norm,   dataset_name, l) for l in limb_list]
        shifted_list = [extract_limb_motion(pose2d_s_norm, dataset_name, l) for l in limb_list]
        motion  = torch.tensor(np.stack(motion_list,  axis=0)).float()
        shifted = torch.tensor(np.stack(shifted_list, axis=0)).float()
        S, T, C, J = motion.shape
        motion  = motion.reshape(S, T, C * J)   # (S, T, Cm)
        shifted = shifted.reshape(S, T, C * J)  # (S, T, Cm)

        # imu: (T, S, Ci) → (S, T, Ci)
        imu_t = torch.tensor(imu).permute(1, 0, 2).float()  # (S, T, Ci)

        # sliding windows → (W, S, C, T)
        imu_wins    = imu_t.unfold(1, wd_imu, stride_imu).permute(1, 0, 2, 3)       # (W, S, Ci, Ti)
        motion_wins = shifted.unfold(1, wd_mot, stride_mot).permute(1, 0, 2, 3)     # (W, S, Cm, Tm)

        W = imu_wins.shape[0]
        assert imu_wins.shape[0] == motion_wins.shape[0], "Window count mismatch"

        with torch.no_grad():
            out = model({'imu': imu_wins.to(device), 'motion': motion_wins.to(device)},
                        global_weight=1.0, local_weight=0.0)
            imu_feats = out['cls_i']   # (W, D) — raw, no normalize
            mot_feats = out['cls_m']   # (W, D)

        sim = torch.mm(imu_feats, mot_feats.t())  # (W, W)

        _, imu_idx    = torch.topk(sim,     k=topk, dim=1)
        _, motion_idx = torch.topk(sim.t(), k=topk, dim=1)

        indices = torch.arange(W, device=sim.device).unsqueeze(1)
        offsets_r = -(imu_idx    - indices)
        offsets_c =  (motion_idx - indices)

        scores_r = torch.gather(sim,     1, imu_idx)
        scores_c = torch.gather(sim.t(), 1, motion_idx)

        all_offsets = torch.cat([offsets_r, offsets_c], dim=1).flatten().cpu().numpy().astype(np.int32)
        all_scores  = torch.cat([scores_r,  scores_c],  dim=1).flatten().cpu().numpy()

        unique_offsets, inv = np.unique(all_offsets, return_inverse=True)
        score_sums = np.bincount(inv, weights=all_scores)
        pred_ms = int(unique_offsets[np.argmax(score_sums)] * sampled_stride_s * 1000)
        pred_offsets.append(pred_ms)

    _print_results(np.array(gt_offsets), np.array(pred_offsets))


def eval_video_level(model, data_list, sync_dir, dataset_name, limb_list,
                     wd_imu, stride_imu, wd_mot, stride_mot, sampled_stride_s, topk, device):
    gt_offsets, pred_offsets = [], []

    for data in tqdm(data_list, desc='Video-level sync'):
        file_idx, w_s, w_e, subject, action, offset_ms = data
        gt_offsets.append(offset_ms)

        imu      = np.load(os.path.join(sync_dir, 'imu',           f"{file_idx:05d}.npy"))
        pose2d   = np.load(os.path.join(sync_dir, 'pose2d',        f"{file_idx:05d}.npy"))
        pose2d_s = np.load(os.path.join(sync_dir, 'pose2d_shifted', f"{file_idx:05d}.npy"))

        pose2d_norm   = normalize_pose(pose2d,   dataset_name)
        pose2d_s_norm = normalize_pose(pose2d_s, dataset_name)

        # motion: (S, P, T, C, J) → (P, T, S, Cm)
        motion_list  = [extract_limb_motion(pose2d_norm,   dataset_name, l) for l in limb_list]
        shifted_list = [extract_limb_motion(pose2d_s_norm, dataset_name, l) for l in limb_list]
        motion  = torch.tensor(np.stack(motion_list,  axis=0)).float()  # (S, P, T, C, J)
        shifted = torch.tensor(np.stack(shifted_list, axis=0)).float()  # (S, P, T, C, J)
        S, P, T, C, J = motion.shape
        Cm = C * J
        motion  = motion.reshape(S, P, T, Cm).permute(1, 2, 0, 3).contiguous()   # (P, T, S, Cm)
        shifted = shifted.reshape(S, P, T, Cm).permute(1, 2, 0, 3).contiguous()  # (P, T, S, Cm)

        # imu: (P, T, S, Ci)
        imu_t = torch.tensor(imu).float()
        assert imu_t.dim() == 4, f"Expected IMU 4D (P,T,S,Ci), got {imu_t.shape}"

        # sliding windows → (P*W, S, C, T)
        imu_wins    = imu_t.unfold(1, wd_imu, stride_imu)    # (P, W, S, Ci, Ti)
        motion_wins = shifted.unfold(1, wd_mot, stride_mot)  # (P, W, S, Cm, Tm)

        W = imu_wins.shape[1]
        assert imu_wins.shape[1] == motion_wins.shape[1], "Window count mismatch"

        imu_wins    = imu_wins.contiguous().view(P * W, S, imu_wins.shape[3],    wd_imu)
        motion_wins = motion_wins.contiguous().view(P * W, S, motion_wins.shape[3], wd_mot)

        with torch.no_grad():
            out = model({'imu': imu_wins.to(device), 'motion': motion_wins.to(device)},
                        global_weight=1.0, local_weight=0.0)
            imu_feats = F.normalize(out['cls_i'], dim=-1)  # (P*W, D)
            mot_feats = F.normalize(out['cls_m'], dim=-1)  # (P*W, D)

        sim = torch.mm(imu_feats, mot_feats.t()).view(P, W, P, W)

        all_offsets, all_scores = [], []
        for p in range(P):
            block = sim[p, :, p, :]  # (W, W)

            topk_val_r, topk_idx_r = torch.topk(block,     k=topk, dim=1)
            topk_val_c, topk_idx_c = torch.topk(block.t(), k=topk, dim=1)

            idx = torch.arange(W, device=block.device).unsqueeze(1)
            offsets_r = -(topk_idx_r - idx)
            offsets_c =  (topk_idx_c - idx)

            offsets = torch.cat([offsets_r, offsets_c], dim=1).reshape(-1).cpu().numpy().astype(np.int32)
            scores  = torch.cat([topk_val_r, topk_val_c], dim=1).reshape(-1).cpu().numpy()
            all_offsets.append(offsets)
            all_scores.append(scores)

        all_offsets = np.concatenate(all_offsets)
        all_scores  = np.concatenate(all_scores)

        unique_offsets, inv = np.unique(all_offsets, return_inverse=True)
        score_sums = np.bincount(inv, weights=all_scores)
        pred_ms = int(unique_offsets[np.argmax(score_sums)] * sampled_stride_s * 1000)
        pred_offsets.append(pred_ms)

    _print_results(np.array(gt_offsets), np.array(pred_offsets))


def _print_results(gt, pred):
    err = np.abs(gt - pred)
    print(f"\nMAE:     {err.mean()/1000:.4f}s")
    print(f"Acc@0.1: {np.mean(err < 100):.4f}")
    print(f"Acc@0.2: {np.mean(err < 200):.4f}")
    print(f"Acc@0.5: {np.mean(err < 500):.4f}")


def main():
    parser = ArgumentParser(description='Sync evaluation for EgoHumans')
    parser.add_argument('--exp_dir',   default='./checkpoints/EgoHumans/stage2', help='Path to experiment directory')
    parser.add_argument('--dataset',    default='EgoHumans')
    parser.add_argument('--task',       choices=['person', 'video'], required=True,
                        help='person: single-person sync, video: multi-person video-level sync')
    parser.add_argument('--window_sec', type=int, default=20)
    parser.add_argument('--stride_sec', type=int, default=5)
    parser.add_argument('--topk',       type=int, default=5)
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.exp_dir, 'config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sync_dir = os.path.join(DATA_ROOT, args.dataset, f'cache_sync_{config.data.split}_{args.window_sec}_{args.stride_sec}')

    model = build_model(config)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, 'checkpoints/best.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M params | Device: {device}")

    limb_list        = DATASET_CONFIG[args.dataset]['limb_list']
    target_imu_srate = config.data.imu_srate
    target_mot_srate = config.data.motion_srate
    sampled_wd_sec   = config.data.window_sec
    sampled_stride_s = 0.2

    wd_imu    = int(sampled_wd_sec * target_imu_srate)
    stride_imu = int(sampled_stride_s * target_imu_srate)
    wd_mot    = int(sampled_wd_sec * target_mot_srate)
    stride_mot = int(sampled_stride_s * target_mot_srate)

    print(f"IMU window: {wd_imu} frames, stride: {stride_imu}")
    print(f"Motion window: {wd_mot} frames, stride: {stride_mot}")

    data_list = get_sync_annotations(args.dataset, sync_dir, 'test')
    print(f"Test samples: {len(data_list)}")

    if args.task == 'person':
        eval_person_level(model, data_list, sync_dir, args.dataset, limb_list,
                          wd_imu, stride_imu, wd_mot, stride_mot, sampled_stride_s, args.topk, device)
    else:
        eval_video_level(model, data_list, sync_dir, args.dataset, limb_list,
                         wd_imu, stride_imu, wd_mot, stride_mot, sampled_stride_s, args.topk, device)


if __name__ == '__main__':
    main()