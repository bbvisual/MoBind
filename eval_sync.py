import os
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from builder import build_model
from datasets.dataset import SyncDataset
from configs.config import DATASET_CONFIG, DATA_ROOT


def main():
    parser = ArgumentParser(description='Evaluate Synchronization task')
    parser.add_argument('--exp_dir', required=True, help='Path to experiment directory')
    parser.add_argument('--test_dataset', default='mRi', help='Test dataset name')
    parser.add_argument('--window_sec', type=int, default=20)
    parser.add_argument('--stride_sec', type=int, default=5)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    ckpt_path = os.path.join(args.exp_dir, 'checkpoints/best.pt')
    config = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#"*50)
    print(device)
    print(config)
    print("#"*50)

    model = build_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}'))
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.3f} M")

    target_imu_srate = config.data.imu_srate
    target_motion_srate = config.data.motion_srate

    window_sec = args.window_sec
    stride_sec = args.stride_sec
    motion_type = config.data.motion_type

    data_root = DATA_ROOT

    sync_dir = os.path.join(data_root, args.test_dataset, f'cache_sync_subject_{window_sec}_{stride_sec}')
    limb_list = DATASET_CONFIG[args.test_dataset]['limb_list']
    test_dataset = SyncDataset(
        dataset_name=args.test_dataset,
        sync_dir=sync_dir,
        phase='test',
        motion_type=motion_type,
        limb_list=limb_list,
    )
    print('Number of test sample:', len(test_dataset))
    print('Topk:', args.topk)
    print('Window size:', window_sec, 's')

    sampled_wd_sec = config.data.window_sec
    sampled_stride_sec = 0.2

    sampled_wd_imu = int(sampled_wd_sec * target_imu_srate)
    sampled_wd_imu_stride = int(sampled_stride_sec * target_imu_srate)
    sampled_wd_motion = int(sampled_wd_sec * target_motion_srate)
    sampled_wd_motion_stride = int(sampled_stride_sec * target_motion_srate)

    print(f"Sampled window size: {sampled_wd_imu}, stride: {sampled_wd_imu_stride}")
    print(f"Sampled motion window size: {sampled_wd_motion}, stride: {sampled_wd_motion_stride}")

    gt_offsets = []
    pred_offsets = []
    topk = args.topk

    for idx in tqdm(range(len(test_dataset))):
        imu_signal = test_dataset[idx]['imu'].to(device)
        shifted_motion_signal = test_dataset[idx]['shifted_motion'].to(device)

        imu_wins   = imu_signal.unfold(1, sampled_wd_imu, sampled_wd_imu_stride).permute(1, 0, 2, 3)
        motion_wins = shifted_motion_signal.unfold(1, sampled_wd_motion, sampled_wd_motion_stride).permute(1, 0, 2, 3)

        if imu_wins.shape[0] != motion_wins.shape[0]:
            continue

        gt_offset = test_dataset[idx]['offset']
        gt_offsets.append(gt_offset)
        inputs = {'imu': imu_wins, 'motion': motion_wins}

        with torch.no_grad():
            outputs = model(inputs, global_weight=1.0, local_weight=0.0)
            imu_feats = outputs['cls_i']
            motion_feats = outputs['cls_m']

        imu_simi = torch.mm(imu_feats, motion_feats.t())
        _, imu_idx = torch.topk(imu_simi, k=topk)
        _, motion_idx = torch.topk(imu_simi.t(), k=topk)

        indices = torch.arange(imu_idx.shape[0], device=imu_idx.device).unsqueeze(1)
        imu_offsets = -(imu_idx - indices)
        motion_offsets = motion_idx - indices

        imu_scores = torch.gather(imu_simi, 1, imu_idx)
        motion_scores = torch.gather(imu_simi.t(), 1, motion_idx)

        all_offsets = torch.cat([imu_offsets, motion_offsets], dim=1).flatten().cpu().numpy()
        all_scores = torch.cat([imu_scores, motion_scores], dim=1).flatten().cpu().numpy()
        all_offsets_int = all_offsets.astype(np.int32)

        unique_offsets, inverse_indices = np.unique(all_offsets_int, return_inverse=True)
        score_sums = np.bincount(inverse_indices, weights=all_scores)

        pred_offset = unique_offsets[np.argmax(score_sums)]
        pred = int(pred_offset * 0.2 * 1000)
        pred_offsets.append(pred)

    gt = np.array(gt_offsets)
    pred = np.array(pred_offsets)

    abs_error = np.abs(gt - pred)
    mae = np.mean(abs_error)
    acc_10 = np.mean(abs_error < 100)
    acc_20 = np.mean(abs_error < 200)
    acc_50 = np.mean(abs_error < 500)
    acc_100 = np.mean(abs_error < 750)
    print(f"MAE: {mae/1000:.4f}s")
    print(f"Acc@0.1: {acc_10:.4f}%")
    print(f"Acc@0.2: {acc_20:.4f}%")
    print(f"Acc@0.5: {acc_50:.4f}%")
    print(f"Acc@0.75: {acc_100:.4f}%")


if __name__ == "__main__":
    main()
