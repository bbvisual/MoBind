import os
import json
import numpy as np
import torch
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from omegaconf import OmegaConf
from builder import build_model
from datasets.utils import normalize_pose, extract_limb_motion
from configs.config import DATASET_CONFIG, DATA_ROOT


def evaluate_person_localization(model, data_list, dataset_name, limb_list, device):
    perP = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct, overall_total = 0, 0

    for _, imu_path, pose_path in tqdm(data_list, desc='Person localization'):
        imu_signal = np.load(imu_path)      # (T, P, N, C)
        pose_signal = np.load(pose_path)    # (T, P, 2, J)
        pose_signal = normalize_pose(pose_signal, dataset_name)

        motion_signal = np.stack(
            [extract_limb_motion(pose_signal, dataset_name, limb) for limb in limb_list], axis=0
        )  # (S, T, P, 2, J)
        motion_signal = torch.tensor(motion_signal).float()
        S, T, P, C, J = motion_signal.shape
        motion_signal = motion_signal.reshape(S, T, P, C * J).permute(2, 0, 3, 1)  # (P, S, CJ, T)
        imu_signal = torch.tensor(imu_signal.transpose(1, 2, 3, 0)).float()        # (P, N, C, T)

        with torch.no_grad():
            out = model({'imu': imu_signal.to(device), 'motion': motion_signal.to(device)},
                        global_weight=1.0, local_weight=0.0)
            imu_feats = out['cls_i']   # (P, D)
            mot_feats = out['cls_m']   # (P, D)

        pred = (imu_feats @ mot_feats.t()).argmax(dim=1)
        gt = torch.arange(P, device=pred.device)
        correct = (pred == gt).sum().item()

        perP[P]['correct'] += correct
        perP[P]['total'] += P
        overall_correct += correct
        overall_total += P

    print("\nPerson Localization Accuracy by #People:")
    print(f"{'P':<5} {'Clips':>8} {'Accuracy':>10}")
    print("-" * 25)
    for P_val in sorted(perP.keys()):
        d = perP[P_val]
        acc = d['correct'] / d['total'] * 100
        print(f"{P_val:<5} {d['total']//P_val:>8} {acc:>9.2f}%")
    print(f"\nOverall: {overall_correct / overall_total * 100:.2f}%\n")


def evaluate_limb_localization(imu_encoder, motion_encoder, model, data_list, dataset_name, limb_list, device):
    S = len(limb_list)
    overall_correct, overall_total = 0, 0

    for _, imu_path, pose_path in tqdm(data_list, desc='Limb localization'):
        imu_np   = np.load(imu_path)    # (T, P, N, C)
        pose_np  = np.load(pose_path)   # (T, P, 2, J)
        pose_np  = normalize_pose(pose_np, dataset_name)

        motion_np = np.stack(
            [extract_limb_motion(pose_np, dataset_name, limb) for limb in limb_list], axis=0
        )  # (S, T, P, 2, J)
        motion = torch.tensor(motion_np).float().to(device)
        S_check, T, P, C2, J = motion.shape
        motion = motion.reshape(S_check, T, P, C2 * J).permute(2, 0, 3, 1)  # (P, S, CJ, T)
        imu = torch.tensor(imu_np.transpose(1, 2, 3, 0)).float().to(device)  # (P, N, C, T)

        # find correctly matched persons using stage2 global features
        with torch.no_grad():
            out = model({'imu': imu, 'motion': motion}, global_weight=1.0, local_weight=0.0)
            imu_p = out['cls_i']
            mot_p = out['cls_m']

        correct_persons = ((imu_p @ mot_p.t()).argmax(dim=1) == torch.arange(P, device=imu_p.device))
        idx_keep = correct_persons.nonzero(as_tuple=False).flatten()
        if idx_keep.numel() == 0:
            continue

        imu_sel    = imu[idx_keep]     # (Bcorr, N, C, T)
        motion_sel = motion[idx_keep]  # (Bcorr, S, CJ, T)
        Bcorr = imu_sel.size(0)
        Cim, Ti = imu_sel.shape[2], imu_sel.shape[3]
        CJ,  Tm = motion_sel.shape[2], motion_sel.shape[3]

        # use stage1 encoders for limb-level features
        with torch.no_grad():
            mot_feat, _ = motion_encoder(motion_sel.reshape(Bcorr * S, CJ, Tm))
            imu_feat, _ = imu_encoder(imu_sel.reshape(Bcorr * S, Cim, Ti))

        mot_feat = mot_feat.view(Bcorr, S, -1)
        imu_feat = imu_feat.view(Bcorr, S, -1)

        pred_limbs = torch.einsum('bsd,btd->bst', imu_feat, mot_feat).argmax(dim=2)  # (Bcorr, S)
        gt_limbs   = torch.arange(S, device=pred_limbs.device).unsqueeze(0).expand(Bcorr, -1)

        overall_correct += (pred_limbs == gt_limbs).sum().item()
        overall_total   += Bcorr * S

    if overall_total == 0:
        print("No correctly-matched persons found.")
        return

    acc = 100.0 * overall_correct / overall_total
    print(f"\nLimb Localization Accuracy (conditioned on correct person): {acc:.2f}%")


def main():
    parser = ArgumentParser(description='Evaluate person and limb localization on EgoHumans')
    parser.add_argument('--exp_path', default='./checkpoints/EgoHumans/stage2', help='Path to stage2 experiment directory')
    parser.add_argument('--dataset',   default='EgoHumans')
    parser.add_argument('--task',      choices=['person', 'limb', 'all'], default='all')
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.exp_path, 'config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache_name = f"cache_{config.data.split}_multi_{config.data.window_sec}_{config.data.stride_sec}"
    data_dir = os.path.join(DATA_ROOT, config.data.dataset_name, cache_name)

    model = build_model(config)
    model.load_state_dict(torch.load(os.path.join(args.exp_path, 'checkpoints/best.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M params | Device: {device}")

    limb_list = DATASET_CONFIG[args.dataset]['limb_list']

    with open(os.path.join(data_dir, 'meta.json')) as f:
        meta_data = json.load(f)

    data_list = [
        (idx,
         os.path.join(data_dir, 'data/imu',   f"{idx}.npy"),
         os.path.join(data_dir, 'data/pose2d', f"{idx}.npy"))
        for idx in sorted(meta_data['meta'].keys())
        if meta_data['meta'][idx]['phase'] == 'test'
    ]
    print(f"Test samples: {len(data_list)}")

    if args.task in ('person', 'all'):
        evaluate_person_localization(model, data_list, args.dataset, limb_list, device)

    if args.task in ('limb', 'all'):
        stage1_path = config.model.stage1_exp
        stage1_config = OmegaConf.load(os.path.join(stage1_path, 'config.yaml'))
        stage1_model = build_model(stage1_config)
        stage1_model.load_state_dict(
            torch.load(os.path.join(stage1_path, 'checkpoints/best.pt'), map_location=device)
        )
        stage1_model = stage1_model.to(device)
        stage1_model.eval()
        imu_encoder    = stage1_model.imu_encoder
        motion_encoder = stage1_model.motion_encoder
        evaluate_limb_localization(imu_encoder, motion_encoder, model, data_list, args.dataset, limb_list, device)


if __name__ == '__main__':
    main()
