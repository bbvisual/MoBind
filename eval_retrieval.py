import os
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pprint import pprint
from builder import build_model, build_dataloader
from utils.train import get_clip_metrics, get_clip_metrics_multi_pos
from collections import defaultdict


def print_metrics_results(metrics):
    """
    Show IMU→Motion and Motion→IMU retrieval scores in one table.

    Expected keys in `metrics`:
        imu_to_motion_R@1,  motion_to_imu_R@1,  …
        imu_to_motion_mean_rank, motion_to_imu_mean_rank, …
    """
    rows = defaultdict(dict)
    for k, v in metrics.items():
        if k.startswith("imu_to_motion_"):
            rows[k.replace("imu_to_motion_", "")]["imu→motion"] = v
        elif k.startswith("motion_to_imu_"):
            rows[k.replace("motion_to_imu_", "")]["motion→imu"] = v

    def sort_key(label):
        if label.startswith("R@"):
            return (0, int(label.split("@")[1]))
        elif label == "mean_rank":
            return (1, 0)
        elif label == "median_rank":
            return (2, 0)
        else:
            return (3, 0)

    print(f"{'Metric':<15} {'IMU → Video':>12} {'Motion → IMU':>12}")
    print("-" * 41)
    for label in sorted(rows.keys(), key=sort_key):
        imu2vid = rows[label].get("imu→motion", float("nan"))
        vid2imu = rows[label].get("motion→imu", float("nan"))
        print(f"{label:<15} {imu2vid:>12.4f} {vid2imu:>12.4f}")


def main():
    parser = ArgumentParser(description='Evaluate retrieval')
    parser.add_argument('--exp_dir', required=True, help='Path to experiment directory')
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    ckpt_path = os.path.join(args.exp_dir, 'checkpoints/best.pt')
    config = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#"*50)
    pprint(config)
    print("#"*50)

    model = build_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}'))
    model.eval()

    print(f"Testing on dataset: {config.data.dataset_name}")
    _, _, test_loader = build_dataloader(config)

    all_imus, all_motions = [], []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_imus = batch['imu'].to(device)
            input_motions = batch['motion'].to(device)

            inputs = {'imu': input_imus, 'motion': input_motions}
            outputs = model(inputs, global_weight=1.0, local_weight=0.0)
            imu_feats = outputs['cls_i']
            motion_feats = outputs['cls_m']

            all_imus.append(imu_feats.cpu())
            all_motions.append(motion_feats.cpu())

    k_list = [1, 3, 5, 10, 25, 50]
    metrics = get_clip_metrics(
        imu_features=torch.cat(all_imus),
        motion_features=torch.cat(all_motions),
        k_list=k_list
    ) if config.experiment.name in ['stage1', 'stage2', 'mae'] else get_clip_metrics_multi_pos(
        imu_features=torch.cat(all_imus),
        motion_features=torch.cat(all_motions),
        k_list=k_list
    )
    print_metrics_results(metrics)


if __name__ == "__main__":
    main()
