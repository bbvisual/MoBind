import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import argparse
import random
from collections import defaultdict
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from datasets.utils import get_windows_in_clip
from configs.config import DATA_ROOT, DATASET_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Build sync cache for mRi')
    parser.add_argument('--window_sec', type=int, default=20)
    parser.add_argument('--stride_sec', type=int, default=5)
    parser.add_argument('--anno_file',  type=str, default=None,
                        help='Path to an existing annotations.txt to reproduce exactly; '
                             'if omitted, offsets are sampled randomly')
    args = parser.parse_args()

    dataset_name = 'mRi'
    dataset_config = DATASET_CONFIG[dataset_name]
    imu_sample_rate = dataset_config['imu_sample_rate']
    motion_sample_rate = dataset_config['motion_sample_rate']
    limb_list = dataset_config['limb_list']

    imu_dir  = os.path.join(DATA_ROOT, dataset_name, 'processed_data/imu')
    pose_dir = os.path.join(DATA_ROOT, dataset_name, 'processed_data/pose2d_mmpose')

    save_dir = os.path.join(DATA_ROOT, dataset_name, f'cache_sync_subject_{args.window_sec}_{args.stride_sec}')
    os.makedirs(os.path.join(save_dir, 'imu'),           exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pose2d'),        exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pose2d_shifted'), exist_ok=True)

    anno_list = []

    def _load_subject(sub):
        imu_signals = []
        for limb in limb_list:
            limb_path = os.path.join(imu_dir, f'subject{sub}', f'{limb}.csv')
            if not os.path.exists(limb_path):
                raise FileNotFoundError(f"IMU data not found: {limb_path}")
            imu_signals.append(pd.read_csv(limb_path)[['axg', 'ayg', 'azg', 'q0', 'q1', 'q2', 'q3']].values)
        imu = np.stack(imu_signals, axis=0).transpose(1, 0, 2)   # (T, num_limb, feat_dim)
        pose = np.load(os.path.join(pose_dir, f'subject{sub}.npy'))
        return imu, pose

    def _save_entry(fidx, w_s, w_e, sub_str, action_str, offset, imu_signals, pose_signals):
        sf       = int(w_s * motion_sample_rate)
        ef       = int(w_e * motion_sample_rate)
        shift_sf = sf + int(offset * motion_sample_rate / 1000)
        shift_ef = ef + int(offset * motion_sample_rate / 1000)

        np.save(os.path.join(save_dir, 'imu',           f'{fidx:05d}.npy'),
                resample(imu_signals[sf:ef], int(args.window_sec * imu_sample_rate)))
        np.save(os.path.join(save_dir, 'pose2d',        f'{fidx:05d}.npy'), pose_signals[sf:ef])
        np.save(os.path.join(save_dir, 'pose2d_shifted', f'{fidx:05d}.npy'), pose_signals[shift_sf:shift_ef])
        anno_list.append((fidx, w_s, w_e, sub_str, action_str, offset))

    if args.anno_file:
        entries_by_sub = defaultdict(list)
        with open(args.anno_file) as f:
            for line in f:
                fidx, w_s, w_e, sub_str, action_str, offset = line.strip().split()
                entries_by_sub[sub_str].append(
                    (int(fidx), int(w_s), int(w_e), sub_str, action_str, int(offset))
                )

        for sub_str in tqdm(sorted(entries_by_sub.keys()), desc='Subjects'):
            imu_signals, pose_signals = _load_subject(int(sub_str))
            for entry in entries_by_sub[sub_str]:
                _save_entry(*entry, imu_signals, pose_signals)

        anno_list.sort(key=lambda x: x[0])

    else:
        splits = dataset_config['split']['subject']
        all_subjects = splits['train'] + splits['val'] + splits['test']
        file_idx = 0

        for sub_str in tqdm(all_subjects, desc='Subjects'):
            imu_signals, pose_signals = _load_subject(int(sub_str))

            end_sec   = len(pose_signals) / motion_sample_rate
            start_sec = 0
            windows   = get_windows_in_clip(
                s_time=int(start_sec), e_time=int(end_sec),
                window_sec=args.window_sec, stride=args.stride_sec,
            )

            for w_s, w_e in windows:
                for _ in range(3):
                    end_ms   = int(w_e) * 1000
                    start_ms = int(w_s) * 1000
                    offset   = random.randint(-70, 70) * 100
                    if offset > 0:
                        offset = min(offset, int(end_sec * 1000) - end_ms)
                    else:
                        offset = max(offset, int(start_sec * 1000) - start_ms)

                    _save_entry(file_idx, w_s, w_e, sub_str, 'None', offset, imu_signals, pose_signals)
                    file_idx += 1

    anno_path = os.path.join(save_dir, 'annotations.txt')
    with open(anno_path, 'w') as f:
        for entry in anno_list:
            f.write(' '.join(map(str, entry)) + '\n')
    print(f"Saved {len(anno_list)} entries to {anno_path}")


if __name__ == '__main__':
    main()
