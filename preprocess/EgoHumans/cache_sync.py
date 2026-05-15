import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import argparse
import random
from collections import defaultdict
from tqdm import tqdm

from datasets.utils import get_windows_in_clip
from configs.config import DATA_ROOT, DATASET_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Build sync cache for EgoHumans')
    parser.add_argument('--window_sec', type=int, default=20)
    parser.add_argument('--stride_sec', type=int, default=5)
    parser.add_argument('--anno_file',  type=str, default=None,
                        help='Path to an existing annotations.txt to reproduce exactly; '
                             'if omitted, offsets are sampled randomly')
    args = parser.parse_args()

    dataset_name = 'EgoHumans'
    dataset_config = DATASET_CONFIG[dataset_name]
    motion_sample_rate = dataset_config['motion_sample_rate']

    data_dir = os.path.join(DATA_ROOT, dataset_name, 'extracted_data')

    save_dir = os.path.join(DATA_ROOT, dataset_name, f'cache_sync_action_{args.window_sec}_{args.stride_sec}')
    os.makedirs(os.path.join(save_dir, 'imu'),           exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pose2d'),        exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pose2d_shifted'), exist_ok=True)

    all_seq_files = sorted(os.listdir(data_dir))
    anno_list = []

    def _load_sequence(seq_key):
        seq_files = [s for s in all_seq_files if s.startswith(seq_key)]
        seq_imu, seq_pose2d = [], []
        for seq_file in seq_files:
            data = np.load(os.path.join(data_dir, seq_file), allow_pickle=True).item()
            seq_imu.append(data['imu'])
            seq_pose2d.append(data['pose2d'])
        imu_data   = np.array(seq_imu).transpose(1, 0, 2, 3)    # (T, P, N, C)
        pose2d_data = np.array(seq_pose2d).transpose(1, 0, 2, 3) # (T, P, 17, 2)
        return imu_data, pose2d_data

    def _save_entry(fidx, w_s, w_e, seq_idx, act_idx, offset, imu_data, pose2d_data):
        sf       = int(w_s * motion_sample_rate)
        ef       = int(w_e * motion_sample_rate)
        shift_sf = sf + int(offset * motion_sample_rate / 1000)
        shift_ef = ef + int(offset * motion_sample_rate / 1000)

        np.save(os.path.join(save_dir, 'imu',           f'{fidx:05d}.npy'),
                imu_data[sf:ef].transpose(1, 0, 2, 3))           # (P, T_w, N, C)
        np.save(os.path.join(save_dir, 'pose2d',        f'{fidx:05d}.npy'),
                pose2d_data[sf:ef].transpose(1, 0, 3, 2))        # (P, T_w, 2, 17)
        np.save(os.path.join(save_dir, 'pose2d_shifted', f'{fidx:05d}.npy'),
                pose2d_data[shift_sf:shift_ef].transpose(1, 0, 3, 2))
        anno_list.append((fidx, w_s, w_e, seq_idx, act_idx, offset))

    if args.anno_file:
        entries_by_seq = defaultdict(list)
        with open(args.anno_file) as f:
            for line in f:
                fidx, w_s, w_e, seq_idx, act_idx, offset = line.strip().split()
                seq_key = f"{act_idx}_{seq_idx}"
                entries_by_seq[seq_key].append(
                    (int(fidx), int(w_s), int(w_e), seq_idx, act_idx, int(offset))
                )

        for seq_key in tqdm(sorted(entries_by_seq.keys()), desc='Sequences'):
            imu_data, pose2d_data = _load_sequence(seq_key)
            for entry in entries_by_seq[seq_key]:
                _save_entry(*entry, imu_data, pose2d_data)

        anno_list.sort(key=lambda x: x[0])

    else:
        splits = dataset_config['split']['action']
        all_sequences = splits['train'] + splits['val'] + splits['test']
        file_idx = 0

        for seq_key in tqdm(all_sequences, desc='Sequences'):
            act_idx, seq_idx = seq_key.split('_')
            imu_data, pose2d_data = _load_sequence(seq_key)

            end_frame  = min(len(pose2d_data), len(imu_data))
            end_sec    = end_frame / motion_sample_rate
            start_sec  = 0
            windows    = get_windows_in_clip(
                s_time=int(start_sec), e_time=int(end_sec),
                window_sec=args.window_sec, stride=args.stride_sec,
            )

            for w_s, w_e in windows:
                for _ in range(10):
                    end_ms   = int(w_e) * 1000
                    start_ms = int(w_s) * 1000
                    offset   = random.randint(-70, 70) * 100
                    if offset > 0:
                        offset = min(offset, int(end_sec * 1000) - end_ms)
                    else:
                        offset = max(offset, int(start_sec * 1000) - start_ms)

                    _save_entry(file_idx, w_s, w_e, seq_idx, act_idx, offset, imu_data, pose2d_data)
                    file_idx += 1

    anno_path = os.path.join(save_dir, 'annotations.txt')
    with open(anno_path, 'w') as f:
        for entry in anno_list:
            f.write(' '.join(map(str, entry)) + '\n')
    print(f"Saved {len(anno_list)} entries to {anno_path}")


if __name__ == '__main__':
    main()
