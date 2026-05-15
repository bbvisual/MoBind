import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import argparse
import json
from tqdm import tqdm

from datasets.utils import get_windows_in_clip
from configs.config import DATA_ROOT, DATASET_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Cache EgoHumans multi-person dataset')
    parser.add_argument('--window_sec', type=int, default=5)
    parser.add_argument('--stride',     type=int, default=2)
    parser.add_argument('--split',      type=str, default='action')
    args = parser.parse_args()

    dataset_name = 'EgoHumans'
    dataset_config = DATASET_CONFIG[dataset_name]
    imu_srate    = dataset_config['imu_sample_rate']
    motion_srate = dataset_config['motion_sample_rate']
    data_split   = dataset_config['split'][args.split]['test']

    data_dir = os.path.join(DATA_ROOT, dataset_name, 'extracted_data')
    save_dir = os.path.join(DATA_ROOT, dataset_name, f'cache_{args.split}_multi_{args.window_sec}_{args.stride}')
    os.makedirs(os.path.join(save_dir, 'data/imu'),    exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'data/pose2d'), exist_ok=True)

    meta_dict = {
        'window_sec':   args.window_sec,
        'stride':       args.stride,
        'imu_srate':    imu_srate,
        'motion_srate': motion_srate,
        'meta': {}
    }

    seq_list = sorted(os.listdir(data_dir))
    file_idx = 0

    for seq in tqdm(data_split, desc='Processing sequences'):
        act_idx, seq_idx = seq.split('_')
        seq_files = [s for s in seq_list if s.startswith(seq)]

        all_imu, all_pose2d = [], []
        for seq_file in seq_files:
            data = np.load(os.path.join(data_dir, seq_file), allow_pickle=True).item()
            assert len(data['imu']) == len(data['pose2d'])
            all_imu.append(data['imu'])
            all_pose2d.append(data['pose2d'])

        imu_data    = np.array(all_imu).transpose(1, 0, 2, 3)    # (T, P, N, C)
        pose2d_data = np.array(all_pose2d).transpose(1, 0, 2, 3) # (T, P, 17, 2)

        windows = get_windows_in_clip(
            s_time=0, e_time=len(pose2d_data),
            window_sec=args.window_sec * motion_srate,
            stride=args.stride * motion_srate,
        )

        for w_s, w_e in windows:
            pose2d_signal = pose2d_data[w_s:w_e].transpose(0, 1, 3, 2)  # (T, P, 17, 2) → (T, P, 2, 17)
            np.save(os.path.join(save_dir, f'data/pose2d/{file_idx:05d}.npy'), pose2d_signal)
            np.save(os.path.join(save_dir, f'data/imu/{file_idx:05d}.npy'),    imu_data[w_s:w_e])

            meta_dict['meta'][f'{file_idx:05d}'] = {
                'action':   act_idx,
                'sequence': seq_idx,
                'window_s': (w_s // motion_srate, w_e // motion_srate),
                'window_f': (w_s, w_e),
                'phase':    'test',
            }
            file_idx += 1

    print(f'Files: {file_idx}')
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f, indent=4)
    print(f'Done. Cache saved to {save_dir}')


if __name__ == '__main__':
    main()
