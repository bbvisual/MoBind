import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pickle
import argparse
import json
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

from datasets.utils import get_windows_in_clip, preprocess_imu_window
from configs.config import DATA_ROOT, DATASET_CONFIG


def get_action_label(ws, we, action_labels):
    max_overlap = 0
    matched_label = 'null'
    for label, (start, end) in action_labels.items():
        overlap = max(0, min(we, end) - max(ws, start))
        if overlap > max_overlap:
            max_overlap = overlap
            matched_label = label
    return matched_label


def main():
    parser = argparse.ArgumentParser(description='Cache mRi dataset')
    parser.add_argument('--window_sec', type=int, default=5)
    parser.add_argument('--stride',     type=int, default=2)
    parser.add_argument('--split',      type=str, default='subject')
    args = parser.parse_args()

    dataset_name = 'mRi'
    dataset_config = DATASET_CONFIG[dataset_name]
    imu_srate    = dataset_config['imu_sample_rate']
    motion_srate = dataset_config['motion_sample_rate']
    limb_list    = dataset_config['limb_list']
    data_split   = dataset_config['split'][args.split]

    data_dir  = os.path.join(DATA_ROOT, dataset_name)
    label_dir = os.path.join(data_dir, 'dataset_release/aligned_data/pose_labels')
    imu_dir   = os.path.join(data_dir, 'processed_data/imu')
    pose_dir  = os.path.join(data_dir, 'processed_data/pose2d_mmpose')

    save_dir = os.path.join(DATA_ROOT, dataset_name, f'cache_{args.split}_{args.window_sec}_{args.stride}')
    os.makedirs(os.path.join(save_dir, 'data/imu'),       exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'data/pose2d'),    exist_ok=True)

    meta_dict = {
        'window_sec': args.window_sec,
        'stride':     args.stride,
        'imu_srate':  imu_srate,
        'motion_srate': motion_srate,
        'meta': {}
    }

    file_idx = 0
    skip_cnt = 0

    for sub in tqdm(range(1, 21), desc='Processing subjects'):
        label_file = os.path.join(label_dir, f"subject{sub}_all_labels.cpl")
        anno = pickle.load(open(label_file, 'rb'))
        action_labels = anno['video_label']

        first_start = min(s for s, _ in action_labels.values())
        action_labels = {a: [s - first_start, e - first_start] for a, (s, e) in action_labels.items()}

        imu = {
            limb: pd.read_csv(os.path.join(imu_dir, f'subject{sub}/{limb}.csv'))[
                ["axg", "ayg", "azg", 'q0', 'q1', 'q2', 'q3']].values
            for limb in limb_list
        }
        pose2d     = np.load(os.path.join(pose_dir,     f'subject{sub}.npy'))
        windows = get_windows_in_clip(
            s_time=0, e_time=len(pose2d),
            window_sec=args.window_sec * motion_srate,
            stride=args.stride * motion_srate,
        )

        for w_s, w_e in windows:
            if any(len(imu[l][w_s:w_e]) < 50 for l in limb_list) or len(pose2d[w_s:w_e]) < 50:
                skip_cnt += 1
                continue

            action = get_action_label(w_s, w_e, action_labels)
            action_idx = DATASET_CONFIG[dataset_name]['action2idx'][action]
            if action_idx == -1:
                skip_cnt += 1
                continue

            if str(sub) in data_split['train']:
                phase = 'train'
            elif str(sub) in data_split['val']:
                phase = 'val'
            else:
                phase = 'test'
            
            for limb in limb_list:
                signal = preprocess_imu_window(imu[limb][w_s:w_e], args.window_sec, imu_srate)
                np.save(os.path.join(save_dir, f'data/imu/{file_idx:05d}_{limb}.npy'), signal)

            meta_dict['meta'][f'{file_idx:05d}'] = {
                'subject':  str(sub),
                'action':   action_idx,
                'window_s': (w_s // motion_srate, w_e // motion_srate),
                'window_f': (w_s, w_e),
                'phase':    phase,
            }
            file_idx += 1

    print(f'Files: {file_idx} | Skipped: {skip_cnt}')
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f, indent=4)
    print(f'Done. Cache saved to {save_dir}')


if __name__ == '__main__':
    main()
