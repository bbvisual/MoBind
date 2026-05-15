import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import torch
import pickle
import argparse
import json
import shutil
import cv2
from tqdm import tqdm
import pandas as pd
from configs.config import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description='Preprocessing mRi dataset')
    parser.add_argument('--data_dir', default=os.path.join(DATA_ROOT, 'mRi'), type=str, help='Path to the mRi dataset directory')
    args = parser.parse_args()

    save_dir = os.path.join(args.data_dir, 'processed_data')
    mmpose_dir = os.path.join(args.data_dir, 'extracted_pose')
    out_imu_dir = os.path.join(save_dir, 'imu')
    out_pose_mmpose_dir = os.path.join(save_dir, 'pose2d_mmpose')
    
    os.makedirs(out_pose_mmpose_dir, exist_ok=True)
    os.makedirs(out_imu_dir, exist_ok=True)   

    for sub in range(1, 21):
        print(f"Processed subject{sub}:")
        os.makedirs(os.path.join(out_imu_dir, f'subject{sub}'), exist_ok=True)

        imu_left_wrist_path = os.path.join(args.data_dir, f'aligned_data/imu/subject{sub}/IMU2.csv')
        out_imu_left_wrist_path = os.path.join(out_imu_dir, f'subject{sub}/LeftWrist.csv')
        shutil.copy(imu_left_wrist_path, out_imu_left_wrist_path)

        imu_right_wrist_path = os.path.join(args.data_dir, f'aligned_data/imu/subject{sub}/IMU1.csv')
        out_imu_right_wrist_path = os.path.join(out_imu_dir, f'subject{sub}/RightWrist.csv')
        shutil.copy(imu_right_wrist_path, out_imu_right_wrist_path)

        imu_right_leg_path = os.path.join(args.data_dir, f'aligned_data/imu/subject{sub}/IMU4.csv')
        out_imu_right_leg_path = os.path.join(out_imu_dir, f'subject{sub}/RightKnee.csv')
        shutil.copy(imu_right_leg_path, out_imu_right_leg_path)

        imu_left_leg_path = os.path.join(args.data_dir, f'aligned_data/imu/subject{sub}/IMU5.csv')
        out_imu_left_leg_path = os.path.join(out_imu_dir, f'subject{sub}/LeftKnee.csv')
        shutil.copy(imu_left_leg_path, out_imu_left_leg_path)

        label_file = os.path.join(args.data_dir, "aligned_data/pose_labels", f"subject{sub}_all_labels.cpl") 
        anno = pickle.load(open(label_file, 'rb'))
        imu_avail_frames = anno['imu_avail_frames']
        frame_start = imu_avail_frames[0]
        frame_end = imu_avail_frames[1]

        mmpose_path = os.path.join(mmpose_dir, f'subject{sub}_results_det.json')
        with open(mmpose_path, 'r') as f:
            mmpose_data = json.load(f)
        mmpose_data = mmpose_data['instance_info']
        mmpose_2d_pose = []
        for frame_det in mmpose_data:
            mmpose_instance = np.array(frame_det['instances'][0]['keypoints'])
            mmpose_2d_pose.append(mmpose_instance.T)
        mmpose_2d_pose = np.array(mmpose_2d_pose)[frame_start:frame_end+2]
        np.save(os.path.join(out_pose_mmpose_dir, f'subject{sub}.npy'), mmpose_2d_pose)


if __name__ == "__main__":
    main()