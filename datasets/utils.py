import numpy as np
import os
import json
import pandas as pd
from scipy.signal import resample
import torch
from configs.config import DATA_ROOT, DATASET_CONFIG, SKELETON_CONFIG



def get_windows_in_clip(s_time: float, e_time: float, window_sec: float, stride: float):
    windows = []
    for window_start, window_end in zip(
        np.arange(s_time, e_time, stride),
        np.arange(
            s_time + window_sec,
            e_time,
            stride,
        ),
    ):
        windows.append([window_start, window_end])
    return windows


def normalize_pose(pose2d_signal, dataset_name):
    dataset_config = DATASET_CONFIG[dataset_name]
    img_size = dataset_config['image_size']

    # Normalize the pose with respect to the image size
    pose2d = pose2d_signal.copy()
    if pose2d.ndim == 3:
        pose2d[:, 0, :] /= img_size[1]  # Normalize X
        pose2d[:, 1, :] /= img_size[0]  # Normalize Y
    elif pose2d.ndim == 4:
        pose2d[:, :, 0, :] /= img_size[1]
        pose2d[:, :, 1, :] /= img_size[0]

    return pose2d


def pad_signal(signal, signal_length):
    n = len(signal)
    if n == signal_length:
        return signal
    
    if n < signal_length:
        padding = signal_length - signal.shape[0]
        padded_zeros = np.zeros((padding, 3))
        signal = np.concatenate([signal, padded_zeros], 0)
        return signal

    num_redundant = n - signal_length
    left_trim = num_redundant // 2
    right_trim = num_redundant - left_trim
    left_trim, right_trim = int(left_trim), int(right_trim)
    return signal[left_trim: (n - right_trim)]


def preprocess_imu_window(imu_signal, window_sec, target_srate):
    new_length = int(window_sec * target_srate)
    resampled_signal = resample(imu_signal, new_length)
    resampled_signal = pad_signal(resampled_signal, target_srate*window_sec)
    return resampled_signal


def get_cache_annotations(meta_path, phase):
    with open(meta_path, "r") as json_file:
        meta_data = json.load(json_file)
    idx_list = []
    for idx in meta_data['meta'].keys():
        if meta_data['meta'][idx]['phase']==phase:
            idx_list.append(idx)
    return meta_data, idx_list


def get_har_annotations(meta_path, phase):
    with open(meta_path, "r") as json_file:
        meta_data = json.load(json_file)
    idx_list = []
    for idx in meta_data['meta'].keys():
        if meta_data['meta'][idx]['phase']==phase:
            if meta_data['meta'][idx]['action'] != -1:
                idx_list.append(idx)
    return meta_data, idx_list


def extract_limb_motion(pose2d_centered, dataset_name, limbs):
    if limbs == 'RightWrist':
        joint_idx = [6, 8, 10]
    elif limbs == 'LeftWrist':
        joint_idx = [5, 7, 9]
    elif limbs == 'RightKnee':
        joint_idx = [12, 14, 16]
    elif limbs == 'LeftKnee':
        joint_idx = [11, 13, 15]
    elif limbs == 'Head':
        joint_idx = [0, 1, 2]

    motion_signal = pose2d_centered[..., joint_idx]
    return motion_signal


def get_sync_annotations(dataset_name, sync_dir, phase):
    dataset_config = DATASET_CONFIG[dataset_name]
    split = 'subject' if dataset_name != 'EgoHumans' else 'action'
    subject_list = dataset_config['split'][split][phase]

    anno_path = os.path.join(sync_dir, 'annotations.txt')
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Sync cache not found at '{sync_dir}'. "
            f"Run preprocess/{dataset_name}/cache_sync.py to build it first."
        )

    data_list = []
    with open(anno_path, 'r') as f:
        for line in f:
            line_data = line.strip().split()
            file_idx, w_s, w_e, subject, action, offset = line_data
            subject_idx = str(subject) if dataset_name != 'EgoHumans' else f"{action}_{subject}"
            if subject_idx in subject_list:
                data_list.append((int(file_idx), int(w_s), int(w_e), subject, action, int(offset)))
    return data_list


def parse_TC_sensor_signal(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()

    head_imu = []
    right_hand_imu = []
    left_hand_imu = []
    pelvis_imu = []
    right_leg_imu = []
    left_leg_imu = []

    current_frame = -1
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Check if line indicates a new frame index
        if line.isdigit():
            current_frame += 1
            continue

        parts = line.split()
        sensor_pos = parts[0]
        ori_quat = list(map(float, parts[1:5]))  # Quaternion (w, x, y, z)
        accel = list(map(float, parts[5:8]))     # Acceleration (x, y, z)
        gyro = list(map(float, parts[8:11]))      # Gyroscope (x, y, z)
        mag = list(map(float, parts[11:14]))      # Magnetometer (x, y, z)
        imu_data = accel + ori_quat + gyro + mag

        if sensor_pos == 'Head':
            head_imu.append(imu_data)
        elif sensor_pos == 'L_LowArm':
            left_hand_imu.append(imu_data)
        elif sensor_pos == 'R_LowArm':
            right_hand_imu.append(imu_data)
        elif sensor_pos == 'Pelvis':
            pelvis_imu.append(imu_data)
        elif sensor_pos == 'L_LowLeg':
            left_leg_imu.append(imu_data)
        elif sensor_pos == 'R_LowLeg':
            right_leg_imu.append(imu_data)

    return np.array(head_imu), np.array(right_hand_imu), np.array(left_hand_imu), \
           np.array(pelvis_imu), np.array(right_leg_imu), np.array(left_leg_imu)


def get_sync_imu_signal(dataset_name, subject, action, limbs_list, w_s, w_e, window_sec, target_srate):
    dataset_config = DATASET_CONFIG[dataset_name]
    imu_sample_rate = dataset_config['imu_sample_rate'] if dataset_name != 'mRi' else 10

    start_frame = int(w_s*imu_sample_rate)
    end_frame = int(w_e*imu_sample_rate)

    if dataset_name == 'mRi':
        imu_dir = os.path.join(DATA_ROOT, dataset_name, 'processed_data/imu')

        imu_signals = []
        for limb in limbs_list:
            limb_imu_path = os.path.join(imu_dir, f'subject{subject}', f"{limb}.csv")
            if not os.path.exists(limb_imu_path):
                assert False, f"IMU data for {limb} not found at {limb_imu_path}"
            imu_signal = pd.read_csv(limb_imu_path)[["axg", "ayg", "azg", 'q0', 'q1', 'q2', 'q3']].values
            imu_signal = imu_signal[start_frame:end_frame, :]
            new_length = int(window_sec * target_srate)
            resampled_signal = resample(imu_signal, new_length) 
            imu_signals.append(resampled_signal)

    elif dataset_name == 'TotalCapture':
        imu_path = os.path.join(DATA_ROOT, dataset_name, f's{subject}', f'{action}', f's{subject}_{action}_Xsens.sensors')
        imu_head, imu_right_wrist, imu_left_wrist, \
            imu_body, imu_right_leg, imu_left_leg = parse_TC_sensor_signal(imu_path)

        imu_signals = []
        for limb in limbs_list:
            if limb == 'RightWrist':
                imu_signal = imu_right_wrist
            elif limb == 'LeftWrist':
                imu_signal = imu_left_wrist
            elif limb == 'RightKnee':
                imu_signal = imu_right_leg
            elif limb == 'LeftKnee':
                imu_signal = imu_left_leg
            elif limb == 'Head':
                imu_signal = imu_head

            imu_signal = imu_signal[start_frame:end_frame, :]
            imu_signals.append(imu_signal)

    imu_signals = np.stack(imu_signals, axis=0)
    return imu_signals


def get_sync_motion_signal(dataset_name, subject, action, limb_list, w_s, w_e, motion_type, window_sec, target_srate, offset):   
    dataset_config = DATASET_CONFIG[dataset_name]
    motion_sample_rate = dataset_config['motion_sample_rate']

    start_frame = int(w_s * motion_sample_rate)
    end_frame = int(w_e * motion_sample_rate)

    shift_start_frame = start_frame + int(offset * motion_sample_rate / 1000)
    shift_end_frame = end_frame + int(offset * motion_sample_rate / 1000)

    if dataset_name == 'mRi':
        if motion_type == 'wjoint':
            pose2d_dir = os.path.join(DATA_ROOT, dataset_name, 'processed_data/pose2d_mmpose')
            pose2d = np.load(os.path.join(pose2d_dir, f'subject{subject}.npy'))
            original_pose2d = pose2d[start_frame:end_frame, :, :]
            shifted_pose2d = pose2d[shift_start_frame:shift_end_frame, :, :]
            original_pose2d_norm = normalize_pose(original_pose2d, dataset_name)
            shifted_pose2d_norm = normalize_pose(shifted_pose2d, dataset_name)

            motion_signal = []
            shifted_motion_signal = []
            for limb in limb_list:
                motion_signal.append(extract_limb_motion(original_pose2d_norm, dataset_name, limb))
                shifted_motion_signal.append(extract_limb_motion(shifted_pose2d_norm, dataset_name, limb))
            motion_signal = np.stack(motion_signal, axis=0)
            motion_signal = torch.tensor(motion_signal).float()
            S, T, C, J = motion_signal.shape
            motion_signal = motion_signal.reshape(S, T, C * J)

            shifted_motion_signal = np.stack(shifted_motion_signal, axis=0)
            shifted_motion_signal = torch.tensor(shifted_motion_signal).float()
            S, T, C, J = shifted_motion_signal.shape
            shifted_motion_signal = shifted_motion_signal.reshape(S, T, C * J)

            return motion_signal, shifted_motion_signal, original_pose2d, shifted_pose2d
        
        elif motion_type == 'pose2d':
            pose2d_dir = os.path.join(DATA_ROOT, dataset_name, 'processed_data/pose2d')
            pose2d = np.load(os.path.join(pose2d_dir, f'subject{subject}.npy'))
            original_pose2d = pose2d[start_frame:end_frame, :, :]
            shifted_pose2d = pose2d[shift_start_frame:shift_end_frame, :, :]

            original_pose2d = normalize_pose(original_pose2d, dataset_name)
            shifted_pose2d = normalize_pose(shifted_pose2d, dataset_name)

            new_length = int(window_sec * target_srate)
            original_pose2d = resample(original_pose2d, new_length)
            shifted_pose2d = resample(shifted_pose2d, new_length)

            T, C, J = original_pose2d.shape
            original_pose2d = original_pose2d.reshape(T, C * J)
            shifted_pose2d = shifted_pose2d.reshape(T, C * J)
            return torch.tensor(original_pose2d[np.newaxis, :]).float(), torch.tensor(shifted_pose2d[np.newaxis, :]).float(), original_pose2d, shifted_pose2d