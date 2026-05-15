import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.utils import get_cache_annotations, normalize_pose, extract_limb_motion,\
                            get_sync_annotations

class ContrastiveDataset(Dataset):
    def __init__(self,
            dataset_name: str,
            cache_dir: str,
            phase: str,
            split: str,
            motion_type: str,
            limbs_list: list,
            multi_sensor: bool = False):

        super(ContrastiveDataset, self).__init__()
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.phase = phase
        self.split = split
        self.motion_type = motion_type
        self.multi_sensor = multi_sensor
        self.limbs_list = limbs_list

        imu_dir = os.path.join(cache_dir, 'data/imu')
        pose2d_dir = os.path.join(cache_dir, 'data/pose2d')
        meta_path = os.path.join(cache_dir, 'meta.json')
        self.meta, self.idx_list = get_cache_annotations(meta_path, phase)

        self.data_list = []
        if multi_sensor:
            for idx in sorted(self.idx_list):
                imu_path = os.path.join(imu_dir, f"{idx}")
                pose2d_path = os.path.join(pose2d_dir, f"{idx}.npy")
                self.data_list.append((idx, None, imu_path, pose2d_path)) # limb is None for multi-sensor
        else:
            for limb in limbs_list:
                for idx in sorted(self.idx_list):
                    imu_path = os.path.join(imu_dir, f"{idx}/{limb}.npy")
                    pose_path = os.path.join(pose2d_dir, f"{idx}.npy")
                    if not os.path.exists(imu_path):
                        continue
                    self.data_list.append((idx, limb, imu_path, pose_path))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        meta_idx, limb, imu_path, pose_path = self.data_list[idx]
        pose_signal = np.load(pose_path)
        pose_vis_signal = pose_signal.copy()
        pose_signal = normalize_pose(pose_signal, self.dataset_name) # T, C, J

        if self.multi_sensor:
            imu_signal = []
            for limb in self.limbs_list:
                limb_imu_path = os.path.join(imu_path, f"{limb}.npy")
                if not os.path.exists(limb_imu_path):
                    assert False, f"IMU data for {limb} not found at {limb_imu_path}"
                imu_signal.append(np.load(limb_imu_path))
            imu_signal = np.stack(imu_signal, axis=0)
            imu_signal = imu_signal.transpose(0, 2, 1)    
            imu_signal = torch.tensor(imu_signal).float()

            if self.motion_type == 'wjoint':
                motion_signal = []
                for limb in self.limbs_list:
                    motion_signal.append(extract_limb_motion(pose_signal, self.dataset_name, limb))
                motion_signal = np.stack(motion_signal, axis=0)
                motion_signal = torch.tensor(motion_signal).float()
                S, T, C, J = motion_signal.shape
                motion_signal = motion_signal.reshape(S, T, C * J).permute(0, 2, 1) # S, C*J, T
            elif self.motion_type == 'pose2d':
                motion_signal = torch.tensor(pose_signal, dtype=torch.float32)
                T, C, J = motion_signal.shape
                motion_signal = motion_signal.reshape(T, C * J).T   # C* J, T

        else:
            imu_signal = np.load(imu_path)
            imu_signal = torch.tensor(imu_signal.T).float()

            motion_signal = extract_limb_motion(pose_signal, self.dataset_name, limb)
            motion_signal = torch.tensor(motion_signal, dtype=torch.float32)
            T, C, J = motion_signal.shape
            motion_signal = motion_signal.reshape(T, C * J).T
        subject = self.meta['meta'][meta_idx]['subject']
        action = int(self.meta['meta'][meta_idx]['action']) - 1 if self.dataset_name == 'EgoHumans' else self.meta['meta'][meta_idx]['action']

        dict_out = {}
        dict_out['uid'] = meta_idx
        dict_out['limb'] = limb
        dict_out['imu'] = imu_signal
        dict_out['motion'] = motion_signal
        dict_out['pose2d'] = torch.tensor(pose_vis_signal).float()
        dict_out['action'] = action
        dict_out['subject'] = subject
        return dict_out
    

class SyncDataset(Dataset):
    def __init__(self,
            dataset_name: str,
            sync_dir: str,
            phase: str,
            motion_type: str,
            limb_list: list = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.phase = phase
        self.motion_type = motion_type
        self.limb_list = limb_list
        self.sync_dir = sync_dir

        self.data_list = get_sync_annotations(dataset_name, sync_dir, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        file_idx, w_s, w_e, subject, action, offset = data

        imu_path = os.path.join(self.sync_dir, 'imu', f"{file_idx:05d}.npy")
        pose_path = os.path.join(self.sync_dir, 'pose2d', f"{file_idx:05d}.npy")
        pose_shifted_path = os.path.join(self.sync_dir, 'pose2d_shifted', f"{file_idx:05d}.npy")

        imu_signal = np.load(imu_path)
        pose2d = np.load(pose_path)
        shifted_pose2d = np.load(pose_shifted_path)

        pose2d_norm = normalize_pose(pose2d, self.dataset_name)
        shifted_pose2d_norm = normalize_pose(shifted_pose2d, self.dataset_name)

        if self.motion_type == 'wjoint':
            motion_signal = []
            shifted_motion_signal = []
            for limb in self.limb_list:
                motion_signal.append(extract_limb_motion(pose2d_norm, self.dataset_name, limb))
                shifted_motion_signal.append(extract_limb_motion(shifted_pose2d_norm, self.dataset_name, limb))
            motion_signal = np.stack(motion_signal, axis=0)
            motion_signal = torch.tensor(motion_signal).float()

            shifted_motion_signal = np.stack(shifted_motion_signal, axis=0)
            shifted_motion_signal = torch.tensor(shifted_motion_signal).float()

            if len(motion_signal.shape) == 4:
                S, T, C, J = motion_signal.shape
                motion_signal = motion_signal.reshape(S, T, C * J)
                shifted_motion_signal = shifted_motion_signal.reshape(S, T, C * J)

            elif len(motion_signal.shape) == 5:
                S, T, P, C, J = motion_signal.shape
                motion_signal = motion_signal.reshape(S, T, P, C * J)

        elif self.motion_type == 'pose2d':
            motion_signal = torch.tensor(pose2d_norm, dtype=torch.float32)
            T, C, J = motion_signal.shape
            motion_signal = motion_signal.reshape(T, C * J).T

            shifted_motion_signal = torch.tensor(shifted_pose2d_norm, dtype=torch.float32)
            T, C, J = shifted_motion_signal.shape
            shifted_motion_signal = shifted_motion_signal.reshape(T, C * J).T

        dict_out = {}
        dict_out['imu'] = torch.tensor(imu_signal).permute(1, 0, 2).float()
        dict_out['motion'] = motion_signal
        dict_out['shifted_motion'] = shifted_motion_signal
        dict_out['pose2d'] = pose2d
        dict_out['shifted_pose2d'] = shifted_pose2d
        dict_out['offset'] = offset
        dict_out['subject'] = subject
        dict_out['action'] = action
        dict_out['start_sec'] = w_s
        dict_out['end_sec'] = w_e
        return dict_out