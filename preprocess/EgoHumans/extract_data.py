import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from configs.config import DATA_ROOT, BODY_MODEL_PATH
import argparse
from pathlib import Path
import torch
import numpy as np
import subprocess
from collections import defaultdict
from pathlib import Path
from deps.articulate import math
from deps.articulate.model import ParametricModel



def unzip_data(data_dir):
    raw_data_dir = os.path.join(data_dir, 'data')
    seq_list = sorted(os.listdir(raw_data_dir))

    for seq in seq_list:
        seq_path = os.path.join(raw_data_dir, seq)
        sub_seq_list = sorted(os.listdir(seq_path))
        
        for sub_seq in sub_seq_list:
            if sub_seq.endswith(".tar.gz"):
                tar_path = os.path.join(seq_path, sub_seq)

                extracted_folder = os.path.join(seq_path, sub_seq.replace('.tar.gz', ''))
                if os.path.exists(extracted_folder):
                    print(f"✅ Skipping {tar_path}, folder already exists.")
                    continue
                
                # Run the tar extraction command
                print(f"Extracting {tar_path}...")
                subprocess.run(
                    ["tar", "-xzvf", tar_path, "--strip-components=8"],
                    cwd=seq_path,
                    check=True
                )


def extract_imu(data_dir, target_fps=20):
    raw_data_dir = os.path.join(data_dir, 'data')
    extracted_data_dir = os.path.join(data_dir, 'extracted_data')
    if not os.path.exists(extracted_data_dir):
        os.makedirs(extracted_data_dir)

    def _syn_acc(v, smooth_n=4):
        """Synthesize accelerations from vertex positions."""
        mid = smooth_n // 2
        scale_factor = target_fps ** 2

        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc
    
    def _new_pid():
        return {
            'pose2d': [],          # list of (17,2)
            'pose2d_mask': [],     # list of () -> 1 if present, 0 if missing
            'global_orient': [],
            'body_pose': [],
            'trans': [],
            'betas': None,         # set once
        }
    

    num_seq = 0
    skip_seq = 0

    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411])
    ji_mask = torch.tensor([18, 19, 4, 5, 15])
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])

    body_model = ParametricModel(Path(BODY_MODEL_PATH))

    action_list = sorted(os.listdir(raw_data_dir))
    for act in action_list:
        act_path = os.path.join(raw_data_dir, act)
        seq_list = sorted([d for d in os.listdir(act_path) if os.path.isdir(os.path.join(act_path, d))])

        for seq in seq_list:
            num_seq += 1
            print(f"Processing {num_seq}-th sequence: {seq} in action {act}.")
            pose2d_path = os.path.join(act_path, seq, 'processed_data', 'poses2d/cam03/rgb')
            smpl_path = os.path.join(act_path, seq, 'processed_data', 'smpl')

            if not os.path.exists(pose2d_path) or not os.path.exists(smpl_path):
                print(f"Skipping {seq} in action {act} because {pose2d_path} or {smpl_path} does not exist.")
                skip_seq += 1
                continue

            num_frame_pose2d = len(os.listdir(pose2d_path))
            num_frame_smpl = len(os.listdir(smpl_path))
            num_frames = min(num_frame_pose2d, num_frame_smpl)

            data_dict = defaultdict(_new_pid)
            zero_kp = np.zeros((17, 2), dtype=np.float32)
            for i in range(1, num_frames + 1):
                pose2d_file = os.path.join(pose2d_path, f"{i:05d}.npy")
                smpl_file = os.path.join(smpl_path, f"{i:05d}.npy")

                pose2d_frame = np.load(pose2d_file, allow_pickle=True)
                pose_map = {e['human_name']: e['keypoints'] for e in pose2d_frame}
                smpl_frame = np.load(smpl_file, allow_pickle=True).item()

                for pid, smpl_data in smpl_frame.items():
                    rec = data_dict[pid]
                    rec['global_orient'].append(smpl_data['global_orient'])
                    rec['body_pose'].append(smpl_data['body_pose'])
                    rec['trans'].append(smpl_data['transl'])
                    if rec['betas'] is None:
                        rec['betas'] = smpl_data['betas'][:10]

                    kp = pose_map.get(pid, None)
                    if kp is None:
                        rec['pose2d'].append(zero_kp)
                        rec['pose2d_mask'].append(0)
                    else:
                        # print(kp.shape)
                        rec['pose2d'].append(kp[:17, :2])
                        rec['pose2d_mask'].append(1)

            for pid in data_dict.keys():
                kpnts_2d = np.array(data_dict[pid]['pose2d'])                                   # (T, 21, 2)
                body_poses = torch.tensor(np.array(data_dict[pid]['body_pose'])).float()
                body_poses = body_poses.reshape(body_poses.shape[0], -1, 3)                     # (T, 23, 3)
                global_oris = torch.tensor(np.array(data_dict[pid]['global_orient'])).float()   # (T, 3)
                trans = torch.tensor(np.array(data_dict[pid]['trans'])).float()                 # (T, 3)
                poses = torch.cat((global_oris.unsqueeze(1), body_poses), dim=1)                # (T, 24, 3)
                shapes = torch.tensor(np.array(data_dict[pid]['betas'])).float().unsqueeze(0)   # (1, 10)

                trans = amass_rot.matmul(trans.unsqueeze(-1)).view_as(trans)
                poses[:, 0] = math.rotation_matrix_to_axis_angle(
                    amass_rot.matmul(math.axis_angle_to_rotation_matrix(poses[:, 0])))
                p = math.axis_angle_to_rotation_matrix(poses).view(-1, 24, 3, 3)
                grot, joint, vert = body_model.forward_kinematics(p, shapes, trans, calc_mesh=True)

                sacc = _syn_acc(vert[:, vi_mask])
                srot = grot[:, ji_mask]
                srot_quat = math.axis_angle_to_quaternion(math.rotation_matrix_to_axis_angle(srot))
                srot_quat = srot_quat.view(-1, 5, 4)
                imu = torch.cat((sacc, srot_quat), dim=-1).numpy()

                extracted_file = act.split('_')[0] + '_' + seq.split('_')[0] + '_' + pid + '.npy'
                extracted_file = os.path.join(extracted_data_dir, extracted_file)

                # print(imu.shape, kpnts_2d.shape)
                extracted_data = {
                    'imu': imu,
                    'pose2d': kpnts_2d,
                    'pose3d': joint.numpy()
                }
                np.save(extracted_file, extracted_data)

    print('Total sequences processed:', num_seq)
    print(f"Skipped {skip_seq} sequences due to missing data.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, 'EgoHumans'))
    ap.add_argument("--mode", required=True, help="unzip, extract_imu")
    args = ap.parse_args()

    assert args.mode in ["unzip", "extract_imu"], "Invalid mode"

    if args.mode == "unzip":
        unzip_data(args.data_dir)
    elif args.mode == "extract_imu":
        extract_imu(args.data_dir, target_fps=20)