# Update these paths to match your local setup before running any scripts.
DATA_ROOT = "/path/to/datasets"
BODY_MODEL_PATH = "/path/to/smpl/basicmodel_m.pkl"


DATASET_CONFIG = {
    "mRi": {
        "imu_sample_rate": 50,
        "motion_sample_rate": 10,
        "class_names": [
            "T pose", "pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6", 
            "pose_7", "pose_8", "pose_9", "pose_10", "free_form", "walk", 'T pose 2', 'null'
        ],
        "idx2action": {
            0: 'T pose', 1: 'pose_1', 2: 'pose_2', 3: 'pose_3', 4: 'pose_4', 5: 'pose_5', 6: 'pose_6',
            7: 'pose_7', 8: 'pose_8', 9: 'pose_9', 10: 'pose_10', 11: 'free_form', 12: 'walk', -1: 'null'
        },
        "action2idx": {
            'T pose': 0, 'pose_1': 1, 'pose_2': 2, 'pose_3': 3, 'pose_4': 4, 'pose_5': 5, 'pose_6': 6,
            'pose_7': 7, 'pose_8': 8, 'pose_9': 9, 'pose_10': 10, 'free_form': 11, 'walk': 12, 'T pose 2': 0, 'null': -1
        },
        "image_size": (1080, 1920),
        "split": {
            "subject": {
                "train": [str(i) for i in range(1, 15)],
                "val": ['16', '18'],
                "test": ['15', '17', '19', '20']
            },
            "action": {
                "train": ['T pose', 'null', 'pose_1', 'pose_2', 'pose_4', 'pose_6', 'pose_7', 'pose_9', 'pose_10', 'walk'],
                "val": ['pose_2', 'pose_5'],
                "test": ['pose_3', 'pose_8', "free_form"]
            }
        },
        "limb_list": ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee"],
    },
    
    "EgoHumans": {
        "imu_sample_rate": 20,
        "motion_sample_rate": 20,
        "image_size": (2160, 3840),
        'action2idx': {'tagging': 0, 'lego': 1, 'fencing': 2, 'basketball': 3, 'volleyball': 4, 'badminton': 5, 'tennis': 6},
        'idx2action': {0: 'tagging', 1: 'lego', 2: 'fencing', 3: 'basketball', 4: 'volleyball', 5: 'badminton', 6: 'tennis'},
        "split": {
            "action": {
                "train": ['01_006', '01_007', '01_008', '01_009', '01_010', '01_011', '01_012', '01_013', '01_014', '02_006', '03_006', '03_007', '03_008', '03_009', '03_010', '03_011', '03_012', '03_013', '03_014', '04_006', '04_007', '04_008', '04_009', '04_010', '04_011', '04_012', '04_013', '04_014', '05_006', '05_007', '05_008', '05_009', '05_010', '05_011', '06_006', '06_007', '06_008', '06_009', '06_010', '06_011', '06_012', '06_013', '06_014', '06_015', '06_016', '06_017', '06_018', '06_019', '06_020', '06_021', '06_022', '06_023', '06_024', '06_025', '06_026', '06_027', '06_028', '06_029', '06_030', '06_031', '06_032', '06_033', '06_034', '06_035', '06_036', '06_037', '06_038', '06_039', '06_040', '06_041', '06_042', '06_043', '06_044', '06_045', '06_046', '06_047', '06_048', '06_049', '06_050', '06_051', '06_052', '06_053', '06_054', '06_055', '06_056', '06_057', '06_058', '06_059', '06_060', '06_061', '07_006', '07_007', '07_008', '07_009', '07_010', '07_011', '07_012', '07_013'],
                "val": ['01_005', '03_005', '04_005', '05_005', '06_005', '07_005'],
                "test": ['01_001', '01_002', '01_003', '01_004', '03_001', '03_002', '03_003', '03_004', '04_001', '04_002', '04_003', '04_004', '05_001', '05_002', '05_003', '05_004', '06_001', '06_002', '06_003', '06_004', '07_001', '07_002', '07_003', '07_004'],
            },
        },
        "limb_list": ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]
    },

    "TotalCapture": {
        "imu_sample_rate": 60,
        "motion_sample_rate": 60,
        "image_size": (1080, 1920),
        "action2idx": {'acting': 0, 'freestyle': 1, 'rom': 2, 'walking': 3},
        'idx2action': {0: 'acting', 1: 'freestyle', 2: 'rom', 3: 'walking'},
        "split": {
            "subject": {
                "train": ['1', '2', '3'],
                "val": ['4'],
                "test": ['5']
            },
        },
        "limb_list": ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]
    },
}

SKELETON_CONFIG = {
    "keypoint_names": [
        "Nose", "LeftEye", "RightEye", "LeftEar", "RightEar",
        "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
        "LeftWrist", "RightWrist", "LeftHip", "RightHip",
        "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle"
    ],
    "skeleton": [
        (0, 1), (0, 2), (2, 4), (1, 3), (0, 5), (0, 6),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (11, 13), (13, 15),
        (6, 8), (8, 10), (12, 14), (14, 16),
    ],
    "bone_lr": [
        1, 1, 1, 1, 1, 1,   # Head
        1, 1, 1, 1,         # Body
        0, 0, 0, 0,         # Left arm and leg
        2, 2, 2, 2,         # Right arm and leg
    ],
}