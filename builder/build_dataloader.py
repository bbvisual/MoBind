import os
import logging
from torch.utils.data import DataLoader
from datasets.dataset import ContrastiveDataset
from configs.config import DATASET_CONFIG


def build_dataset(config):
    root_dir = config.data.root_dir
    motion_type = config.data.motion_type
    window_sec = config.data.window_sec
    stride_sec = config.data.stride_sec
    experiment_name = config.experiment.name
    dataset_name = config.data.dataset_name
    limb_list = DATASET_CONFIG[dataset_name]['limb_list']
    split = config.data.split if 'split' in config.data else 'subject'
    cache_name = f"cache_{split}_{window_sec}_{stride_sec}"
    cache_dir = os.path.join(root_dir, cache_name)
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist.")

    multi_sensor = experiment_name != 'stage1'
    train_dataset = ContrastiveDataset(dataset_name, cache_dir, 'train', split, motion_type, limb_list, multi_sensor)
    val_dataset   = ContrastiveDataset(dataset_name, cache_dir, 'val',   split, motion_type, limb_list, multi_sensor)
    test_dataset  = ContrastiveDataset(dataset_name, cache_dir, 'test',  split, motion_type, limb_list, multi_sensor)

    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def build_dataloader(config):
    train_dataset, val_dataset, test_dataset = build_dataset(config)

    batch_size  = config.data.loader.batch_size
    num_workers = config.data.loader.num_workers
    pin_memory  = config.data.loader.pin_memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
