import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from omegaconf import OmegaConf
import logging
from pprint import pformat
from argparse import ArgumentParser
import wandb

from utils.train import random_seed, train_one_epoch, evaluate
from utils.early_stopping import EarlyStopping
from builder import build_dataloader, build_model, build_optimizer
from utils.logger import setup_logging


def main():
    parser = ArgumentParser(description='Imu Skeletal Contrastive Learning')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    random_seed(seed=config.experiment.seed)

    save_dir = config.experiment.output_dir
    dataset_name = config.data.dataset_name
    current_time = datetime.now()
    experiment_date = current_time.strftime("%m-%d-%Y:%H:%M:%S")
    save_dir = os.path.join(save_dir, f"{dataset_name}/{experiment_date}")
    log_path = os.path.join(save_dir, "training.log")
    log_level = logging.DEBUG if args.debug else logging.INFO

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'checkpoints'))

    setup_logging(log_path, log_level)
    best_ckpt_path = os.path.join(save_dir, f"checkpoints/best.pt")
    last_ckpt_path = os.path.join(save_dir, f"checkpoints/last.pt")

    # setup wandb
    config.wandb.name = f"{config.experiment.name}_{dataset_name}_{experiment_date}"
    wandb_config_dict = OmegaConf.to_container(config.wandb.config, resolve=True)
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=wandb_config_dict,
    )

    config.experiment.save_dir = save_dir
    config.experiment.best_ckpt_path = best_ckpt_path
    config.experiment.last_ckpt_path = last_ckpt_path

    train_loader, val_loader, test_loader = build_dataloader(config)
    model = build_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    gpus_list = config.distributed.gpus
    if len(gpus_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus_list)
        logging.info(f"Using multiple gpus: {gpus_list}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.experiment.device = device
        device = torch.device(device)
        model = model.to(device)
        logging.info(f"Using device: {device}")
    
    optimizer = build_optimizer(config, model)
    early_stopper = EarlyStopping(
        patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta,
        mode=config.training.early_stopping.mode,
        verbose=True,
        save_path=best_ckpt_path
    )

    OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))
    logging.info(pformat(config))
    drop_sensors = config.data.get('drop_sensors', False)
    logging.info("#"*50)
    if drop_sensors:
        logging.info("IMU Sensor Drop Enabled during training")
    else:
        logging.info("IMU Sensor Drop Disabled during training")
    logging.info("#"*50)

    best_val_acc = -np.inf
    start_time = time.time()

    for epoch in range(int(config.training.epochs)):
        train_one_epoch(model, train_loader, epoch, optimizer, config)
        val_metrics = evaluate(model, val_loader, epoch, 'val', config)
        val_epoch_acc = val_metrics['imu_to_motion_R@1']

        if len(gpus_list) > 1:
            torch.save(model.module.state_dict(), last_ckpt_path)
        else:
            torch.save(model.state_dict(), last_ckpt_path)
        early_stopper.step(val_epoch_acc, model)

        if val_epoch_acc > best_val_acc:
            logging.info(f"Validation accuracy improved from {best_val_acc:.6f} to {val_epoch_acc:.6f}. Saving best model.")
            evaluate(model, test_loader, epoch, 'test', config)
            best_val_acc = val_epoch_acc

        if early_stopper.should_stop:
            logging.info("Early stopping triggered.")
            break
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_hours = elapsed_seconds / 3600
    logging.info(f"Training done after {elapsed_hours:.4f} hours\n")


if __name__=="__main__":
    main()