import torch
import os
from omegaconf import OmegaConf
import torch.nn as nn

from models.contrastive_module import ContrastiveStage1Module, ContrastiveStage2Module, \
                                    ContrastiveBaselineModule, ContrastiveMAE, ContrastiveConcatBaselineModule
from models.imu_encoder.conv_former import ConvFormer
from models.imu_encoder.baseline import LSTMEncoder, MW2StackRNNPooling
from models.imu_encoder.conv_lstm import Temporal1DEncoder



def build_model(config):
    imu_srate = config.data.imu_srate
    motion_srate = config.data.motion_srate

    embedding_size = config.model.embedding_size

    exp_name = config.experiment.name
    imu_encoder_name = config.model.imu_encoder
    motion_encoder_name = config.model.motion_encoder
    imu_channels = config.model.imu_channels
    motion_channels = config.model.motion_channels
    
    if exp_name == "stage1":
        if imu_encoder_name == "conv_former":
            imu_encoder = ConvFormer(
                sample_rate=imu_srate, input_channels=imu_channels, embedding_size=embedding_size,
                num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
        elif imu_encoder_name == "temporal_1d":
            imu_encoder = Temporal1DEncoder(
                input_channels=imu_channels, embedding_size=embedding_size, sample_rate=imu_srate)

        if motion_encoder_name == "conv_former":
            motion_encoder = ConvFormer(
                sample_rate=motion_srate, input_channels=motion_channels, embedding_size=embedding_size,
                num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
        elif motion_encoder_name == "temporal_1d":
            motion_encoder = Temporal1DEncoder(
                input_channels=motion_channels, embedding_size=embedding_size, sample_rate=motion_srate)
        return ContrastiveStage1Module(motion_encoder=motion_encoder, imu_encoder=imu_encoder)

    elif exp_name == "stage2":
        global_embedding_size = config.model.global_embedding_size if 'global_embedding_size' in config.model else embedding_size
        agg = config.model.agg if 'agg' in config.model else 'transformer'
        if config.model.stage1_exp is not None:
            print(f"Loading stage 1 model from {config.model.stage1_exp}")
            stage1_config_path = os.path.join(config.model.stage1_exp, 'config.yaml')
            ckpt_path = os.path.join(config.model.stage1_exp, 'checkpoints/best.pt')
            stage1_config = OmegaConf.load(stage1_config_path)

            model = build_model(stage1_config)
            model.load_state_dict(torch.load(f'{ckpt_path}')) 
            imu_encoder = model.imu_encoder
            motion_encoder = model.motion_encoder
        else:
            print("Training Global REP from scratch")
            if imu_encoder_name == "conv_former":
                imu_encoder = ConvFormer(
                        sample_rate=imu_srate, input_channels=imu_channels, embedding_size=embedding_size,
                        num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
            if motion_encoder_name == "conv_former":
                motion_encoder = ConvFormer(
                        sample_rate=motion_srate, input_channels=motion_channels, embedding_size=embedding_size,
                        num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
        return ContrastiveStage2Module(motion_encoder=motion_encoder, imu_encoder=imu_encoder, 
                                       num_limbs=config.data.num_limbs, global_emb_dim=global_embedding_size, agg=agg)
    elif exp_name == "mae":
        global_embedding_size = config.model.global_embedding_size if 'global_embedding_size' in config.model else embedding_size
        lambda_mae = config.model.lambda_mae if 'lambda_mae' in config.model else 0.3
        mask_ratio = config.model.mask_ratio if 'mask_ratio' in config.model else 0.75
        
        if config.model.stage1_exp is not None:
            print(f"Loading stage 1 model from {config.model.stage1_exp}")
            stage1_config_path = os.path.join(config.model.stage1_exp, 'config.yaml')
            ckpt_path = os.path.join(config.model.stage1_exp, 'checkpoints/best.pt')
            stage1_config = OmegaConf.load(stage1_config_path)

            model = build_model(stage1_config)
            model.load_state_dict(torch.load(f'{ckpt_path}')) 
            imu_encoder = model.imu_encoder
            motion_encoder = model.motion_encoder
        else:
            print("Training Global REP from scratch")
            if imu_encoder_name == "conv_former":
                imu_encoder = ConvFormer(
                        sample_rate=imu_srate, input_channels=imu_channels, embedding_size=embedding_size,
                        num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
            if motion_encoder_name == "conv_former":
                motion_encoder = ConvFormer(
                        sample_rate=motion_srate, input_channels=motion_channels, embedding_size=embedding_size,
                        num_heads=config.model.num_heads, num_layers=config.model.num_blocks)
        return ContrastiveMAE(motion_encoder=motion_encoder, imu_encoder=imu_encoder, 
                                       num_limbs=config.data.num_limbs, global_emb_dim=global_embedding_size,
                                       lambda_mae=lambda_mae, mask_ratio=mask_ratio)

    else:
        raise NotImplementedError(f"Experiment {exp_name} is not implemented.")