import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(config, model):
    assert config.optimizer.opt in ['adamw', 'adam']
    if config.optimizer.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )
    elif config.optimizer.opt == 'adam':
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.optimizer.lr,
                weight_decay=config.optimizer.weight_decay
            )
    return optimizer