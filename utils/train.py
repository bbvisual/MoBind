import torch
import os
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import math
import logging
from contextlib import suppress
from functools import partial
import wandb
import json


def random_seed(seed=1903):
    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed (single GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed (multiple GPUs)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility


def evaluate_batch_similarity(source_embeddings, target_embeddings, device):
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy, t_s_accuracy


def evaluate_batch_similarity_multi_pos(pose_embeddings, imu_embeddings, device):
    """
    pose_embeddings : (B, D)
    imu_embeddings  : (B, N, D)
    Return:
        pose_to_imu_acc, imu_to_pose_acc  (Top-1 accuracies)
    """

    B, N, D = imu_embeddings.shape
    # Normalize
    pose_norm = F.normalize(pose_embeddings, dim=1)          # (B, D)
    imu_norm  = F.normalize(imu_embeddings, dim=2)           # (B, N, D)

    # ----- Pose → IMU -----
    # Flatten IMUs: (B*N, D)
    imu_flat = imu_norm.reshape(B * N, D)
    # similarities: (B, B*N)
    sim_p2i = torch.mm(pose_norm, imu_flat.t())

    # Top-1 index per pose
    pred_idx = torch.argmax(sim_p2i, dim=1)                  # (B,)
    # Each pose i has N correct IMU indices: [i*N, i*N+1, ..., i*N+N-1]
    correct_owner = pred_idx // N
    pose2imu_acc = (correct_owner == torch.arange(B, device=device)).float().mean()

    # ----- IMU → Pose -----
    # similarities: (B*N, B)
    sim_i2p = torch.mm(imu_flat, pose_norm.t())
    # Top-1 pose index per IMU view
    pred_pose = torch.argmax(sim_i2p, dim=1)                 # (B*N,)
    # Ground truth pose index for each IMU view
    gt_pose = torch.arange(B, device=device).repeat_interleave(N)
    imu2pose_acc = (pred_pose == gt_pose).float().mean()
    return pose2imu_acc, imu2pose_acc


def get_autocast(precision, device_type='cuda'):
    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        return suppress
    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model
    
    
def generate_IMU_mask(batch_size, num_sensors, max_drop, device):
    S = num_sensors
    K = min(max_drop, S)
    if K == 0:
        return torch.ones(batch_size, S, dtype=torch.bool, device=device)
    
    probs = torch.ones(K + 1, device=device) / (K + 1)
    drop_counts = torch.multinomial(probs, num_samples=batch_size, replacement=True)  # (B,)

    # pick K candidate sensors per sample using random scores
    scores = torch.rand(batch_size, S, device=device)
    cand_idx = scores.topk(k=K, dim=1, largest=True).indices  # (B, K)
    sel = (torch.arange(K, device=device)[None, :] < drop_counts[:, None])  # (B, K) bool
    oh = F.one_hot(cand_idx, num_classes=S).bool()            # (B, K, S)
    drop_bool = (oh & sel.unsqueeze(-1)).any(dim=1)           # (B, S) True = drop
    keep_mask = ~drop_bool                                    # (B, S) True = keep
    return keep_mask


def train_one_epoch(model, dataloader, epoch, optimizer, config):
    model.train()
    device = torch.device(config.experiment.device)
    autocast = get_autocast(config.experiment.precision, device_type=device.type)

    train_total_loss, train_g_loss, train_l_loss, train_mse_loss, train_acc = [], [], [], [], []
    global_weight = config.training.global_weight
    local_weight = config.training.local_weight
    multi_weight = config.training.multi_weight if 'multi_weight' in config.training else 0.0

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_imus = batch['imu'].to(device=device)         # B, N, C, T
        input_motions = batch['motion'].to(device=device)   # B, N, C, T
        input_pose2d = batch['pose2d'].to(device=device)

        B, S, *_ = input_imus.shape
        # IMU drop
        drop_sensors = config.data.get('drop_sensors', False)
        if drop_sensors:
            max_drop = 2 if config.data.num_limbs == 4 else 3
            keep_mask = generate_IMU_mask(
                batch_size=B, num_sensors=S, max_drop=max_drop,
                device=input_imus.device
            )  # (B, N) bool
            input_imus = input_imus * keep_mask.unsqueeze(2).unsqueeze(3).to(input_imus.dtype)

        inputs = {'imu': input_imus, 'motion': input_motions, 'pose2d': input_pose2d}

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs, global_weight=global_weight, local_weight=local_weight, 
                            multi_weight=multi_weight, is_train=True)
            total_loss = outputs['total_loss']
            loss_g = outputs.get('loss_g', None)
            loss_l = outputs.get('loss_l', None)
            feats_i = outputs['cls_i']
            feats_m = outputs['cls_m']
            loss_mse = outputs.get('loss_mse', None)

            total_loss = total_loss.mean()
            loss_g = loss_g.mean()
            loss_l = loss_l.mean()
            loss_mse = loss_mse.mean() if loss_mse is not None else None

        train_total_loss.append(total_loss.item())
        train_g_loss.append(loss_g.item() if loss_g is not None else 0.0)
        train_l_loss.append(loss_l.item() if loss_l is not None else 0.0)
        train_mse_loss.append(loss_mse.item() if loss_mse is not None else 0.0)


        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            if hasattr(unwrap_model(model), 'logit_scale_local'):
                unwrap_model(model).logit_scale_local.clamp_(0, math.log(100))

        if config.experiment.name in ['stage1', 'stage2', 'mae'] or "cat" in config.experiment.name:
            i2m_acc, m2i_acc = evaluate_batch_similarity(feats_i.float(), feats_m.float(), device)
        else:
            i2m_acc, m2i_acc = evaluate_batch_similarity_multi_pos(feats_m.float(), feats_i.float(), device)
        train_acc.append((i2m_acc.item() + m2i_acc.item()) / 2)

    train_epoch_loss = np.mean(train_total_loss)
    train_epoch_acc = np.mean(train_acc)
    train_global_loss = np.mean(train_g_loss)
    train_local_loss = np.mean(train_l_loss)
    train_mse_loss = np.mean(train_mse_loss)
    if (epoch + 1) % config.training.log_interval == 0:
        logging.info(
            f"Train Epoch: {epoch+1}/{config.training.epochs} | "
            f"Train Loss: {train_epoch_loss:.6f} | "
            f"Train Global Loss: {train_global_loss:.6f} | "
            f"Train Local Loss: {train_local_loss:.6f} | "
            f"Train MSE Loss: {train_mse_loss:.6f} | "
            f"Train Accuracy: {train_epoch_acc:.4f}"
        )
    metrics = {
        "loss": train_epoch_loss,
        "global_loss": train_global_loss,
        "local_loss": train_local_loss,
        "mse_loss": train_mse_loss,
        "accuracy": train_epoch_acc,
        'logit_scale': unwrap_model(model).logit_scale.item(),
    }
    log_data = {"train/" + name: val for name, val in metrics.items()}
    wandb.log(log_data)

    with open(os.path.join(config.experiment.save_dir, "results.jsonl"), "a+") as f:
        f.write(json.dumps(metrics))
        f.write("\n")


def evaluate(model, dataloader, epoch, phase, config):
    model.eval()
    device = config.experiment.device
    metrics = {}

    num_samples = 0
    cumulative_loss = 0.0
    all_imu_features, all_motion_features = [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_imus = batch['imu'].to(device)
            input_motions = batch['motion'].to(device)
            input_pose2d = batch['pose2d'].to(device)
            inputs = {'imu': input_imus, 'motion': input_motions, 'pose2d': input_pose2d}
            batch_size = input_imus.shape[0]

            outputs = model(inputs, global_weight=1.0, local_weight=0.0, is_train=False)
            total_loss = outputs['total_loss']
            loss_g = outputs.get('loss_g', None)
            loss_l = outputs.get('loss_l', None)
            feats_i = outputs['cls_i']
            feats_m = outputs['cls_m']

            total_loss = total_loss.mean()
            loss_g = loss_g.mean()
            loss_l = loss_l.mean()

            all_imu_features.append(feats_i.cpu())
            all_motion_features.append(feats_m.cpu())
            cumulative_loss += total_loss * batch_size
            num_samples += batch_size

    val_metrics = get_clip_metrics(
        imu_features=torch.cat(all_imu_features),
        motion_features=torch.cat(all_motion_features),
    ) if config.experiment.name in ['stage1', 'stage2', 'mae'] or "cat" in config.experiment.name else get_clip_metrics_multi_pos(
        imu_features=torch.cat(all_imu_features),
        motion_features=torch.cat(all_motion_features),
    )
    loss = cumulative_loss / num_samples
    metrics.update(
        {**val_metrics, f"{phase} loss": loss.item()}
    )

    if (epoch + 1)% config.training.log_interval == 0 or phase == 'test':
        header = f"{phase.capitalize()} Epoch: {epoch}"
        lines = [f"{k:<30}: {v:.4f}" for k, v in metrics.items()]
        log_message = header + "\n" + "\n".join(lines)
        logging.info(log_message)

    log_data = {f"{phase}/" + name: val for name, val in metrics.items()}
    wandb.log(log_data)

    with open(os.path.join(config.experiment.save_dir, "results.jsonl"), "a+") as f:
        f.write(json.dumps(metrics))
        f.write("\n")
    return metrics


def get_clip_metrics(imu_features, motion_features, k_list=[1, 3, 5, 10]):
    metrics = {}
    logits_per_imu = (imu_features @ motion_features.t()).detach().cpu()
    logits_per_motion = logits_per_imu.t().detach().cpu()

    logits = {"imu_to_motion": logits_per_imu, "motion_to_imu": logits_per_motion}
    ground_truth = torch.arange(len(imu_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in k_list:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def get_clip_metrics_multi_pos(imu_features, motion_features, k_list=(1,3,5,10), logit_scale=None):
    """
    Asymmetric late score fusion:
      - motion -> IMU : mean over N IMUs
      - IMU   -> motion : max over N IMUs

    Args
    ----
    imu_features    : (B, N, D)
    motion_features : (B, D)
    k_list          : list of Recall@K
    logit_scale     : optional scalar or log-scale tensor
    """
    B, N, D = imu_features.shape
    imu   = F.normalize(imu_features, dim=-1)
    motion= F.normalize(motion_features, dim=-1)
    imu_flat = imu.reshape(B * N, D)

    # --- compute similarities ---
    scale = (logit_scale.exp().item() if isinstance(logit_scale, torch.Tensor)
             else float(logit_scale) if logit_scale is not None else 1.0)
    logits_i2m = scale * (imu_flat @ motion.t())   # (B*N, B)
    logits_m2i = scale * (motion @ imu_flat.t())   # (B, B*N)

    # --- fuse scores ---
    # IMU→Motion: max over N sensors of sample i
    logits_i2m_fused = logits_i2m.reshape(B, N, B).mean(dim=1)   # (B, B)
    # Motion→IMU: mean over N sensors of sample j
    logits_m2i_fused = logits_m2i.reshape(B, B, N).max(dim=2).values         # (B, B)

    metrics = {}

    def _eval_one(name, logits):
        ranking = torch.argsort(logits, dim=1, descending=True)   # (B,B)
        gt = torch.arange(B, device=logits.device).view(-1,1)
        pos = torch.where(ranking == gt)[1].cpu().numpy()
        metrics[f"{name}_mean_rank"]   = pos.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(pos)) + 1
        for k in k_list:
            k_eff = min(k, B)
            metrics[f"{name}_R@{k}"] = float((pos < k_eff).mean())

    _eval_one("motion_to_imu", logits_m2i_fused)
    _eval_one("imu_to_motion", logits_i2m_fused)
    return metrics