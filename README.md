# MoBind: Motion Binding for Fine-Grained IMU–Video Pose Alignment

Duc Duy Nguyen, Tat-Jun Chin, Minh Hoai

*Australian Institute for Machine Learning (AIML), Adelaide University* 

<div>
    🎉 <strong>Accepted to CVPR 2026</strong>
</div>

[[`Project Page`]()] [[`Paper`](https://arxiv.org/pdf/2602.19004)] [[`Demo`]()] [[`BibTex`](#Citation)]


ImageBind learns a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation.

# Environment Setup

We tested our code on Ubuntu 24.04 with `Python 3.10`, `Pytorch 2.11.0` with `cuda 12.8`, other dependencies are specified in `requirements.txt`.

```bash
conda create -n mobind python=3.10 -y
conda activate mobind
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```


# Datasets and Models

### Configuration

Before running any script, set your paths in `configs/config.py`:

```python
DATA_ROOT        = "/path/to/datasets"       # root directory for all datasets
BODY_MODEL_PATH  = "/path/to/smpl/basicmodel_m.pkl"  # required for EgoHumans only
```

### Datasets

We use three publicly available datasets. Download the raw data from:

- [mRi](https://github.com/sizhean/mri)
- [EgoHumans](https://rawalkhirodkar.github.io/egohumans/)
- [mmPose extraction]()

### SMPL Model

Required for EgoHumans preprocessing. Download from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and set `BODY_MODEL_PATH` in `configs/config.py`.

---

### Data Processing

Each dataset goes through the same two-stage pipeline: **extract → cache**.

The final cache (`cache_subject_5_2` / `cache_action_5_2`) is used for **training and all evaluations except sync**.  
The sync cache (`cache_sync_*`) is built separately and used **only for synchronization evaluation**.

#### mRi

**1. Download raw data** into `$DATA_ROOT/mRi/`.

**2. Run MMPose** on the mRi videos and place the detection JSON results at:
```
$DATA_ROOT/mRi/extracted_pose/subject{N}_results_det.json
```

**3. Extract** — copies IMU CSVs per limb and converts MMPose detections to `.npy`:
```bash
python preprocess/mRi/extract_data.py
```

**4. Build contrastive cache** (training, retrieval, localization):
```bash
python preprocess/mRi/cache.py --window_sec 5 --stride 2
```

**5. Build sync cache** (sync evaluation only):

We provide predefined `annotations.txt` files to ensure reproducible evaluation splits. Pass them via `--anno_file`:
```bash
python preprocess/mRi/cache_sync.py --anno_file preprocess/mRi/annotations.txt
```
---

#### EgoHumans

**1. Download raw data** into `$DATA_ROOT/EgoHumans/`.

**2. Extract** — unzips archives, extracts IMU signals and 3D pose, saves per-sequence `.npy` files:
```bash
python preprocess/EgoHumans/extract_data.py
```

**3. Build contrastive cache** (training, retrieval, localization):
```bash
python preprocess/EgoHumans/cache.py --window_sec 5 --stride 2
```

**4. Build multi-person cache** (multi-person retrieval eval only):
```bash
python preprocess/EgoHumans/cache_multi_person.py --window_sec 5 --stride 2
```

**5. Build sync cache** (sync evaluation only):

We provide predefined `annotations.txt` files to ensure reproducible evaluation splits. Pass them via `--anno_file`:
```bash
python preprocess/EgoHumans/cache_sync.py --anno_file preprocess/EgoHumans/annotations.txt
```
---

The processed data directory should look like this:

```
$DATA_ROOT/
├── mRi/
│   ├── aligned_data/          # raw download
│   ├── extracted_pose/        # MMPose detection JSONs (place here manually)
│   ├── processed_data/        # output of extract_data.py
│   ├── cache_subject_5_2/     # contrastive cache → training & eval
│   └── cache_sync_subject_20_5/   # sync cache → sync eval only
│
└── EgoHumans/
    ├── data/                  # raw download
    ├── extracted_data/        # output of extract_data.py
    ├── cache_action_5_2/      # contrastive cache → training & eval
    ├── cache_action_multi_5_2/    # multi-person cache → retrieval eval
    └── cache_sync_action_20_5/    # sync cache → sync eval only
```

## Training

MoBind uses a two-stage training pipeline. Run Stage 1 first, then set `model.stage1_exp` in the Stage 2 config to the Stage 1 output directory before running Stage 2.

```bash
# Stage 1: per-limb contrastive learning
python train_contrastive.py --config configs/mRi/MoBind_stage1.yaml

# Stage 2: multi-limb contrastive + MAE reconstruction
python train_contrastive.py --config configs/mRi/MoBind_stage2.yaml
```

Replace `mRi` with `EgoHumans` for the EgoHumans dataset. Experiments are saved to `./outputs/{stage}/{dataset}/{timestamp}/`.

## Evaluation

All evaluation scripts accept `--exp_dir` pointing to a training output directory (which contains `config.yaml` and `checkpoints/best.pt`).

**Cross-modal retrieval** (mRi or EgoHumans):
```bash
python eval_retrieval.py --exp_dir <path/to/experiment>
```

**Person and limb localization** (EgoHumans only):
```bash
python eval_localization.py --exp_path <path/to/stage2/experiment> --task all
```
Use `--task person` or `--task limb` to run a single evaluation.

**Temporal synchronization** (mRi):
```bash
python eval_sync.py --exp_dir <path/to/experiment> --test_dataset mRi
```

**Temporal synchronization** (EgoHumans):
```bash
# person-level
python eval_sync_egoh.py --exp_dir <path/to/experiment> --task person

# video-level (multi-person)
python eval_sync_egoh.py --exp_dir <path/to/experiment> --task video
```

# Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```
@InProceedings{nguyen2026mobind,
  author    = {Nguyen, Duc Duy and Chin, Tat-Jun and Hoai, Minh},
  title     = {MoBind: Motion Binding for Fine-Grained IMU--Video Pose Alignment},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
