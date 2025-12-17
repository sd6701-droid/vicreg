# VICReg Pretraining (ResNet-50x2) on a Flat Image Folder (96×96)

This repository trains a **VICReg** (Variance–Invariance–Covariance Regularization) self-supervised model on an **unlabeled flat folder of images** (no class subfolders required).

Key features:
- **ResNet backbones** (e.g., `resnet50x2`)
- **Global VICReg loss** (invariance + variance + covariance)
- Optional **Local (VICRegL-style) loss** on feature maps (`--use-local-loss`)
- Optional **ViewMix** augmentation (implemented; disabled unless enabled)
- **LARS** optimizer + **warmup + cosine** learning-rate schedule
- Optional **Weights & Biases (W&B)** logging

---

## Pretrained Models
Pre-trained Models
You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and projection head weights.
[ckpt](https://drive.google.com/file/d/1qTiBTeOpE-zEGb6toaAa45FDXtecGaRp/view?usp=sharing)

## Repository Layout

- `main_vicreg.py` — training script
- `augmentations.py` — `TrainTransform` that generates two augmented views
- `resnet.py` — backbone definitions (must support `return_featmap=True` if local loss is enabled)
- `distributed.py` — distributed helpers (**not used** in the provided single-GPU run)

---

## Requirements

- Python 3.9+
- PyTorch + torchvision (CUDA recommended)
- `wandb` (optional)

Example:
```bash
conda create -n vicreg python=3.10 -y
conda activate vicreg
pip install torch torchvision wandb
```

---

## Dataset Format (Flat Folder)

Your dataset must be a single directory containing images directly (no class subfolders):

```
/path/to/all_images/train/
  img_000001.jpg
  img_000002.png
  ...
```

Supported extensions (as coded): `jpg, jpeg, png, bmp, tif, tiff`

### Notes
- The dataset class (`FlatImageFolder`) **skips corrupted images** (up to 10 retries).
- Each sample returns two augmented views `(v1, v2)` and a dummy label `0`.

### Using 500K + Additional 200K Images
If you have **500K original + 200K supplemental** images, the simplest setup is to **merge them into one folder** (≈700K total):
```
all_images/train/   # contains both sources
```
(Alternatively, use symlinks to avoid duplicating storage.)

---

## Data Augmentations (`TrainTransform`)

The training script uses:
```python
transforms = aug.TrainTransform()
```
Your `TrainTransform` returns two views `(v1, v2)` per image. Transforms used include:
- RandomResizedCrop (size=96, scale 0.2–1.0, bicubic)
- RandomHorizontalFlip
- RandomRotation (small degrees)
- ColorJitter (via RandomApply)
- RandomGrayscale
- GaussianBlur (strong for view1, weak for view2)
- ToTensor + ImageNet normalization
- Solarization is present but disabled if `p=0.0`

---

## Training (Single GPU)

Example run command (matches your provided configuration):

```bash
python main_vicreg.py   --data-dir /home/sd6701/datasets/dl_vicreg_final_exp_dataset_v2/all_images/train/   --exp-dir /home/sd6701/fall2025_deeplearning/final-opts/vicreg_cc3m_resnet50x2_v13/   --arch resnet50x2   --epochs 500   --batch-size 1024   --base-lr 0.25   --log-freq-time 5   --device cuda   --mlp 2048-2048-1024   --use-local-loss   --local-mlp 1024-512   --local-loss-weight 0.2   --enable-wandb   --wandb-project dl_final_vicreg   --wandb-entity sd6701-new-york-university   --wandb-name r50_augs_glo_loc_13
```

### Learning Rate Schedule (as implemented)
- Warmup for **10% of epochs** (e.g., 50 epochs when training 500 epochs)
- Then cosine decay to **0.001 × peak LR**
- Peak LR is computed in code as: `base_lr * batch_size / 256`

---

## SLURM Script (Example)

Submit:
```bash
sbatch run_vicreg.slurm
```

**Important:** when splitting the command across lines, every continued line must end with `\`.

A corrected snippet (note the `\` after `--wandb-name ...`):

```bash
python main_vicreg.py   --data-dir /home/sd6701/datasets/dl_vicreg_final_exp_dataset_v2/all_images/train/   --exp-dir /home/sd6701/fall2025_deeplearning/final-opts/vicreg_cc3m_resnet50x2_v13/   --arch resnet50x2   --epochs 500   --batch-size 1024   --base-lr 0.25   --log-freq-time 5   --device cuda   --enable-wandb   --wandb-project dl_final_vicreg   --wandb-entity sd6701-new-york-university   --wandb-name r50_augs_glo_loc_13   --mlp 2048-2048-1024   --use-local-loss   --local-mlp 1024-512   --local-loss-weight 0.2
```

Tips:
- If SLURM allocates `--cpus-per-task=8`, consider setting `--num-workers 8` to match.
- Ensure SLURM wall-time is sufficient for 500 epochs.

---

## Outputs and Checkpoints

All outputs are written to `--exp-dir`.

### Files
- `stats.txt` — periodic JSON logs
- `model.pth` — checkpoint saved every epoch (contains `epoch`, `model`, `optimizer`)
- `<arch>.pth` — final backbone-only weights (e.g., `resnet50x2.pth`)

### Resume (Already Supported)
If this file exists:
```
<exp-dir>/model.pth
```
the script automatically resumes from it. Just rerun the same command with the same `--exp-dir`.

---

## Optional: Add a `--resume` Flag (Load Any Checkpoint Path)

Right now the script only loads `<exp-dir>/model.pth`. If you want to resume from an arbitrary checkpoint path, add:

### 1) Add argument
```python
parser.add_argument("--resume", type=Path, default=None,
                    help="Path to checkpoint to resume from")
```

### 2) Replace the resume block with
```python
ckpt_path = args.resume if args.resume is not None else (args.exp_dir / "model.pth")
if ckpt_path.is_file():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    start_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
else:
    start_epoch = 0
```

---

## W&B Notes (Recommended Practice)

Avoid hardcoding API keys in code. Prefer:
```bash
export WANDB_API_KEY="YOUR_KEY"
```
Then use `--enable-wandb` and the project/entity/name flags.

---

## License / Attribution

This training code is derived from Meta’s VICReg implementation and is governed by the LICENSE file in the original source tree (if included in your project).
