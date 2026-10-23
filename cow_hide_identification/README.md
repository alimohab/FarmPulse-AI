# Cow Hide-Pattern Identification

Identify individual cows from their unique hide patterns using a **Siamese network** trained with
**deep metric learning** (triplet margin loss with online semi-hard mining).

- **Backbone:** ImageNet-pretrained ResNet-50 (transfer learning), shared by both Siamese branches.
  You can swap it via `model.backbone` in the config.
- **Embedding:** 128-d by default, L2-normalised. Two images are compared with **cosine similarity**.
- **Training:** P×K batches (P cow/view groups × K images), so every batch contains positives and
  negatives. Positives share the cow **and the view** (left/right/top). Cross-view training is one
  config switch away (`training.cross_view_positives: true`).
- **Evaluation:** retrieval of validation/test images against a gallery of known images, reported as
  top-1 accuracy, Recall@K and mAP.

> No trained model ships with this project. `models/checkpoints/best_model.pth` only exists after
> you have trained on your own dataset.

## Project layout

```text
cow_hide_identification/
├── configs/training_config.yaml      all settings (paths, data, model, training, evaluation)
├── data/raw/                         your dataset: Cow_XXX/<view>/...
├── data/processed/                   optional YOLO crops (scripts/crop_cows.py)
├── data/splits/                      train.csv / val.csv / test.csv (generated, reproducible)
├── models/checkpoints/               best_model.pth, last_checkpoint.pth
├── outputs/                          train.log, training_history.csv, evaluation_*.json
├── scripts/                          train_model.py, evaluate_model.py, compare_images.py, crop_cows.py
├── src/cowid/                        the library (data, detection, reid, training, utils)
└── tests/                            pytest tests
```

## 1. Installation (Windows, PowerShell)

```powershell
cd C:\Users\Lenovo\FarmPulse-AI-1\cow_hide_identification
python -m venv .venv
.venv\Scripts\Activate.ps1          # cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**NVIDIA GPU (optional):** `pip install -r requirements.txt` installs the CPU build of PyTorch.
For CUDA, install the matching build first (pick the CUDA version at https://pytorch.org/get-started/locally/), e.g.:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Training uses the GPU automatically when one is available (`training.device: auto`), and the CPU
otherwise.

**Optional:** `pip install ultralytics` is only needed for YOLO cow cropping (`crop_cows.py`,
`compare_images.py --detect`).

The first training run downloads the ImageNet ResNet-50 weights (~100 MB) into the torch cache.

## 2. Organise the dataset

```text
data/raw/
├── Cow_001/
│   ├── left/     img_0001.jpg ...
│   ├── right/    ...
│   └── top/      ...
├── Cow_002/
│   ├── left/
│   ├── right/
│   └── top/
└── ...
```

- One folder per cow. The folder name is the label, so use any naming (`Cow_001`, `Bella`, ...).
- View folders must be named `left`, `right` or `top`. Any subset per cow is fine.
- Supported formats: `.jpg .jpeg .png .bmp .webp`. Phone-photo EXIF rotation is respected.
- Use tight crops of one cow per image if possible. For full camera frames, see step 6.
- You need **at least 2 cows** and **at least 2 images per cow and view** to form triplets. With
  ≥ 3 images per cow and view, every cow also gets validation/test images. Aim for many more:
  20+ images per cow and view from different days and lighting.

**Recording sessions (strongly recommended).** Frames from the same video are near-duplicates. If
they land in both train and test, the scores look better than they really are. Put each
recording session in its own sub-folder:

```text
data/raw/Cow_001/left/2024-10-15_morning/frame_000120.jpg
data/raw/Cow_001/left/2024-10-16_evening/frame_000360.jpg
```

Cows with at least `data.min_sessions_for_grouping` (default 3) sessions are split **by session**.
Others fall back to an image-level split, with a warning. You can also give sessions in a CSV
(`image,session` columns, image path relative to `data/raw`) via `data.session_metadata`.

## 3. Train

```powershell
python scripts\train_model.py
```

- The first run scans `data/raw`, validates it, writes the reproducible splits to `data\splits\`
  (seeded by `seed`) and trains.
- After adding images, regenerate the splits: `python scripts\train_model.py --resplit`.
- Resume an interrupted run: `python scripts\train_model.py --resume`.
- To train longer, resume with more epochs: `python scripts\train_model.py --resume --epochs 80`.
- Quick overrides: `--epochs`, `--batch-size`, `--device cpu|cuda`, `--config path\to\other.yaml`.

Each epoch logs train loss, validation loss, top-1, Recall@K, mAP and learning rate to the console
and to `outputs\train.log`. The full per-epoch table goes to `outputs\training_history.csv`.

### Where the best model is saved

| File | Written when | Use |
|------|--------------|-----|
| `models\checkpoints\best_model.pth` | **only** when the monitored validation metric (`training.monitor`, default `top1`) improves by more than `early_stopping.min_delta` | inference and evaluation |
| `models\checkpoints\last_checkpoint.pth` | after every epoch | resuming training |

Each checkpoint contains:
- model and optimizer state dicts
- scheduler/AMP state
- epoch and best validation metric
- model configuration and embedding dimension
- image size, full training config, split fingerprint and training history

A worse final epoch never overwrites `best_model.pth`. A **fresh** (non-resumed) run first moves
any existing checkpoints to `models\checkpoints\archive\<timestamp>\`.

Early stopping ends training after `early_stopping.patience` epochs without improvement.

## 4. Evaluate

```powershell
python scripts\evaluate_model.py
```

This loads `best_model.pth` and uses the **test** images as queries against a gallery of the
train + val images (same view only, by default). It prints and saves
`outputs\evaluation_test.json` with:
- top-1, Recall@K and mAP
- test triplet loss
- the mean cosine similarity of same-cow vs. different-cow pairs

Use `--split val` or `--checkpoint <path>` to change what is evaluated.

## 5. Compare two images

```powershell
python scripts\compare_images.py path\to\cow_a.jpg path\to\cow_b.jpg
python scripts\compare_images.py a.jpg b.jpg --threshold 0.75     # also print SAME / DIFFERENT
python scripts\compare_images.py a.jpg b.jpg --detect             # crop the cow with YOLO first
```

The score is the cosine similarity of the two embeddings, from -1 to 1. Higher means more likely
the same cow. There is no universal threshold. Choose one between the same-cow and different-cow
mean similarities that `evaluate_model.py` reports for your data. Compare images of the same view
(left with left, and so on), because the model is trained on same-view pairs.

## 6. Optional: crop cows from full frames

If your images are whole camera frames, crop the largest cow in each image with YOLO and train on
the crops:

```powershell
pip install ultralytics
python scripts\crop_cows.py --model ..\cow-detection\models\yolo11m.pt   # or yolo11s.pt (auto-download)
```

This mirrors `data\raw` into `data\processed`. Review the crops, because the largest cow is not
always the labelled one. Then set `paths.dataset_dir: data/processed` and train with `--resplit`.

## Configuration highlights (`configs/training_config.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `model.backbone` | `resnet50` | also `resnet18/34/101`, `efficientnet_b0/b2`. Add more with `cowid.reid.backbone.register_backbone` |
| `model.embedding_dim` | `128` | embedding size |
| `training.epochs`, `batch_size`, `learning_rate` | `50`, `32`, `3e-4` | `batch_size` = P × `samples_per_class` |
| `training.margin` | `0.2` | triplet margin, in cosine distance |
| `training.freeze_backbone_epochs` | `2` | warm up the head before fine-tuning the backbone |
| `training.backbone_lr_multiplier` | `0.1` | slower fine-tuning of the pretrained backbone |
| `training.monitor` | `top1` | metric that selects `best_model.pth` (`top1`, `mAP`, `recall@K`, `val_loss`) |
| `training.early_stopping` | patience 10 | stops after N epochs without improvement |
| `training.cross_view_positives` | `false` | `true` treats left/right/top of a cow as positives |
| `evaluation.same_view_retrieval` | `true` | match queries only against same-view gallery images |
| `data.image_size` | `[224, 224]` | `[height, width]`; elongated side views may benefit from `[224, 448]` |

Augmentations keep the hide pattern intact:
- **Used:** mild crops, rotation and perspective; brightness/contrast; blur; small occlusions.
- **Not used:** no flips, because a mirrored flank is a pattern that doesn't exist on the cow. No hue
  shifts either.

## Tests

```powershell
python -m pytest
```

The tests build tiny synthetic image folders in a temporary directory. They check:
- model output size and embedding normalisation
- weight sharing
- triplet mining and loss
- dataset loading, splitting and validation
- metrics
- checkpoint save/load
- a short end-to-end training run

The synthetic images are used for these code checks only and never for training.
