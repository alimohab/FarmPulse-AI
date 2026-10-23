# Cow Hide Identification: Project Notes and Next Steps

These notes summarise the session in which the `cow_hide_identification` project was built. They
record what was built, how it was checked, where things stand now, and **what to do once the
dataset is ready**.

---

## 1. What was asked

Build a complete, modular Python project that identifies individual cows from their hide patterns
using **Siamese networks and deep metric learning**. It should run in VS Code on Windows. The
requirements were:

- PyTorch/torchvision with a pretrained **ResNet-50** backbone, shared weights, and a configurable
  **128-d L2-normalised embedding**.
- **Triplet margin loss** with **semi-hard triplet mining** and **cosine similarity**.
- A dataset loaded from `data/raw/Cow_XXX/{left,right,top}/`, with reproducible
  train/val/test splits (grouped by recording session when possible) and clear errors when the
  data is insufficient.
- A config-driven training pipeline:
  - CUDA when available, otherwise CPU
  - early stopping
  - top-1 accuracy and Recall@K
  - logging and resuming
- `best_model.pth` saved **only when the validation metric improves**, plus `last_checkpoint.pth`
  for resuming.
- Scripts to evaluate the test set and to compare two images.
- A README with Windows commands, and basic tests.
- Nothing installed automatically, and no claims of a trained model before real training.

---

## 2. What was built

```text
cow_hide_identification/
├── README.md                     full usage guide
├── PROJECT_NOTES.md              this file
├── requirements.txt
├── pytest.ini
├── configs/training_config.yaml  all settings
├── data/raw/                     <- your dataset goes here (currently empty)
├── data/processed/               optional YOLO crops
├── data/splits/                  train/val/test CSVs (generated on first training run)
├── models/checkpoints/           best_model.pth / last_checkpoint.pth (after training)
├── scripts/
│   ├── train_model.py            train / resume / regenerate splits
│   ├── evaluate_model.py         evaluate best_model.pth on the test split
│   ├── compare_images.py         cosine similarity between two cow images
│   └── crop_cows.py              optional: crop cows from full frames with YOLO
├── src/cowid/
│   ├── config.py                 config loading + project-root-aware paths
│   ├── inference.py              CowEmbedder: embed images, compare two images
│   ├── data/                     dataset.py, augmentations.py, split_dataset.py
│   ├── detection/detector.py     optional YOLO cow cropper
│   ├── reid/                     backbone.py, siamese_network.py, losses.py,
│   │                             triplet_mining.py, similarity.py
│   ├── training/                 train.py, validate.py, metrics.py
│   └── utils/                    device.py, checkpoint_utils.py, logging_utils.py
└── tests/                        conftest.py, test_dataset.py, test_model.py
```

### Key design decisions

| Topic | Decision |
|-------|----------|
| Model | ResNet-50 (ImageNet) → Linear 512 → BN → ReLU → Dropout → Linear 128 → L2 normalise. Both Siamese branches use the same module, so weights are shared by construction. |
| Swappable backbone | `model.backbone` accepts `resnet18/34/50/101` and `efficientnet_b0/b2`. Register new ones with `register_backbone()`. |
| Transfer learning | The backbone is frozen for the first 2 epochs, then fine-tuned at 0.1× the head's learning rate. |
| Loss | Triplet margin loss on cosine distance (`1 − cos`), margin 0.2. |
| Mining | FaceNet-style semi-hard: for each anchor/positive pair, take the closest negative that is still farther away than the positive. |
| Positives | Same cow **and same view** (left with left). `cross_view_positives: true` enables cross-view training later. |
| Batches | P×K sampling (P cow/view groups × K images), so every batch has positives and negatives. |
| Splits | Every cow appears in train (the gallery); its other images go to val/test (the queries). When a cow has ≥ 3 known recording sessions, whole sessions stay in one split. |
| Augmentation | Mild crops, rotation and perspective; brightness/contrast; blur; small occlusions. **No flips** (a mirrored flank is a fake pattern) and **no hue shifts**. |
| Metrics | Top-1, Recall@K (only where the gallery is large enough) and mAP. By default, queries are matched only against same-view gallery images. |
| Best checkpoint | Written **only** when `training.monitor` (default `top1`) improves by more than `min_delta`. A fresh run archives old checkpoints instead of overwriting them. |
| Checkpoint contents | Model and optimizer state, scheduler/AMP state, epoch, best metric, model config, embedding dim, image size, full config, split fingerprint, training history. |

---

## 3. How it was checked

- **Tests:** all 21 pass. pytest wasn't installed, so they ran through a temporary stand-in runner.
  Once pytest is installed, run `python -m pytest` to confirm.
- **Scripts:** training, resuming, evaluation and image comparison ran end to end on a small set of
  fake images in a temporary folder. Nothing was written into the project.
- **Best checkpoint:** `best_model.pth` was written at epoch 1 and was **not** overwritten by later
  epochs that didn't improve.
- **Pretrained backbone:** ResNet-50 loads from the local torch cache and outputs normalised
  128-d embeddings.
- **Bugs found and fixed:**
  1. Checkpoints failed to reload with PyTorch's safe loader because the torch version was stored
     as a special object. It's now stored as plain text.
  2. Resuming with more epochs left the learning rate at 0. The cosine schedule now follows the
     current `epochs` value, and the learning rate is recalculated on resume.

---

## 4. Current status

- ✅ Code complete and tested.
- ❌ **No real dataset yet:** `data/raw/` is empty.
- ❌ **No trained model:** `models/checkpoints/` is empty. Any accuracy number from the session
  came from fake test images and means nothing for real cows.
- ⚠️ `pytest` is not installed in the repo's `.venv`. Everything else needed (torch, torchvision,
  Pillow, PyYAML, numpy, ultralytics) is already there.
- ⚠️ The installed PyTorch is the **CPU** build (no CUDA), so training ResNet-50 will be slow.

---

## 5. ✅ What to do when the data is ready

### Step 1: Collect and sort the images

1. Extract frames from each video. The repo's `frame_extraction.py` does this; give each video
   **its own output folder** so it can serve as a recording session.
2. Sort the frames into one folder per cow and one sub-folder per view. Keep the session (video or
   day) as a sub-folder:

   ```text
   cow_hide_identification\data\raw\
   ├── Cow_001\
   │   ├── left\
   │   │   ├── hiv00005\  frame_000120.jpg ...
   │   │   └── hiv00006\  ...
   │   ├── right\
   │   └── top\
   ├── Cow_002\
   └── ...
   ```

3. Check the minimums:
   - **at least 2 cows**
   - **at least 2 images per cow and view** (the code refuses to train otherwise)
   - **at least 3 images per cow and view** so each cow also gets validation/test images
   - recommended: **20+ images per cow and view**, from **3+ different sessions** (days, videos,
     lighting)
4. Each image should show **one cow**, ideally cropped tightly. If your images are full camera
   frames, see Step 3.
5. View folders must be named exactly `left`, `right` or `top`. Images directly inside `Cow_XXX\`
   are ignored, with a warning.

### Step 2: Set up the environment (once)

```powershell
cd C:\Users\Lenovo\FarmPulse-AI-1\cow_hide_identification
..\.venv\Scripts\Activate.ps1
pip install pytest
python -m pytest                 # should report 21 passed
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first, which makes training much
faster:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 3 (optional): Crop cows from full frames

```powershell
python scripts\crop_cows.py --model ..\cow-detection\models\yolo11m.pt
```

Then:
- Review the crops in `data\processed\`; the largest detected cow is not always the labelled one.
- Set `paths.dataset_dir: data/processed` in `configs\training_config.yaml`.

### Step 4: Check the config

Open `configs\training_config.yaml`. The defaults are sensible; on a CPU-only machine consider:

- `model.backbone: resnet18` for a much faster first run (switch back to `resnet50` later).
- `training.batch_size: 16` if you run out of memory.
- `training.epochs`: keep 50. Early stopping ends the run sooner if nothing improves.

### Step 5: Train

```powershell
python scripts\train_model.py
```

- The first run scans `data\raw`, checks it, and writes the splits to `data\splits\`. Read the
  printed split table: every cow should have train **and** val images.
- If you see the warning *"No session grouping for N cow(s)"*, the scores will look better than
  they are. Add session sub-folders if possible.
- Progress is logged to the console and `outputs\train.log`, with per-epoch numbers in
  `outputs\training_history.csv`.
- If training is interrupted: `python scripts\train_model.py --resume`
- If you add or remove images later: `python scripts\train_model.py --resplit` (this starts a fresh
  run).

### Step 6: Evaluate on the test set

```powershell
python scripts\evaluate_model.py
```

Results are printed and saved to `outputs\evaluation_test.json`:
- **top1:** how often the most similar known image is the correct cow.
- **recall@5 / recall@10:** how often the correct cow is among the 5 or 10 most similar images.
- **mAP:** overall ranking quality.
- **Mean similarity, same cow vs. different cows:** use these to choose a decision threshold
  (pick a value between the two).

### Step 7: Compare two images

```powershell
python scripts\compare_images.py path\to\cow_a.jpg path\to\cow_b.jpg
python scripts\compare_images.py path\to\cow_a.jpg path\to\cow_b.jpg --threshold 0.75
```

Compare images of the **same view** (left with left), because the model is trained that way.

### Step 8: Where the trained model is

```text
cow_hide_identification\models\checkpoints\best_model.pth     ← use this one
cow_hide_identification\models\checkpoints\last_checkpoint.pth  (for resuming only)
```

---

## 6. Troubleshooting (messages you may see)

| Message | Fix |
|---------|-----|
| `Dataset folder not found` / `No cow folders found` | Put the images in `data\raw\Cow_XXX\<view>\`. |
| `The train split has 1 cow(s)` | Add at least one more cow. |
| `No valid triplets ... no cow/view combination has at least 2 images` | Add more images per cow **and view**. |
| `The validation split is empty` | Add images (≥ 3 per cow and view), then rerun with `--resplit`. |
| `images listed in data\splits no longer exist` | Images were moved or deleted. Rerun with `--resplit`. |
| `Checkpoint not found ... best_model.pth` | Train first (Step 5). |
| `Could not load ImageNet weights` | The first run needs internet to download ResNet-50 weights, or set `model.pretrained: false` (not recommended). |
| Training is very slow | Use the CUDA build of PyTorch on a GPU, or `model.backbone: resnet18` and a smaller `image_size`. |

---

## 7. Possible next steps (after the first real training)

- Try `data.image_size: [224, 448]` for long side-view crops.
- Enable `training.cross_view_positives: true` once same-view results are good.
- Build a "who is this cow?" tool: embed one reference image per cow and view, then match new
  images to the nearest one. `cowid.inference.CowEmbedder.embed()` already provides the embeddings.
