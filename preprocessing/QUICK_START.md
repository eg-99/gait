# Gait Recognition - Preprocessing Pipeline

Extract silhouettes and GEI from walking videos for gait recognition models.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
python test_installation.py  # Verify setup
```

### Option 1: Process CASIA-B Dataset
```bash
python casia_b_loader.py \
    --dataset_root /path/to/CASIA-B \
    --output_root output \
    --create_splits
```

### Option 2: Process Your Own Videos
```bash
python extract_silhouettes_ai.py \
    --input your_video.mp4 \
    --output silhouettes_output
```

## What You Get

After preprocessing, you'll have:
- **Silhouettes**: Binary images (64×128) of walking person
- **GEI**: Gait Energy Image (temporal average of silhouettes)
- **Train/Val/Test splits**: Automatically created by subject

## File Guide

| File | Purpose |
|------|---------|
| `casia_b_loader.py` | Batch process CASIA-B dataset |
| `extract_silhouettes_ai.py` | Extract silhouettes from any video (AI-based) |
| `data_loader.py` | PyTorch DataLoaders for training |
| `visualization.py` | View preprocessed data |
| `gait_preprocessor.py` | Core preprocessing classes |

## For Model Training

Load preprocessed data in your training script:
```python
from data_loader import GaitDataLoader

loaders = GaitDataLoader.create_loaders(
    data_root='output',
    data_type='gei',  # or 'silhouettes'
    batch_size=32
)

for gei_images, labels, metadata in loaders['train']:
    # gei_images: (32, 1, 128, 64)
    # Train your model here
    pass
```

See `TRAINING_EXAMPLES.md` for more examples.

## Output Structure

```
output/
├── 001/                           # Subject folders
│   ├── 001_nm-01_090_gei.npy
│   ├── 001_nm-01_090_silhouettes.npy
│   └── 001_nm-01_090_metadata.json
├── data_splits.json               # Train/val/test splits
└── dataset_statistics.json
```

## Notes

- **CASIA-B**: Preprocessing works out of the box (silhouettes already provided)
- **Custom videos**: Uses AI (rembg) - works on any background but slower (~2-5 sec/frame)
- **Pose data**: Only works with RGB images (not CASIA-B silhouettes)

## Docs

- `QUICK_START.md` - Detailed setup guide
- `TRAINING_EXAMPLES.md` - Code examples for loading data
- `SILHOUETTE_EXTRACTION.md` - Custom video processing guide

## Requirements

- Python 3.11 (MediaPipe doesn't support 3.13 yet)
- ~2GB disk space for dependencies
- ~20GB for CASIA-B dataset (if using)

## Troubleshooting

**"No module named cv2"**: `pip install opencv-python`

**"MediaPipe not found"**: Only needed for pose estimation (optional)

**Silhouettes look bad**: Use `extract_silhouettes_ai.py` with `--every 2` to skip frames and speed up

Run `python test_installation.py` to verify everything works.
