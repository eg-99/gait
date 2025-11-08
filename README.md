# Computer vision tools for video segmentation and gait recognition.

## Features

- **Video Segmentation**: Extract object silhouettes from videos using YOLO + SAM2
- **Gait Recognition**: Preprocessing and models for gait analysis

## Installation

```bash
# Python 3.10.18 works (and some other versions as well)
pip install -r requirements.txt

# Install SAM2 (if needed)
pip install sam2
# pip install 'git+https://github.com/facebookresearch/sam2.git'
```

## Usage

### Video Segmentation

Extract object silhouettes from videos:

```bash
# Basic usage (detects person by default)
python segment_video.py video.mp4

# Detect specific object
python segment_video.py video.mp4 --object-name dog

# Custom output size and display
python segment_video.py video.mp4 --width 512 --height 512 --display

# See all options
python segment_video.py --help
```

**Output**: White silhouettes on black background saved as PNG frames.

### Gait Preprocessing

Process CASIA-B dataset for gait recognition:

```bash
cd gait/preprocessing
python casia_b_loader.py \
    --dataset_root /path/to/CASIA-B \
    --output_root preprocessed_data \
    --create_splits
```

Generates silhouettes, GEI (Gait Energy Images), and pose data.

### Gait Recognition Models

Train and use models for gait recognition:

```bash
# GEI-CNN (classification)
cd gait/models/gei_cnn
python train.py --epochs 50 --batch_size 32

# Autoencoder (feature extraction)
cd gait/models/autoencoder
python train.py --epochs 50 --batch_size 16
python extract_embeddings.py --checkpoint checkpoints/best_model.pth

# VAE (variational feature extraction)
cd gait/models/vae
python train.py --epochs 50 --batch_size 16 --beta 1.0
python extract_embeddings.py --checkpoint checkpoints/best_model.pth
```

See `gait/models/README.md` for detailed usage and examples.

## Project Structure

```

├── gait/
│   ├── preprocessing/         # Gait data preprocessing
│   └── models/               # Gait recognition models
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.10.18 (some other versions work as well)
- PyTorch
- OpenCV
- Ultralytics (YOLO/SAM)
- MediaPipe (gait preprocessing)

See `requirements.txt` for complete list.

