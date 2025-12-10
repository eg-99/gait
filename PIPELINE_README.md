# Gait Abnormality Detection Pipeline

Detects normal vs abnormal gait from video input.

## Quick Start

```bash
python complete_pipeline.py VIDEO_PATH
```

## Examples

```bash
# Basic usage (uses best model by default)
python complete_pipeline.py walking_video.mp4

# Save visualizations (silhouettes, GEIs)
python complete_pipeline.py walking_video.mp4 --visualize --output results/

# Get JSON output
python complete_pipeline.py walking_video.mp4 --json
```

## Models

| Flag | Model | Use Case |
|------|-------|----------|
| `--model exp1_vae` | VAE trained on CASIA-B | **Default. Best for binary detection (96.9% acc)** |
| `--model exp1_contrastive` | Contrastive VAE on CASIA-B | Alternative (needs different threshold) |
| `--model exp3_vae` | VAE trained on pathology data | Multi-class tasks |
| `--model exp3_contrastive` | Contrastive VAE on pathology | Best multi-class (89.5% acc) |

**Recommendation:** Use the default `exp1_vae` for normal/abnormal classification.

## All Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Which model to use | `exp1_vae` |
| `--output DIR` | Save results to directory | current dir |
| `--visualize` | Save silhouettes and GEI images | off |
| `--json` | Print results as JSON | off |
| `--frames-per-cycle` | Frames per gait cycle | 30 |
| `--checkpoint-dir` | Path to checkpoints folder | `experiments_final/checkpoints` |

## Required Files

```
experiments_final/checkpoints/
└── exp1_vae_casia.pth    # Required for default model

preprocessing/                # YOLO & SAM2 models (auto-downloaded if missing)
├── yolov8n.pt
└── sam2_t.pt
```

## Output

```
FINAL RESULT: NORMAL | ABNORMAL
Confidence: XX.XX%
Votes: X abnormal, X normal
```

## How It Works

1. **Extract silhouettes** - YOLO detects person, SAM2 segments them
2. **Group into gait cycles** - ~30 frames per cycle
3. **Generate GEIs** - Average silhouettes into Gait Energy Images
4. **Run inference** - VAE reconstruction error determines normal/abnormal
5. **Aggregate** - Majority vote across cycles

High reconstruction error = gait doesn't match learned "normal" patterns = ABNORMAL
