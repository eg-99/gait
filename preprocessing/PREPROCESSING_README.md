# Gait Preprocessing Pipeline

Complete pipeline for processing gait videos into model-ready format.

## Quick Start

### Process a single video
```bash
cd preprocessing
python process_video_pipeline.py video.mp4 --subject_id 001 --sequence_id walk1
```

### With video stabilization (for shaky videos)
```bash
python process_video_pipeline.py shaky_video.mp4 --subject_id 001 --sequence_id walk1 --stabilize
```

### Custom output directory
```bash
python process_video_pipeline.py video.mp4 --subject_id 001 --sequence_id walk1 --output_dir my_data
```

## What It Does

1. **Stabilizes video** (optional) - Removes camera shake
2. **Segments person** - YOLO + SAM2 extract silhouettes
3. **Tracks joints** - Optical flow tracking across frames
4. **Generates GEI** - Averaged silhouette for model input
5. **Saves outputs** - Model-ready `.npy` files

## Output Format
```
preprocessed_data/
└── 001/                          # Subject ID
    ├── 001_walk1_unknown_gei.npy          # (128, 64) - For model training
    ├── 001_walk1_unknown_silhouettes.npy  # (N, 128, 64) - Full sequence
    ├── 001_walk1_unknown_pose.npy         # (N, 33, 3) - Joint trajectories
    └── 001_walk1_unknown_metadata.json    # Frame count, stats, etc.
```

## Full Parameters

### Main Pipeline (`process_video_pipeline.py`)
```bash
python process_video_pipeline.py VIDEO --subject_id ID --sequence_id SEQ [OPTIONS]

Required:
  VIDEO                    Path to input video file
  --subject_id ID          Subject identifier (e.g., '001', '002')
  --sequence_id SEQ        Sequence identifier (e.g., 'walk1', 'run1')

Optional:
  --view_angle ANGLE       Camera view angle (default: 'unknown')
  --output_dir DIR         Output directory (default: 'preprocessed_data')
  --stabilize              Enable video stabilization
  --temp_dir DIR           Temporary directory for intermediate files
```

### Video Stabilization (`stabilize_video.py`)
```bash
python stabilize_video.py INPUT [OPTIONS]

Required:
  INPUT                    Path to input video file

Optional:
  --output PATH            Output video path (default: INPUT_stabilized.mp4)
  --smoothing N            Smoothing radius in frames (default: 30)
                          Higher = smoother but may lose fast motions
  --border N               Border crop in pixels (default: 50)
                          Larger crop = less black edges but smaller output
```

**Stabilization tips:**
- `--smoothing 50` for very shaky handheld footage
- `--smoothing 15` for slight shake
- `--border 20` for less cropping (but more black edges visible)
- `--border 80` for maximum edge removal (smaller output size)

### Video Segmentation (`segment_video.py`)
```bash
python segment_video.py VIDEO [OPTIONS]

Required:
  VIDEO                    Path to input video file

Optional:
  --output-dir DIR         Output directory (default: output_silhouettes_VIDEONAME)
  --object-name NAME       Object to detect (default: 'person')
                          Options: person, dog, cat, car, etc.
  --object-id ID           COCO class ID (auto-inferred from --object-name)
  --width W                Output image width (maintains aspect if height not set)
  --height H               Output image height (maintains aspect if width not set)
  --display                Show live tracking preview
  --yolo-interval N        Re-detect with YOLO every N frames (default: 10)
  --sam-imgsz SIZE         SAM processing image size (default: 512)
```

**Segmentation tips:**
- `--display` to visually verify tracking quality
- `--object-name dog` to track dogs instead of people
- `--width 512 --height 512` for custom output resolution

## Requirements

Already installed if you followed setup:
- Python 3.11
- ultralytics (YOLO + SAM2)
- mediapipe (pose detection)
- opencv-python
- numpy, tqdm

## Individual Tools

### Video Stabilization Only
```bash
# Basic stabilization
python stabilize_video.py input.mp4 --output stable.mp4

# Aggressive smoothing for very shaky video
python stabilize_video.py shaky.mp4 --output stable.mp4 --smoothing 50

# Less border cropping
python stabilize_video.py input.mp4 --output stable.mp4 --border 20
```

### Segmentation Only
```bash
# Basic segmentation with live preview
python segment_video.py video.mp4 --display

# Segment a dog
python segment_video.py video.mp4 --object-name dog

# Custom output size
python segment_video.py video.mp4 --width 256 --height 256
```

### Custom Processing
```python
from gait_preprocessor_v2 import GaitPreprocessor

preprocessor = GaitPreprocessor(silhouette_size=(64, 128))
frames = preprocessor.load_video("video.mp4")

# With tracking enabled (recommended)
gait_data = preprocessor.process(
    frames, 
    subject_id="001", 
    sequence_id="walk1", 
    view_angle="090", 
    use_tracking=True
)

preprocessor.save(gait_data, "output_dir")
preprocessor.close()
```

## Notes

- **GEI shape:** Always (128, 64) - matches model requirements
- **Joint tracking:** Uses optical flow + MediaPipe re-detection every 30 frames
- **Stabilization:** Crops border to remove black edges (trade-off for stability)
- **Subject IDs:** Use different IDs (001, 002, etc.) for different people
- **Sequence IDs:** Use descriptive names (walk1, run1, etc.)
- **View angles:** Use standard angles (090, 180, etc.) or 'unknown'

## Troubleshooting

**"No person detected"**: Make sure person is visible in first frame

**"Valid poses: 0/N"**: Check that input is RGB video (not silhouettes)

**Black borders after stabilization**: Reduce `--border` parameter:
```bash
python stabilize_video.py input.mp4 --output stable.mp4 --border 20
```

**Person tracking lost mid-video**: Decrease `--yolo-interval` in segment_video.py:
```bash
python segment_video.py video.mp4 --yolo-interval 5
```

**Out of memory**: Reduce batch processing or close other applications
