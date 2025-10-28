# Custom Video/Image Preprocessing

This script allows you to preprocess **any** video or images of people walking into silhouettes and GEI, not just CASIA-B dataset.

## Quick Start

### 1. Process a Video File
```bash
python preprocess_custom_video.py \
    --video my_walking_video.mp4 \
    --output custom_output \
    --subject person1
```

### 2. Process a Folder of Images
```bash
python preprocess_custom_video.py \
    --images path/to/frames/ \
    --output custom_output \
    --subject person1 \
    --pattern "*.jpg"
```

### 3. Record from Webcam
```bash
python preprocess_custom_video.py \
    --webcam \
    --duration 10 \
    --output custom_output \
    --subject person1
```

## Output

The script generates the same output as CASIA-B preprocessing:
- `{subject}_{sequence}_{view}_silhouettes.npy` - Normalized silhouettes
- `{subject}_{sequence}_{view}_gei.npy` - Gait Energy Image
- `{subject}_{sequence}_{view}_gei.png` - GEI visualization
- `{subject}_{sequence}_{view}_pose.npy` - Pose landmarks (if detected)
- `{subject}_{sequence}_{view}_metadata.json` - Sequence info

## Requirements

- Person should be walking **perpendicular** to camera (side view works best)
- Clear background (or person clearly distinguishable from background)
- Good lighting
- Person should be the dominant object in frame

## Tips for Best Results

1. **Lighting**: Use even lighting, avoid harsh shadows
2. **Background**: Simple, contrasting background works best
3. **Distance**: Person should be 5-10 feet from camera
4. **Walking**: Natural walking pace, walk straight across frame
5. **Duration**: Capture at least 2-3 complete gait cycles (about 4-6 seconds)

## Example Workflow

```bash
# Record yourself walking
python preprocess_custom_video.py --webcam --duration 10 --output my_data --subject me

# Or use an existing video
python preprocess_custom_video.py --video walk.mp4 --output my_data --subject friend

# Visualize the result
python -c "from preprocessing.visualization import visualize_sample; visualize_sample('my_data', 'me', 'walk-01', '090', 'viz')"
```

## Use Cases

- Collect your own gait dataset
- Test preprocessing on custom videos
- Record gait for specific analysis (e.g., before/after injury)
- Augment CASIA-B with your own data
- Quick prototyping without downloading large datasets

## Notes

- Pose detection may still fail if lighting/background is poor (this is expected)
- Focus on silhouettes and GEI for most reliable features
- The webcam mode works best in a well-lit room with clear background
