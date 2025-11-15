"""
Unified Gait Preprocessing Pipeline

Complete end-to-end pipeline:
1. (Optional) Stabilize video
2. Segment person using YOLO + SAM2
3. Extract and track joints from RGB video
4. Generate GEI from silhouettes
5. Save in model-ready format

Usage:
    # Basic usage
    python process_video_pipeline.py video.mp4 --subject_id 001 --sequence_id walk1
    
    # With stabilization
    python process_video_pipeline.py shaky_video.mp4 --subject_id 001 --stabilize
    
    # Custom output directory
    python process_video_pipeline.py video.mp4 --subject_id 001 --output_dir my_data
"""

import argparse
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
from tqdm import tqdm

# Import our modules
from gait_preprocessor_v2 import GaitPreprocessor
from stabilize_video import stabilize_video


def segment_video_to_frames(video_path, output_dir, object_id=0):
    """
    Segment person from video and save silhouette frames.
    
    Returns:
        List of silhouette frame paths
    """
    from ultralytics import SAM, YOLO
    
    print("\n" + "="*80)
    print("STEP 2: Segmenting person with YOLO + SAM2")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading YOLO and SAM2 models...")
    yolo = YOLO("yolov8n.pt")
    sam_model = SAM("sam2_t.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get initial bounding box
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    yolo_result = yolo(first_frame, verbose=False)[0]
    obj_box = None
    
    for box in yolo_result.boxes:
        if int(box.cls) == object_id:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_box = [int(x1), int(y1), int(x2), int(y2)]
            break
    
    if obj_box is None:
        raise ValueError(f"No person detected in first frame!")
    
    cap.release()
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    silhouette_paths = []
    
    print(f"Processing {total_frames} frames...")
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Re-detect every 10 frames
        if frame_count % 10 == 0:
            yolo_result = yolo(frame, verbose=False)[0]
            for box in yolo_result.boxes:
                if int(box.cls) == object_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    obj_box = [int(x1), int(y1), int(x2), int(y2)]
                    break
        
        # Run SAM
        result = sam_model(frame, imgsz=512, bboxes=obj_box, verbose=False)[0]
        
        if result.masks is not None and len(result.masks.data) > 0:
            mask = result.masks.data[0].cpu().numpy()
            
            if len(mask.shape) == 2:
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                
                # Create silhouette
                silhouette = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                silhouette[mask > 0] = 255
                
                # Save
                save_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(save_path, silhouette)
                silhouette_paths.append(save_path)
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"✓ Saved {len(silhouette_paths)} silhouette frames")
    return silhouette_paths


def process_pipeline(video_path, subject_id, sequence_id, view_angle="unknown",
                     output_dir="preprocessed_data", stabilize=False,
                     temp_dir=None):
    """
    Complete preprocessing pipeline.
    
    Args:
        video_path: Path to input video
        subject_id: Subject identifier
        sequence_id: Sequence identifier  
        view_angle: Camera view angle
        output_dir: Output directory for processed data
        stabilize: Whether to stabilize video first
        temp_dir: Temporary directory for intermediate files
    """
    
    print("\n" + "="*80)
    print("GAIT PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Input video: {video_path}")
    print(f"Subject ID: {subject_id}")
    print(f"Sequence ID: {sequence_id}")
    print(f"Stabilization: {'ON' if stabilize else 'OFF'}")
    print("="*80)
    
    # Create temp directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Stabilize (optional)
    if stabilize:
        print("\n" + "="*80)
        print("STEP 1: Stabilizing video")
        print("="*80)
        
        stabilized_path = temp_dir / "stabilized.mp4"
        stabilize_video(video_path, str(stabilized_path), smoothing_radius=30, border_size=50)
        video_to_process = str(stabilized_path)
    else:
        print("\n" + "="*80)
        print("STEP 1: Skipping stabilization")
        print("="*80)
        video_to_process = video_path
    
    # STEP 2: Segment video to silhouettes
    silhouettes_dir = temp_dir / "silhouettes"
    segment_video_to_frames(video_to_process, str(silhouettes_dir))
    
    # STEP 3: Process with gait preprocessor
    print("\n" + "="*80)
    print("STEP 3: Extracting GEI and tracking joints")
    print("="*80)
    
    preprocessor = GaitPreprocessor(silhouette_size=(64, 128))
    
    # Load ORIGINAL RGB video for pose detection
    print("Loading RGB frames for pose detection...")
    rgb_frames = preprocessor.load_video(video_to_process)
    
    # Load silhouette frames
    print("Loading silhouette frames...")
    silhouette_frames = preprocessor.load_image_sequence(str(silhouettes_dir), pattern="*.png")
    
    print(f"Processing {len(rgb_frames)} RGB frames and {len(silhouette_frames)} silhouettes...")
    
    # Process silhouettes for GEI
    silhouettes, gei = preprocessor.silhouette_processor.process_sequence(silhouette_frames)
    
    # Process RGB for pose with tracking
    pose_trajectories = preprocessor.pose_processor.process_sequence_with_tracking(rgb_frames)
    
    # Create metadata
    metadata = {
        'n_frames': len(rgb_frames),
        'frame_height': rgb_frames[0].shape[0],
        'frame_width': rgb_frames[0].shape[1],
        'silhouette_size': (64, 128),
        'n_valid_poses': int(np.sum(pose_trajectories[:, 0, 2] > 0)),
        'tracking_enabled': True,
        'stabilized': stabilize
    }
    
    print(f"✓ GEI shape: {gei.shape}")
    print(f"✓ Silhouettes shape: {silhouettes.shape}")
    print(f"✓ Pose shape: {pose_trajectories.shape}")
    print(f"✓ Valid poses: {metadata['n_valid_poses']}/{len(rgb_frames)}")
    
    # STEP 4: Save results
    print("\n" + "="*80)
    print("STEP 4: Saving processed data")
    print("="*80)
    
    output_path = Path(output_dir) / subject_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_{sequence_id}_{view_angle}"
    
    # Save all outputs
    np.save(output_path / f"{prefix}_silhouettes.npy", silhouettes)
    np.save(output_path / f"{prefix}_gei.npy", gei)
    cv2.imwrite(str(output_path / f"{prefix}_gei.png"), gei)
    np.save(output_path / f"{prefix}_pose.npy", pose_trajectories)
    
    import json
    with open(output_path / f"{prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved to: {output_path}/")
    print(f"  - {prefix}_gei.npy (for models)")
    print(f"  - {prefix}_silhouettes.npy")
    print(f"  - {prefix}_pose.npy")
    print(f"  - {prefix}_metadata.json")
    
    preprocessor.close()
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nProcessed data ready for model training at:")
    print(f"  {output_path}/")


def main():
    parser = argparse.ArgumentParser(
        description="Complete gait preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_video_pipeline.py video.mp4 --subject_id 001 --sequence_id walk1
  
  # With stabilization
  python process_video_pipeline.py shaky.mp4 --subject_id 002 --sequence_id walk2 --stabilize
  
  # Custom output directory
  python process_video_pipeline.py video.mp4 --subject_id 003 --sequence_id run1 --output_dir my_data
        """
    )
    
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--subject_id", type=str, required=True, help="Subject identifier (e.g., '001')")
    parser.add_argument("--sequence_id", type=str, required=True, help="Sequence identifier (e.g., 'walk1')")
    parser.add_argument("--view_angle", type=str, default="unknown", help="Camera view angle (default: 'unknown')")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data", help="Output directory (default: 'preprocessed_data')")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize video before processing")
    parser.add_argument("--temp_dir", type=str, default=None, help="Temporary directory for intermediate files")
    
    args = parser.parse_args()
    
    # Run pipeline
    process_pipeline(
        video_path=args.video,
        subject_id=args.subject_id,
        sequence_id=args.sequence_id,
        view_angle=args.view_angle,
        output_dir=args.output_dir,
        stabilize=args.stabilize,
        temp_dir=args.temp_dir
    )


if __name__ == "__main__":
    main()
