#!/usr/bin/env python3
"""
Video to GEI Pipeline: Complete silhouette extraction and GEI computation

Extracts person silhouettes from video using YOLO+SAM2, then computes
Gait Energy Image (GEI) using proper normalization and preprocessing.

Usage:
    python video_to_gei.py <video_path>
    
Example:
    python video_to_gei.py ../Walking_MP4_Stock_videos/vid_1.mp4
"""

import sys
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np


def extract_silhouettes(video_path, output_dir, width=512):
    """
    Extract silhouette frames using segment_video.py
    
    Args:
        video_path: Path to input video
        output_dir: Directory for silhouette frames
        width: Output width for frames
    
    Returns:
        Path to output directory
    """
    print("="*80)
    print("STEP 1: EXTRACT SILHOUETTE FRAMES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Check if frames already exist
    if output_dir.exists() and len(list(output_dir.glob("frame_*.png"))) > 0:
        frame_count = len(list(output_dir.glob("frame_*.png")))
        print(f"✓ Frames already exist: {output_dir}")
        print(f"  Found {frame_count} frames")
        print()
        return output_dir
    
    print(f"Extracting silhouettes from: {video_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Run segment_video.py from same directory
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable,
        str(script_dir / "segment_video.py"),
        str(video_path),
        "--output-dir", str(output_dir),
        "--object-name", "person",
        "--width", str(width)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running segment_video.py:")
        print(result.stderr)
        raise RuntimeError("Silhouette extraction failed")
    
    print(result.stdout)
    print()
    
    return output_dir


def preprocess_silhouette(silhouette):
    """
    Preprocess silhouette frame for GEI computation
    
    Best practices from literature:
    1. Binarize to ensure {0, 1} values
    2. Remove noise with morphological operations
    3. Ensure single connected component (largest)
    
    Args:
        silhouette: Input silhouette (H, W) grayscale
        
    Returns:
        Preprocessed binary silhouette (H, W) as {0, 255}
    """
    # Binarize
    _, binary = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening: remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fill small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Keep only largest connected component (remove outliers)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:  # 0 is background
        # Find largest component (excluding background at index 0)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == largest_component, 255, 0).astype(np.uint8)
    
    return binary


def compute_gei(frames_dir, output_path, normalize=True):
    """
    Compute Gait Energy Image (GEI) from silhouette frames
    
    GEI computation following Han & Bhanu (2006) and later works:
    1. Load all silhouette frames from one gait cycle
    2. Preprocess each frame (binarize, denoise, clean)
    3. Normalize silhouettes spatially (optional but recommended)
    4. Temporal averaging: GEI(x,y) = (1/N) * Σ B_t(x,y)
    5. Output as grayscale image [0, 255]
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_path: Path to save GEI image
        normalize: Whether to normalize silhouette size/position
    
    Returns:
        GEI array (H, W) with values 0-255
    """
    print("="*80)
    print("STEP 2: COMPUTE GAIT ENERGY IMAGE (GEI)")
    print("="*80)
    
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    
    print(f"Found {len(frame_files)} frames")
    print(f"Computing GEI with preprocessing...")
    print()
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_files[0]}")
    
    H, W = first_frame.shape
    print(f"Frame dimensions: {W}x{H}")
    print()
    
    # Initialize accumulator (float64 for precision)
    gei = np.zeros((H, W), dtype=np.float64)
    valid_frames = 0
    
    # Process and accumulate all frames
    for i, frame_file in enumerate(frame_files):
        if (i + 1) % 50 == 0 or i == 0 or i == len(frame_files) - 1:
            print(f"  Processing frame {i+1}/{len(frame_files)}")
        
        frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"  Warning: Could not read {frame_file}, skipping")
            continue
        
        # Preprocess silhouette
        binary = preprocess_silhouette(frame)
        
        # Optional: Normalize silhouette position/size
        if normalize:
            # Find bounding box of silhouette
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Crop to bounding box
                cropped = binary[y_min:y_max+1, x_min:x_max+1]
                
                # Resize to standard size (maintain aspect ratio)
                h_crop, w_crop = cropped.shape
                aspect = w_crop / h_crop
                
                if aspect > W / H:
                    # Width-limited
                    new_w = W
                    new_h = int(W / aspect)
                else:
                    # Height-limited
                    new_h = H
                    new_w = int(H * aspect)
                
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # Center in frame
                normalized = np.zeros((H, W), dtype=np.uint8)
                y_offset = (H - new_h) // 2
                x_offset = (W - new_w) // 2
                normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                binary = normalized
        
        # Accumulate (convert to [0, 1] range for averaging)
        gei += binary.astype(np.float64) / 255.0
        valid_frames += 1
    
    if valid_frames == 0:
        raise RuntimeError("No valid frames to compute GEI")
    
    # Temporal averaging
    gei = gei / valid_frames
    
    # Convert back to [0, 255] range
    gei_uint8 = (gei * 255).astype(np.uint8)
    
    print()
    print(f"GEI statistics:")
    print(f"  Valid frames: {valid_frames}/{len(frame_files)}")
    print(f"  Min intensity: {gei_uint8.min()}")
    print(f"  Max intensity: {gei_uint8.max()}")
    print(f"  Mean intensity: {gei_uint8.mean():.2f}")
    print(f"  Non-zero pixels: {np.count_nonzero(gei_uint8)} ({100*np.count_nonzero(gei_uint8)/(H*W):.1f}%)")
    print()
    
    # Save GEI
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), gei_uint8)
    
    print(f"✓ Saved GEI to: {output_path}")
    print()
    
    return gei_uint8


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing video path")
        print("Usage: python video_to_gei.py <video_path>")
        print("\nExample:")
        print("  python video_to_gei.py ../Walking_MP4_Stock_videos/vid_1.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    video_name = video_path.stem
    
    print("="*80)
    print("VIDEO TO GEI PIPELINE")
    print("="*80)
    print(f"Video: {video_name}")
    print(f"Path: {video_path}")
    print()
    
    script_dir = Path(__file__).parent
    
    # Step 1: Extract silhouette frames
    frames_dir = script_dir / f"silhouettes_{video_name}"
    frames_dir = extract_silhouettes(video_path, frames_dir, width=512)
    
    # Step 2: Compute GEI
    gei_output = script_dir / "gei_outputs" / f"gei_{video_name}.png"
    gei = compute_gei(frames_dir, gei_output, normalize=True)
    
    print("="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"Outputs:")
    print(f"  Silhouette frames: {frames_dir}/")
    print(f"  GEI image: {gei_output}")
    print("="*80)


if __name__ == "__main__":
    main()
