"""
Video Stabilization Script

Removes camera shake from videos using frame-to-frame motion estimation.
Uses optical flow to track features and compute homography transformations.

Usage:
    python stabilize_video.py input_video.mp4 --output stabilized.mp4
"""

import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def stabilize_video(input_path, output_path, smoothing_radius=30, border_size=50):
    """
    Stabilize a video by removing camera shake.
    
    Args:
        input_path: Path to input video
        output_path: Path to save stabilized video
        smoothing_radius: Number of frames to use for trajectory smoothing
        border_size: Pixels to crop from border (removes black edges)
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Store transforms between frames
    transforms = []
    
    print("Computing motion between frames...")
    pbar = tqdm(total=total_frames - 1)
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        
        if prev_pts is not None:
            # Calculate optical flow (track features)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            # Estimate transformation matrix
            if len(prev_pts) >= 4:  # Need at least 4 points for homography
                transform, _ = cv2.estimateAffinePartial2D(
                    prev_pts, curr_pts, method=cv2.RANSAC
                )
                
                if transform is not None:
                    # Extract translation
                    dx = transform[0, 2]
                    dy = transform[1, 2]
                    
                    # Extract rotation (angle)
                    da = np.arctan2(transform[1, 0], transform[0, 0])
                    
                    transforms.append([dx, dy, da])
                else:
                    # If estimation failed, assume no motion
                    transforms.append([0, 0, 0])
            else:
                transforms.append([0, 0, 0])
        else:
            transforms.append([0, 0, 0])
        
        prev_gray = curr_gray
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Convert to numpy array
    transforms = np.array(transforms)
    
    # Compute trajectory (cumulative sum of transforms)
    trajectory = np.cumsum(transforms, axis=0)
    
    print(f"Smoothing trajectory (radius={smoothing_radius})...")
    
    # Smooth trajectory using moving average
    def smooth(trajectory, radius):
        smoothed = np.copy(trajectory)
        for i in range(len(trajectory)):
            start = max(0, i - radius)
            end = min(len(trajectory), i + radius + 1)
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
        return smoothed
    
    smoothed_trajectory = smooth(trajectory, smoothing_radius)
    
    # Calculate difference between smoothed and original trajectory
    difference = smoothed_trajectory - trajectory
    
    # Apply smoothing to transforms
    transforms_smooth = transforms + difference
    
    # Write stabilized video
    print("Writing stabilized video...")
    
    cap = cv2.VideoCapture(input_path)
    
    # Output dimensions (cropped to remove black borders)
    out_width = width - 2 * border_size
    out_height = height - 2 * border_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (out_width, out_height)
    )
    
    ret, frame = cap.read()
    pbar = tqdm(total=len(transforms))
    
    for i, transform in enumerate(transforms_smooth):
        if not ret:
            break
        
        dx, dy, da = transform
        
        # Reconstruct transformation matrix
        transform_matrix = np.zeros((2, 3), np.float32)
        transform_matrix[0, 0] = np.cos(da)
        transform_matrix[0, 1] = -np.sin(da)
        transform_matrix[1, 0] = np.sin(da)
        transform_matrix[1, 1] = np.cos(da)
        transform_matrix[0, 2] = dx
        transform_matrix[1, 2] = dy
        
        # Apply transformation
        frame_stabilized = cv2.warpAffine(
            frame,
            transform_matrix,
            (width, height)
        )
        
        # Crop borders to remove black edges
        frame_cropped = frame_stabilized[
            border_size:height - border_size,
            border_size:width - border_size
        ]
        
        out.write(frame_cropped)
        
        ret, frame = cap.read()
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ Stabilized video saved to: {output_path}")
    print(f"  Output size: {out_width}x{out_height}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove camera shake from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic stabilization
  python stabilize_video.py shaky_video.mp4 --output stable.mp4
  
  # More aggressive smoothing
  python stabilize_video.py shaky_video.mp4 --output stable.mp4 --smoothing 50
  
  # Less border cropping
  python stabilize_video.py shaky_video.mp4 --output stable.mp4 --border 20
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output stabilized video (default: <input>_stabilized.mp4)"
    )
    
    parser.add_argument(
        "--smoothing",
        type=int,
        default=30,
        help="Smoothing radius in frames (default: 30)"
    )
    
    parser.add_argument(
        "--border",
        type=int,
        default=50,
        help="Border size to crop in pixels (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_stabilized{input_path.suffix}")
    
    # Stabilize
    stabilize_video(
        args.input,
        args.output,
        smoothing_radius=args.smoothing,
        border_size=args.border
    )


if __name__ == "__main__":
    main()
