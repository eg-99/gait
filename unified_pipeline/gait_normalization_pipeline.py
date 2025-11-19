#!/usr/bin/env python3
"""
Unified Gait Normalization Pipeline

Complete SMPL-based view normalization pipeline in a single file.
Transforms gait videos into canonical left→right walking silhouettes.

Usage:
    python gait_normalization_pipeline.py <video_path>
    
Example:
    python gait_normalization_pipeline.py ../Walking_MP4_Stock_videos/vid_1.mp4

Output Structure:
    intermediate_outputs/
        <video_name>/
            smpl_meshes.npz           - Stage 1: ROMP extraction
            canonical_meshes.npz      - Stage 2: Canonical transform
            projected_2d.npz          - Stage 3: 2D projection
    final_outputs/
        <video_name>_canonical_silhouette.mp4  - Final normalized video
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import romp
import time


# ============================================================================
# STAGE 1: SMPL MESH EXTRACTION
# ============================================================================

def extract_smpl_meshes(video_path, output_path):
    """
    Extract SMPL meshes from video using ROMP.
    
    Returns:
        dict with keys: 'verts' (T,6890,3), 'joints' (T,71,3), 'frame_ids'
    """
    print("="*80)
    print("STAGE 1: SMPL MESH EXTRACTION")
    print("="*80)
    print(f"Input video: {video_path}")
    print()
    
    # Initialize ROMP
    print("Initializing ROMP...")
    settings = romp.main.default_settings
    settings.mode = 'image'
    settings.calc_smpl = True
    settings.show = False
    settings.save_video = False
    
    romp_model = romp.ROMP(settings)
    print("✓ ROMP initialized")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {frame_count}")
    print(f"  FPS: {fps:.2f}")
    print()
    
    # Process frames
    all_verts = []
    all_joints = []
    frame_ids = []
    frame_idx = 0
    
    print("Extracting SMPL meshes...")
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        if frame_idx % 50 == 0:
            print(f"  Processing frame {frame_idx+1}/{frame_count}")
        
        try:
            outputs = romp_model(img)
            
            verts = outputs.get('verts', None)
            joints = None
            for key in ['joints_3d', 'j3d_smpl24', 'joints', 'j3d']:
                if key in outputs:
                    joints = outputs[key]
                    break
            
            if verts is None or joints is None:
                frame_idx += 1
                continue
            
            if len(verts) == 0:
                frame_idx += 1
                continue
            
            all_verts.append(verts[0])
            all_joints.append(joints[0])
            frame_ids.append(f"frame_{frame_idx:05d}")
            
        except Exception as e:
            print(f"  Warning: Error at frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    if not all_verts:
        raise RuntimeError("No valid frames processed!")
    
    verts_array = np.stack(all_verts, axis=0)
    joints_array = np.stack(all_joints, axis=0)
    
    print()
    print(f"✓ Extracted {len(all_verts)} frames")
    print(f"  Vertices shape: {verts_array.shape}")
    print(f"  Joints shape: {joints_array.shape}")
    
    # Save
    np.savez_compressed(
        output_path,
        verts=verts_array,
        joints=joints_array,
        frame_ids=np.array(frame_ids)
    )
    
    print(f"✓ Saved to: {output_path}")
    print()
    
    return {
        'verts': verts_array,
        'joints': joints_array,
        'frame_ids': frame_ids
    }


# ============================================================================
# STAGE 2: CANONICAL FRAME TRANSFORMATION
# ============================================================================

def transform_to_canonical(smpl_data, output_path):
    """
    Transform SMPL meshes to canonical coordinate frame.
    Walking direction → X-axis (left to right)
    Anatomical up → Y-axis
    
    Returns:
        dict with keys: 'verts' (T,6890,3), 'joints' (T,71,3) in canonical coords
    """
    print("="*80)
    print("STAGE 2: CANONICAL FRAME TRANSFORMATION")
    print("="*80)
    print()
    
    verts = smpl_data['verts']  # (T, 6890, 3)
    joints = smpl_data['joints']  # (T, J, 3)
    
    T = verts.shape[0]
    
    # Extract key joint trajectories with smoothing
    print("Extracting key trajectories...")
    pelvis = joints[:, 0, :]  # (T, 3)
    neck = joints[:, 12, :] if joints.shape[1] > 12 else joints[:, 0, :]
    left_ankle = joints[:, 7, :] if joints.shape[1] > 7 else pelvis
    right_ankle = joints[:, 8, :] if joints.shape[1] > 8 else pelvis
    
    # Smooth trajectories (5-frame moving average)
    window = 5
    if T >= window:
        kernel = np.ones(window) / window
        for i in range(3):
            pelvis[:, i] = np.convolve(pelvis[:, i], kernel, mode='same')
            neck[:, i] = np.convolve(neck[:, i], kernel, mode='same')
            left_ankle[:, i] = np.convolve(left_ankle[:, i], kernel, mode='same')
            right_ankle[:, i] = np.convolve(right_ankle[:, i], kernel, mode='same')
    
    feet = (left_ankle + right_ankle) / 2
    
    print("✓ Key trajectories extracted and smoothed")
    print()
    
    # Compute walking direction (horizontal displacement)
    print("Computing walking direction...")
    displacement = pelvis[-1] - pelvis[0]
    displacement[1] = 0  # Zero out vertical component
    
    if np.linalg.norm(displacement) < 1e-6:
        print("  Warning: Stationary person, using default direction [1,0,0]")
        walking_dir = np.array([1.0, 0.0, 0.0])
    else:
        walking_dir = displacement / np.linalg.norm(displacement)
    
    print(f"  Walking direction: [{walking_dir[0]:.3f}, {walking_dir[1]:.3f}, {walking_dir[2]:.3f}]")
    print()
    
    # Compute anatomical up direction
    print("Computing anatomical up direction...")
    spine = neck.mean(axis=0) - pelvis.mean(axis=0)
    up_dir = spine - np.dot(spine, walking_dir) * walking_dir
    up_dir = up_dir / np.linalg.norm(up_dir)
    
    print(f"  Up direction: [{up_dir[0]:.3f}, {up_dir[1]:.3f}, {up_dir[2]:.3f}]")
    print()
    
    # Construct canonical frame basis
    X_canon = walking_dir
    Y_canon = up_dir
    Z_canon = np.cross(X_canon, Y_canon)
    Z_canon = Z_canon / np.linalg.norm(Z_canon)
    
    # Ensure orthogonality
    Y_canon = np.cross(Z_canon, X_canon)
    Y_canon = Y_canon / np.linalg.norm(Y_canon)
    
    R_world_to_canon = np.stack([X_canon, Y_canon, Z_canon], axis=0)
    
    print("Transforming to canonical coordinates...")
    pelvis_center = pelvis.mean(axis=0)
    
    verts_canon = (verts - pelvis_center) @ R_world_to_canon.T
    joints_canon = (joints - pelvis_center) @ R_world_to_canon.T
    
    print("✓ Transformed to canonical frame")
    print()
    
    # Check orientation (head-up, feet-down)
    print("Checking orientation...")
    neck_canon = joints_canon[:, 12, :] if joints_canon.shape[1] > 12 else joints_canon[:, 0, :]
    feet_canon = (joints_canon[:, 7, :] + joints_canon[:, 8, :]) / 2 if joints_canon.shape[1] > 8 else joints_canon[:, 0, :]
    
    neck_y = neck_canon[:, 1].mean()
    feet_y = feet_canon[:, 1].mean()
    
    if neck_y < feet_y:
        print("  ⚠ Upside down detected! Flipping Y-axis...")
        verts_canon[:, :, 1] *= -1
        joints_canon[:, :, 1] *= -1
        neck_y, feet_y = feet_y, neck_y
    
    print(f"  Neck Y: {neck_y:.3f}")
    print(f"  Feet Y: {feet_y:.3f}")
    print("  ✓ Orientation correct (head up, feet down)")
    print()
    
    # Save
    np.savez_compressed(
        output_path,
        verts=verts_canon,
        joints=joints_canon,
        frame_ids=smpl_data['frame_ids']
    )
    
    print(f"✓ Saved to: {output_path}")
    print()
    
    return {
        'verts': verts_canon,
        'joints': joints_canon,
        'frame_ids': smpl_data['frame_ids']
    }


# ============================================================================
# STAGE 3: 2D PROJECTION
# ============================================================================

def project_to_2d(canonical_data, output_path, output_size=(512, 512), padding_factor=0.1):
    """
    Project 3D canonical coordinates to 2D pixel coordinates.
    Uses orthographic projection with uniform scaling.
    
    Returns:
        dict with keys: 'verts_2d' (T,6890,2), 'joints_2d' (T,J,2), 'projection_params'
    """
    print("="*80)
    print("STAGE 3: 2D PROJECTION")
    print("="*80)
    print()
    
    verts_canon = canonical_data['verts']  # (T, 6890, 3)
    joints_canon = canonical_data['joints']  # (T, J, 3)
    
    H, W = output_size
    
    # Compute bounding box over all frames (X' and Y' only)
    print("Computing bounding box...")
    all_xy = verts_canon[..., :2].reshape(-1, 2)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    
    # Add padding
    size_xy = max_xy - min_xy
    padding = size_xy * padding_factor
    min_xy = min_xy - padding
    max_xy = max_xy + padding
    
    print(f"  X' range: [{min_xy[0]:.3f}, {max_xy[0]:.3f}]")
    print(f"  Y' range: [{min_xy[1]:.3f}, {max_xy[1]:.3f}]")
    print()
    
    # Compute uniform scale
    print("Computing projection scale...")
    scale_x = (W - 1) / (max_xy[0] - min_xy[0])
    scale_y = (H - 1) / (max_xy[1] - min_xy[1])
    scale = min(scale_x, scale_y)  # Uniform scale preserves aspect ratio
    
    print(f"  Scale: {scale:.2f} pixels/unit")
    print()
    
    # Project vertices
    print("Projecting to pixel coordinates...")
    verts_2d = np.zeros((verts_canon.shape[0], verts_canon.shape[1], 2), dtype=np.float32)
    joints_2d = np.zeros((joints_canon.shape[0], joints_canon.shape[1], 2), dtype=np.float32)
    
    for t in range(verts_canon.shape[0]):
        # Extract X', Y' coordinates
        x_prime = verts_canon[t, :, 0]
        y_prime = verts_canon[t, :, 1]
        
        # Map to pixel coordinates with Y-flip
        u = (x_prime - min_xy[0]) * scale
        v = (max_xy[1] - y_prime) * scale  # Flip Y: up in 3D = up in image
        
        verts_2d[t, :, 0] = u
        verts_2d[t, :, 1] = v
        
        # Project joints
        x_prime_j = joints_canon[t, :, 0]
        y_prime_j = joints_canon[t, :, 1]
        
        u_j = (x_prime_j - min_xy[0]) * scale
        v_j = (max_xy[1] - y_prime_j) * scale
        
        joints_2d[t, :, 0] = u_j
        joints_2d[t, :, 1] = v_j
    
    print(f"✓ Projected {verts_2d.shape[0]} frames")
    print(f"  u range: [{verts_2d[..., 0].min():.1f}, {verts_2d[..., 0].max():.1f}]")
    print(f"  v range: [{verts_2d[..., 1].min():.1f}, {verts_2d[..., 1].max():.1f}]")
    print()
    
    # Save
    projection_params = {
        'min_xy': min_xy,
        'max_xy': max_xy,
        'scale': scale,
        'output_size': output_size,
        'padding_factor': padding_factor
    }
    
    np.savez_compressed(
        output_path,
        verts_2d=verts_2d,
        joints_2d=joints_2d,
        projection_params=projection_params,
        verts_canon=canonical_data['verts'],
        joints_canon=canonical_data['joints']
    )
    
    print(f"✓ Saved to: {output_path}")
    print()
    
    return {
        'verts_2d': verts_2d,
        'joints_2d': joints_2d,
        'projection_params': projection_params
    }


# ============================================================================
# STAGE 4: SILHOUETTE RENDERING
# ============================================================================

def load_smpl_faces():
    """Load SMPL mesh topology (faces)."""
    smpl_model_paths = [
        Path.home() / '.romp' / 'SMPL_NEUTRAL.pth',
        Path('smpl_model_files/smpl/models/SMPL_NEUTRAL.pkl'),
        Path('smpl_model_data/SMPL_NEUTRAL.pkl'),
    ]
    
    for model_path in smpl_model_paths:
        if model_path.exists():
            try:
                if model_path.suffix == '.pth':
                    import torch
                    model_data = torch.load(model_path, weights_only=False)
                    if 'f' in model_data:
                        faces = model_data['f']
                        if isinstance(faces, torch.Tensor):
                            faces = faces.cpu().numpy()
                        return faces.astype(np.int32)
                elif model_path.suffix == '.pkl':
                    import pickle
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f, encoding='latin1')
                    if 'f' in model_data:
                        return np.array(model_data['f'], dtype=np.int32)
            except Exception:
                continue
    
    raise RuntimeError("Could not load SMPL faces. Ensure SMPL model exists at ~/.romp/SMPL_NEUTRAL.pth")


def render_silhouette_video(projection_data, output_path, fps=30):
    """
    Render silhouette video from 2D projected vertices.
    White silhouette on black background.
    
    Returns:
        Path to output video
    """
    print("="*80)
    print("STAGE 4: SILHOUETTE RENDERING")
    print("="*80)
    print()
    
    verts_2d = projection_data['verts_2d']  # (T, 6890, 2)
    output_size = projection_data['projection_params']['output_size']
    
    T = verts_2d.shape[0]
    H, W = output_size
    
    print(f"Loading SMPL mesh topology...")
    faces = load_smpl_faces()
    print(f"✓ Loaded {len(faces)} triangular faces")
    print()
    
    print(f"Rendering video...")
    print(f"  Resolution: {W}×{H}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {T}")
    print()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H), isColor=False)
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    
    # Render each frame
    for t in range(T):
        if t % 50 == 0 or t == T - 1:
            print(f"  Frame {t+1}/{T}")
        
        # Create black image
        img = np.zeros((H, W), dtype=np.uint8)
        
        # Fill each triangle
        pts = verts_2d[t]
        for tri in faces:
            tri_pts = pts[tri].astype(np.int32)
            cv2.fillConvexPoly(img, tri_pts, 255)
        
        out.write(img)
    
    out.release()
    
    print()
    print(f"✓ Video rendered: {output_path}")
    print(f"  Duration: {T/fps:.2f} seconds")
    print()
    
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_video(video_path):
    """
    Complete pipeline: video → SMPL → canonical → 2D → silhouette
    
    Args:
        video_path: Path to input video file
    
    Returns:
        Path to final silhouette video
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    video_name = video_path.stem
    
    print("="*80)
    print("UNIFIED GAIT NORMALIZATION PIPELINE")
    print("="*80)
    print(f"Video: {video_name}")
    print(f"Path: {video_path}")
    print()
    
    start_time = time.time()
    
    # Create output directories
    base_dir = Path(__file__).parent
    intermediate_dir = base_dir / "intermediate_outputs" / video_name
    final_dir = base_dir / "final_outputs"
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: SMPL Extraction
    smpl_output = intermediate_dir / "smpl_meshes.npz"
    if not smpl_output.exists():
        smpl_data = extract_smpl_meshes(str(video_path), str(smpl_output))
    else:
        print(f"⏭️  Loading existing SMPL data: {smpl_output}")
        data = np.load(smpl_output)
        smpl_data = {'verts': data['verts'], 'joints': data['joints'], 'frame_ids': data['frame_ids']}
        print()
    
    # Stage 2: Canonical Transformation
    canonical_output = intermediate_dir / "canonical_meshes.npz"
    if not canonical_output.exists():
        canonical_data = transform_to_canonical(smpl_data, str(canonical_output))
    else:
        print(f"⏭️  Loading existing canonical data: {canonical_output}")
        data = np.load(canonical_output)
        canonical_data = {
            'verts': data['verts'],
            'joints': data['joints'],
            'frame_ids': smpl_data.get('frame_ids', [])
        }
        print()
    
    # Stage 3: 2D Projection
    projection_output = intermediate_dir / "projected_2d.npz"
    if not projection_output.exists():
        projection_data = project_to_2d(canonical_data, str(projection_output))
    else:
        print(f"⏭️  Loading existing projection data: {projection_output}")
        data = np.load(projection_output, allow_pickle=True)
        projection_data = {
            'verts_2d': data['verts_2d'],
            'joints_2d': data['joints_2d'],
            'projection_params': data['projection_params'].item()
        }
        print()
    
    # Stage 4: Silhouette Rendering
    video_output = final_dir / f"{video_name}_canonical_silhouette.mp4"
    if not video_output.exists():
        render_silhouette_video(projection_data, str(video_output))
    else:
        print(f"⏭️  Video already exists: {video_output}")
        print()
    
    elapsed = time.time() - start_time
    
    print("="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print()
    print("Output files:")
    print(f"  Intermediate outputs: {intermediate_dir}")
    print(f"    - smpl_meshes.npz       (Stage 1: ROMP extraction)")
    print(f"    - canonical_meshes.npz  (Stage 2: Canonical transform)")
    print(f"    - projected_2d.npz      (Stage 3: 2D projection)")
    print()
    print(f"  Final output: {video_output}")
    print(f"    - White silhouette on black background")
    print(f"    - Walking left→right")
    print(f"    - Stable orientation (head up, feet down)")
    print("="*80)
    
    return video_output


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing video path")
        print("Usage: python gait_normalization_pipeline.py <video_path>")
        print("\nExample:")
        print("  python gait_normalization_pipeline.py ../Walking_MP4_Stock_videos/vid_1.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        output_video = process_video(video_path)
        print(f"\n✅ SUCCESS: {output_video}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
