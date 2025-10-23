"""
Visualization Utilities for Gait Data

Tools for visualizing preprocessed gait data, including GEI, silhouettes, and pose.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import json


def visualize_gei(gei: np.ndarray, title: str = "Gait Energy Image",
                 save_path: Optional[str] = None):
    """
    Visualize a Gait Energy Image.
    
    Args:
        gei: GEI array (H, W)
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(6, 10))
    plt.imshow(gei, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_silhouette_sequence(silhouettes: np.ndarray, 
                                  max_frames: int = 20,
                                  title: str = "Silhouette Sequence",
                                  save_path: Optional[str] = None):
    """
    Visualize a sequence of silhouettes.
    
    Args:
        silhouettes: Array of silhouettes (T, H, W)
        max_frames: Maximum number of frames to display
        title: Plot title
        save_path: Optional path to save the figure
    """
    n_frames = min(silhouettes.shape[0], max_frames)
    cols = 5
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < n_frames:
            axes[row, col].imshow(silhouettes[i], cmap='gray')
            axes[row, col].set_title(f"Frame {i+1}")
        axes[row, col].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_pose_skeleton(landmarks: np.ndarray, frame_shape: Tuple[int, int] = (480, 640),
                           title: str = "Pose Skeleton", save_path: Optional[str] = None):
    """
    Visualize pose skeleton for a single frame.
    
    Args:
        landmarks: Pose landmarks (33, 3) where each row is [x, y, visibility]
        frame_shape: (height, width) for the visualization canvas
        title: Plot title
        save_path: Optional path to save the figure
    """
    # MediaPipe pose connections
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Upper body
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        # Torso
        (11, 23), (12, 24), (23, 24),
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    # Create canvas
    img = np.ones((frame_shape[0], frame_shape[1], 3), dtype=np.uint8) * 255
    
    # Convert normalized coordinates to pixel coordinates
    h, w = frame_shape
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] = landmarks[:, 0] * w
    landmarks_px[:, 1] = landmarks[:, 1] * h
    
    # Draw connections
    for start_idx, end_idx in connections:
        if landmarks[start_idx, 2] > 0.5 and landmarks[end_idx, 2] > 0.5:
            start = tuple(landmarks_px[start_idx, :2].astype(int))
            end = tuple(landmarks_px[end_idx, :2].astype(int))
            cv2.line(img, start, end, (0, 255, 0), 2)
    
    # Draw landmarks
    for i, landmark in enumerate(landmarks_px):
        if landmarks[i, 2] > 0.5:  # Check visibility
            center = tuple(landmark[:2].astype(int))
            cv2.circle(img, center, 3, (255, 0, 0), -1)
    
    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_pose_trajectory(pose_trajectories: np.ndarray,
                             joint_indices: List[int] = [23, 24, 25, 26],  # Hips and knees
                             joint_names: Optional[List[str]] = None,
                             title: str = "Joint Trajectories",
                             save_path: Optional[str] = None):
    """
    Visualize trajectories of specific joints over time.
    
    Args:
        pose_trajectories: Pose trajectory array (T, 33, 3)
        joint_indices: Indices of joints to visualize
        joint_names: Names of joints (optional)
        title: Plot title
        save_path: Optional path to save the figure
    """
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in joint_indices]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot X coordinates
    for idx, name in zip(joint_indices, joint_names):
        x_coords = pose_trajectories[:, idx, 0]
        axes[0].plot(x_coords, label=name, linewidth=2)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('X Coordinate (normalized)')
    axes[0].set_title('Horizontal Movement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Y coordinates
    for idx, name in zip(joint_indices, joint_names):
        y_coords = pose_trajectories[:, idx, 1]
        axes[1].plot(y_coords, label=name, linewidth=2)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Y Coordinate (normalized)')
    axes[1].set_title('Vertical Movement')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compare_geis(gei_list: List[np.ndarray], titles: List[str],
                save_path: Optional[str] = None):
    """
    Compare multiple GEIs side by side.
    
    Args:
        gei_list: List of GEI arrays
        titles: List of titles for each GEI
        save_path: Optional path to save the figure
    """
    n = len(gei_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 8))
    
    if n == 1:
        axes = [axes]
    
    for i, (gei, title) in enumerate(zip(gei_list, titles)):
        axes[i].imshow(gei, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_gait_cycle_animation(silhouettes: np.ndarray, output_path: str,
                               fps: int = 15):
    """
    Create an animated GIF of a gait cycle.
    
    Args:
        silhouettes: Array of silhouettes (T, H, W)
        output_path: Path to save the animation (should end in .gif)
        fps: Frames per second
    """
    import imageio
    
    frames = []
    for silhouette in silhouettes:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2RGB)
        frames.append(frame_rgb)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Animation saved to: {output_path}")


def visualize_sample(data_root: str, subject_id: str, sequence_id: str,
                    view_angle: str, save_dir: Optional[str] = None):
    """
    Visualize all preprocessing outputs for a single sample.
    
    Args:
        data_root: Root directory containing preprocessed data
        subject_id: Subject ID
        sequence_id: Sequence ID
        view_angle: View angle
        save_dir: Optional directory to save visualizations
    """
    data_root = Path(data_root)
    subject_dir = data_root / subject_id
    prefix = f"{subject_id}_{sequence_id}_{view_angle}"
    
    # Create output directory if saving
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None
    
    # Load and visualize GEI
    gei_path = subject_dir / f"{prefix}_gei.npy"
    if gei_path.exists():
        gei = np.load(gei_path)
        save_file = str(save_path / f"{prefix}_gei_viz.png") if save_path else None
        visualize_gei(gei, title=f"GEI - {subject_id} {sequence_id} {view_angle}",
                     save_path=save_file)
        print(f"✓ Visualized GEI")
    
    # Load and visualize silhouettes
    silhouettes_path = subject_dir / f"{prefix}_silhouettes.npy"
    if silhouettes_path.exists():
        silhouettes = np.load(silhouettes_path)
        save_file = str(save_path / f"{prefix}_silhouettes_viz.png") if save_path else None
        visualize_silhouette_sequence(silhouettes, 
                                     title=f"Silhouettes - {subject_id} {sequence_id} {view_angle}",
                                     save_path=save_file)
        print(f"✓ Visualized {len(silhouettes)} silhouette frames")
    
    # Load and visualize pose
    pose_path = subject_dir / f"{prefix}_pose.npy"
    if pose_path.exists():
        pose_trajectories = np.load(pose_path)
        
        # Visualize first frame skeleton
        save_file = str(save_path / f"{prefix}_pose_skeleton.png") if save_path else None
        visualize_pose_skeleton(pose_trajectories[0],
                              title=f"Pose - {subject_id} {sequence_id} {view_angle}",
                              save_path=save_file)
        
        # Visualize joint trajectories
        save_file = str(save_path / f"{prefix}_pose_trajectories.png") if save_path else None
        joint_names = ['Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']
        visualize_pose_trajectory(pose_trajectories, joint_names=joint_names,
                                 title=f"Joint Trajectories - {subject_id} {sequence_id} {view_angle}",
                                 save_path=save_file)
        print(f"✓ Visualized pose with {len(pose_trajectories)} frames")
    
    # Load and print metadata
    metadata_path = subject_dir / f"{prefix}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata:")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    print("Visualization utilities for gait data")
    print("\nExample usage:")
    print("""
    from visualization import visualize_sample
    
    # Visualize a specific sample
    visualize_sample(
        data_root='path/to/preprocessed/data',
        subject_id='001',
        sequence_id='nm-01',
        view_angle='090',
        save_dir='visualizations'
    )
    """)
