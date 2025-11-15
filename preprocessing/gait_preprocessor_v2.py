"""
Gait Preprocessing Pipeline v2 - WITH JOINT TRACKING

Improvements over v1:
- Tracks joints across frames using optical flow
- Falls back to MediaPipe re-detection when tracking fails
- More robust to occlusions and missing detections

Authors: Ariel Ben Avi, Aditya Nangia, Eli Gross
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class GaitSequence:
    """Container for processed gait data"""
    subject_id: str
    sequence_id: str
    view_angle: str
    silhouettes: Optional[np.ndarray] = None  # Shape: (n_frames, height, width)
    gei: Optional[np.ndarray] = None  # Gait Energy Image
    pose_landmarks: Optional[List[np.ndarray]] = None  # List of (33, 3) arrays per frame
    pose_trajectories: Optional[np.ndarray] = None  # Shape: (n_frames, 33, 3)
    metadata: Optional[Dict] = None


class SilhouetteProcessor:
    """Handles silhouette extraction and GEI generation"""
    
    def __init__(self, target_size: Tuple[int, int] = (64, 128)):
        """
        Initialize silhouette processor.
        
        Args:
            target_size: (width, height) for normalized silhouettes
        """
        self.target_size = target_size
        
    def extract_silhouette(self, frame: np.ndarray, threshold: int = 127) -> np.ndarray:
        """Extract binary silhouette from a frame."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def normalize_silhouette(self, silhouette: np.ndarray) -> np.ndarray:
        """Normalize silhouette to standard size and position."""
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 10
        y1 = max(0, y - padding)
        y2 = min(silhouette.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(silhouette.shape[1], x + w + padding)
        
        cropped = silhouette[y1:y2, x1:x2]
        
        h_crop, w_crop = cropped.shape
        aspect_ratio = w_crop / h_crop
        target_aspect = self.target_size[0] / self.target_size[1]
        
        if aspect_ratio > target_aspect:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = self.target_size[1]
            new_w = int(new_h * aspect_ratio)
        
        resized = cv2.resize(cropped, (new_w, new_h))
        
        normalized = np.zeros(self.target_size[::-1], dtype=np.uint8)
        y_offset = (self.target_size[1] - new_h) // 2
        x_offset = (self.target_size[0] - new_w) // 2
        normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return normalized
    
    def generate_gei(self, silhouettes: np.ndarray) -> np.ndarray:
        """Generate Gait Energy Image (GEI) from silhouette sequence."""
        gei = np.mean(silhouettes, axis=0).astype(np.uint8)
        return gei
    
    def process_sequence(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a complete gait sequence to extract silhouettes and GEI."""
        silhouettes = []
        
        for frame in frames:
            silhouette = self.extract_silhouette(frame)
            normalized = self.normalize_silhouette(silhouette)
            silhouettes.append(normalized)
        
        silhouettes = np.array(silhouettes)
        gei = self.generate_gei(silhouettes)
        
        return silhouettes, gei


class PoseProcessorWithTracking:
    """Handles pose estimation with optical flow tracking"""
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """Initialize pose processor with MediaPipe."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def extract_pose_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return np.array(landmarks)
        
        return None
    
    def track_points_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                                  prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points from previous frame to current frame using optical flow.
        
        Returns:
            curr_points: Tracked points in current frame
            status: Array indicating which points were successfully tracked
        """
        # Convert landmarks to pixel coordinates for tracking
        h, w = prev_gray.shape
        prev_pts_px = prev_points[:, :2].copy()
        prev_pts_px[:, 0] *= w
        prev_pts_px[:, 1] *= h
        prev_pts_px = prev_pts_px.astype(np.float32)
        
        # Track using optical flow
        curr_pts_px, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts_px, None, **self.lk_params
        )
        
        if curr_pts_px is None:
            return None, np.zeros(len(prev_points), dtype=bool)
        
        # Convert back to normalized coordinates
        curr_points = prev_points.copy()
        curr_points[:, 0] = curr_pts_px[:, 0] / w
        curr_points[:, 1] = curr_pts_px[:, 1] / h
        
        # Mark points as valid if tracking succeeded and they're within frame bounds
        valid = (status.flatten() == 1) & \
                (curr_points[:, 0] >= 0) & (curr_points[:, 0] <= 1) & \
                (curr_points[:, 1] >= 0) & (curr_points[:, 1] <= 1)
        
        return curr_points, valid
    
    def process_sequence_with_tracking(self, frames: List[np.ndarray],
                                      redetect_interval: int = 30) -> np.ndarray:
        """
        Process gait sequence with joint tracking.
        
        Args:
            frames: List of RGB frames
            redetect_interval: Re-run MediaPipe every N frames to reset tracking
        
        Returns:
            Pose trajectories (n_frames, 33, 3)
        """
        trajectories = []
        prev_landmarks = None
        prev_gray = None
        frames_since_detection = 0
        
        print(f"Processing {len(frames)} frames with joint tracking...")
        
        for i, frame in enumerate(frames):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Decide whether to detect or track
            should_detect = (
                prev_landmarks is None or  # First frame
                frames_since_detection >= redetect_interval or  # Time to re-detect
                np.sum(prev_landmarks[:, 2] > 0.5) < 10  # Lost too many landmarks
            )
            
            if should_detect:
                # Run MediaPipe detection
                landmarks = self.extract_pose_mediapipe(frame)
                
                if landmarks is not None:
                    trajectories.append(landmarks)
                    prev_landmarks = landmarks
                    frames_since_detection = 0
                else:
                    # Detection failed - use zeros
                    trajectories.append(np.zeros((33, 3)))
                    prev_landmarks = None
            else:
                # Track from previous frame
                tracked_landmarks, valid = self.track_points_optical_flow(
                    prev_gray, curr_gray, prev_landmarks
                )
                
                if tracked_landmarks is not None:
                    # Update visibility based on tracking success
                    tracked_landmarks[:, 2] = valid.astype(float) * prev_landmarks[:, 2]
                    trajectories.append(tracked_landmarks)
                    prev_landmarks = tracked_landmarks
                else:
                    # Tracking failed - try detection
                    landmarks = self.extract_pose_mediapipe(frame)
                    if landmarks is not None:
                        trajectories.append(landmarks)
                        prev_landmarks = landmarks
                        frames_since_detection = 0
                    else:
                        trajectories.append(np.zeros((33, 3)))
                        prev_landmarks = None
                
                frames_since_detection += 1
            
            prev_gray = curr_gray
        
        return np.array(trajectories)
    
    def close(self):
        """Release MediaPipe resources"""
        self.pose.close()


class GaitPreprocessor:
    """Main preprocessing pipeline for gait data with tracking"""
    
    def __init__(self, silhouette_size: Tuple[int, int] = (64, 128)):
        """Initialize the complete preprocessing pipeline."""
        self.silhouette_processor = SilhouetteProcessor(target_size=silhouette_size)
        self.pose_processor = PoseProcessorWithTracking()
        
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def load_image_sequence(self, image_dir: str, pattern: str = "*.png") -> List[np.ndarray]:
        """Load sequence of images from directory."""
        image_paths = sorted(Path(image_dir).glob(pattern))
        frames = []
        
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def process(self, frames: List[np.ndarray], subject_id: str, 
                sequence_id: str, view_angle: str,
                use_tracking: bool = True) -> GaitSequence:
        """
        Process a complete gait sequence.
        
        Args:
            frames: List of video frames
            subject_id: Subject identifier
            sequence_id: Sequence identifier
            view_angle: Camera view angle
            use_tracking: If True, use optical flow tracking for joints
        
        Returns:
            GaitSequence object containing all processed data
        """
        # Process silhouettes
        silhouettes, gei = self.silhouette_processor.process_sequence(frames)
        
        # Process poses with tracking
        if use_tracking:
            pose_trajectories = self.pose_processor.process_sequence_with_tracking(frames)
        else:
            # Fallback to frame-by-frame detection
            pose_trajectories = []
            for frame in frames:
                pose = self.pose_processor.extract_pose_mediapipe(frame)
                if pose is not None:
                    pose_trajectories.append(pose)
                else:
                    pose_trajectories.append(np.zeros((33, 3)))
            pose_trajectories = np.array(pose_trajectories)
        
        # Create metadata
        metadata = {
            'n_frames': len(frames),
            'frame_height': frames[0].shape[0],
            'frame_width': frames[0].shape[1],
            'silhouette_size': self.silhouette_processor.target_size,
            'n_valid_poses': np.sum(pose_trajectories[:, 0, 2] > 0),
            'tracking_enabled': use_tracking
        }
        
        return GaitSequence(
            subject_id=subject_id,
            sequence_id=sequence_id,
            view_angle=view_angle,
            silhouettes=silhouettes,
            gei=gei,
            pose_landmarks=None,
            pose_trajectories=pose_trajectories,
            metadata=metadata
        )
    
    def save(self, gait_sequence: GaitSequence, output_dir: str):
        """Save processed gait sequence to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{gait_sequence.subject_id}_{gait_sequence.sequence_id}_{gait_sequence.view_angle}"
        
        # Save silhouettes
        np.save(output_path / f"{prefix}_silhouettes.npy", gait_sequence.silhouettes)
        
        # Save GEI
        np.save(output_path / f"{prefix}_gei.npy", gait_sequence.gei)
        cv2.imwrite(str(output_path / f"{prefix}_gei.png"), gait_sequence.gei)
        
        # Save pose trajectories
        np.save(output_path / f"{prefix}_pose.npy", gait_sequence.pose_trajectories)
        
        # Save metadata
        metadata = {
            'subject_id': gait_sequence.subject_id,
            'sequence_id': gait_sequence.sequence_id,
            'view_angle': gait_sequence.view_angle,
        }
        for key, value in gait_sequence.metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                metadata[key] = value.item()
            elif isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            else:
                metadata[key] = value
        
        with open(output_path / f"{prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def close(self):
        """Release resources"""
        self.pose_processor.close()


if __name__ == "__main__":
    print("Gait preprocessing module v2 with joint tracking loaded successfully!")
