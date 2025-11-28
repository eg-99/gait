"""
Gait Preprocessing Pipeline for CASIA-B Dataset
Handles both silhouette-based and pose-based preprocessing approaches.

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
    
    def __init__(self, target_size: Tuple[int, int] = (64, 128), use_rembg: bool = False):
        """
        Initialize silhouette processor.
        
        Args:
            target_size: (width, height) for normalized silhouettes
            use_rembg: Use AI-based background removal (better for complex backgrounds)
        """
        self.target_size = target_size
        self.use_rembg = use_rembg
        
        if use_rembg:
            try:
                from rembg import remove
                self.rembg_remove = remove
            except ImportError:
                print("Warning: rembg not installed. Falling back to threshold-based extraction.")
                print("Install with: pip install rembg")
                self.use_rembg = False
        
    def extract_silhouette(self, frame: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Extract binary silhouette from a frame.
        
        Args:
            frame: Input image (BGR or grayscale)
            threshold: Binary threshold value (used if not using rembg)
            
        Returns:
            Binary silhouette mask
        """
        if self.use_rembg:
            # Use AI-based background removal
            # rembg expects RGB
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Remove background (returns RGBA)
            result = self.rembg_remove(rgb_frame)
            
            # Extract alpha channel as mask
            if len(result.shape) == 3 and result.shape[2] == 4:
                alpha = result[:, :, 3]
            else:
                # If no alpha, convert to grayscale
                alpha = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            
            # Threshold to binary
            _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            
        else:
            # Use traditional threshold-based extraction
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def normalize_silhouette(self, silhouette: np.ndarray) -> np.ndarray:
        """
        Normalize silhouette to standard size and position.
        
        Args:
            silhouette: Binary silhouette image
            
        Returns:
            Normalized silhouette
        """
        # Find contours to get bounding box
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return empty silhouette if no contours found
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop to bounding box with padding
        padding = 10
        y1 = max(0, y - padding)
        y2 = min(silhouette.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(silhouette.shape[1], x + w + padding)
        
        cropped = silhouette[y1:y2, x1:x2]
        
        # Resize to target size while maintaining aspect ratio
        h_crop, w_crop = cropped.shape
        aspect_ratio = w_crop / h_crop
        target_aspect = self.target_size[0] / self.target_size[1]
        
        if aspect_ratio > target_aspect:
            # Width is limiting factor
            new_w = self.target_size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            # Height is limiting factor
            new_h = self.target_size[1]
            new_w = int(new_h * aspect_ratio)
        
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Center in target size canvas
        normalized = np.zeros(self.target_size[::-1], dtype=np.uint8)
        y_offset = (self.target_size[1] - new_h) // 2
        x_offset = (self.target_size[0] - new_w) // 2
        normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return normalized
    
    def generate_gei(self, silhouettes: np.ndarray) -> np.ndarray:
        """
        Generate Gait Energy Image (GEI) from silhouette sequence.
        
        Args:
            silhouettes: Array of silhouettes (n_frames, height, width)
            
        Returns:
            GEI image (averaged silhouettes over time)
        """
        # Average all silhouettes across time dimension
        gei = np.mean(silhouettes, axis=0).astype(np.uint8)
        return gei
    
    def process_sequence(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a complete gait sequence to extract silhouettes and GEI.
        
        Args:
            frames: List of video frames
            
        Returns:
            Tuple of (silhouettes, gei)
        """
        silhouettes = []
        
        for frame in frames:
            # Extract and normalize silhouette
            silhouette = self.extract_silhouette(frame)
            normalized = self.normalize_silhouette(silhouette)
            silhouettes.append(normalized)
        
        silhouettes = np.array(silhouettes)
        gei = self.generate_gei(silhouettes)
        
        return silhouettes, gei


class PoseProcessor:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose processor with MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Array of shape (33, 3) containing [x, y, visibility] for each landmark,
            or None if pose not detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract landmarks as numpy array
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return np.array(landmarks)
        
        return None
    
    def process_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Process a complete gait sequence to extract pose trajectories.
        
        Args:
            frames: List of video frames
            
        Returns:
            Array of shape (n_frames, 33, 3) containing pose landmarks over time
        """
        trajectories = []
        
        for frame in frames:
            pose = self.extract_pose(frame)
            if pose is not None:
                trajectories.append(pose)
            else:
                # Fill missing frames with zeros
                trajectories.append(np.zeros((33, 3)))
        
        return np.array(trajectories)
    
    def compute_joint_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute key joint angles from pose landmarks.
        
        Args:
            landmarks: Pose landmarks (33, 3)
            
        Returns:
            Dictionary of joint angles in degrees
        """
        def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            """Calculate angle at point b formed by points a-b-c"""
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        # MediaPipe landmark indices
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        RIGHT_HIP = 24
        RIGHT_KNEE = 26
        RIGHT_ANKLE = 28
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        
        angles = {}
        
        # Left knee angle
        if landmarks[LEFT_HIP, 2] > 0.5 and landmarks[LEFT_KNEE, 2] > 0.5 and landmarks[LEFT_ANKLE, 2] > 0.5:
            angles['left_knee'] = calculate_angle(
                landmarks[LEFT_HIP, :2],
                landmarks[LEFT_KNEE, :2],
                landmarks[LEFT_ANKLE, :2]
            )
        
        # Right knee angle
        if landmarks[RIGHT_HIP, 2] > 0.5 and landmarks[RIGHT_KNEE, 2] > 0.5 and landmarks[RIGHT_ANKLE, 2] > 0.5:
            angles['right_knee'] = calculate_angle(
                landmarks[RIGHT_HIP, :2],
                landmarks[RIGHT_KNEE, :2],
                landmarks[RIGHT_ANKLE, :2]
            )
        
        # Hip angle (left)
        if landmarks[LEFT_SHOULDER, 2] > 0.5 and landmarks[LEFT_HIP, 2] > 0.5 and landmarks[LEFT_KNEE, 2] > 0.5:
            angles['left_hip'] = calculate_angle(
                landmarks[LEFT_SHOULDER, :2],
                landmarks[LEFT_HIP, :2],
                landmarks[LEFT_KNEE, :2]
            )
        
        # Hip angle (right)
        if landmarks[RIGHT_SHOULDER, 2] > 0.5 and landmarks[RIGHT_HIP, 2] > 0.5 and landmarks[RIGHT_KNEE, 2] > 0.5:
            angles['right_hip'] = calculate_angle(
                landmarks[RIGHT_SHOULDER, :2],
                landmarks[RIGHT_HIP, :2],
                landmarks[RIGHT_KNEE, :2]
            )
        
        return angles
    
    def close(self):
        """Release MediaPipe resources"""
        self.pose.close()


class GaitPreprocessor:
    """Main preprocessing pipeline for gait data"""
    
    def __init__(self, silhouette_size: Tuple[int, int] = (64, 128), use_rembg: bool = False):
        """
        Initialize the complete preprocessing pipeline.
        
        Args:
            silhouette_size: Target size for normalized silhouettes
            use_rembg: Use AI-based background removal for better silhouettes
        """
        self.silhouette_processor = SilhouetteProcessor(target_size=silhouette_size, use_rembg=use_rembg)
        self.pose_processor = PoseProcessor()
        
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """
        Load video frames from file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames (BGR)
        """
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
        """
        Load sequence of images from directory.
        
        Args:
            image_dir: Directory containing images
            pattern: Glob pattern for image files
            
        Returns:
            List of frames sorted by filename
        """
        image_paths = sorted(Path(image_dir).glob(pattern))
        frames = []
        
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def process(self, frames: List[np.ndarray], subject_id: str, 
                sequence_id: str, view_angle: str) -> GaitSequence:
        """
        Process a complete gait sequence with both silhouette and pose methods.
        
        Args:
            frames: List of video frames
            subject_id: Subject identifier
            sequence_id: Sequence identifier (e.g., walking condition)
            view_angle: Camera view angle
            
        Returns:
            GaitSequence object containing all processed data
        """
        # Process silhouettes
        silhouettes, gei = self.silhouette_processor.process_sequence(frames)
        
        # Process poses
        pose_trajectories = self.pose_processor.process_sequence(frames)
        
        # Create metadata
        metadata = {
            'n_frames': len(frames),
            'frame_height': frames[0].shape[0],
            'frame_width': frames[0].shape[1],
            'silhouette_size': self.silhouette_processor.target_size,
            'n_valid_poses': np.sum(pose_trajectories[:, 0, 2] > 0)
        }
        
        return GaitSequence(
            subject_id=subject_id,
            sequence_id=sequence_id,
            view_angle=view_angle,
            silhouettes=silhouettes,
            gei=gei,
            pose_landmarks=None,  # Raw landmarks not stored to save space
            pose_trajectories=pose_trajectories,
            metadata=metadata
        )
    
    def save(self, gait_sequence: GaitSequence, output_dir: str):
        """
        Save processed gait sequence to disk.
        
        Args:
            gait_sequence: Processed gait sequence
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename prefix
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
        # Convert numpy types to native Python types for JSON serialization
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
    # Example usage
    preprocessor = GaitPreprocessor()
    
    # Example: Process a video file
    # frames = preprocessor.load_video("path/to/video.mp4")
    # gait_data = preprocessor.process(frames, "subject_001", "normal", "90deg")
    # preprocessor.save(gait_data, "output/preprocessed")
    
    print("Gait preprocessing module loaded successfully!")
    preprocessor.close()
