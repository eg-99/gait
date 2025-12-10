#!/usr/bin/env python3
"""
Gait Abnormality Detection Pipeline

Complete end-to-end pipeline that:
1. Takes an input video of a person walking
2. Extracts silhouettes using YOLO + SAM2
3. Groups frames into gait cycles (~30 frames each)
4. Generates Gait Energy Images (GEI) for each cycle
5. Runs inference through trained VAE model
6. Outputs binary prediction: NORMAL vs ABNORMAL

Usage:
    python gait_abnormality_pipeline.py <video_path> [options]
    
Example:
    python gait_abnormality_pipeline.py walking_video.mp4 --output results/
    python gait_abnormality_pipeline.py walking_video.mp4 --model exp1_vae --visualize
    
Authors: Based on work by Ariel Ben Avi, Aditya Nangia, Eli Gross
"""

import os
import sys
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Deep learning
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Supported COCO classes for YOLO detection (extend as needed)
SUPPORTED_OBJECT_CLASSES = {
    'person': 0,
    'dog': 16,
}


def get_object_id_from_name(object_name: str) -> Tuple[int, str]:
    """Resolve a COCO object name to (class_id, normalized_name)."""
    name = object_name.lower().strip()
    if name not in SUPPORTED_OBJECT_CLASSES:
        supported = ", ".join(sorted(SUPPORTED_OBJECT_CLASSES))
        raise ValueError(
            f"Unsupported object '{object_name}'. Supported classes: {supported}"
        )
    return SUPPORTED_OBJECT_CLASSES[name], name


# =============================================================================
# MODEL DEFINITIONS (copied from experiments_final to be self-contained)
# =============================================================================

class GEI_VAE(nn.Module):
    """Variational Autoencoder for GEI images."""
    
    def __init__(self, latent_dim=128):
        super(GEI_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(256)
        
        self.flatten_size = 256 * 8 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def encode(self, x):
        x = torch.relu(self.enc_bn1(self.enc_conv1(x)))
        x = torch.relu(self.enc_bn2(self.enc_conv2(x)))
        x = torch.relu(self.enc_bn3(self.enc_conv3(x)))
        x = torch.relu(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(x.size(0), 256, 8, 4)
        x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = torch.relu(self.dec_bn3(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def get_embedding(self, x):
        mu, _ = self.encode(x)
        return mu


class ContrastiveVAE(nn.Module):
    """Combined VAE + Contrastive Learning model."""
    
    def __init__(self, latent_dim=128, projection_dim=128):
        super(ContrastiveVAE, self).__init__()
        self.vae = GEI_VAE(latent_dim=latent_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, projection_dim)
        )
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
    
    def encode(self, x):
        return self.vae.encode(x)
    
    def decode(self, z):
        return self.vae.decode(z)
    
    def reparameterize(self, mu, log_var):
        return self.vae.reparameterize(mu, log_var)
    
    def get_embedding(self, x):
        mu, _ = self.encode(x)
        return mu
    
    def get_projection(self, x):
        mu, log_var = self.encode(x)
        projection = self.projection_head(mu)
        return projection, mu, log_var
    
    def forward(self, x, return_projection=False):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        if return_projection:
            projection = self.projection_head(mu)
            return reconstruction, mu, log_var, projection
        return reconstruction, mu, log_var


def create_vae(latent_dim=128):
    """Factory function to create VAE."""
    return GEI_VAE(latent_dim=latent_dim)


def create_contrastive_vae(latent_dim=128, projection_dim=128):
    """Factory function to create Contrastive VAE."""
    return ContrastiveVAE(latent_dim=latent_dim, projection_dim=projection_dim)


# =============================================================================
# SILHOUETTE EXTRACTION (from preprocessing/segment_video.py)
# =============================================================================

def get_bbox_from_mask(mask, padding=10):
    """Get bounding box from mask with optional padding."""
    if len(mask.shape) != 2:
        return None
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [
        max(0, int(x_min) - padding),
        max(0, int(y_min) - padding),
        int(x_max) + padding,
        int(y_max) + padding
    ]


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def find_best_matching_box(current_box, detections, iou_threshold=0.3):
    """Find the detection box that best matches the current tracked box."""
    if not detections:
        return None
    best_box = None
    best_iou = iou_threshold
    for det_box in detections:
        iou = calculate_iou(current_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_box = det_box
    return best_box


def extract_silhouettes_from_video(
    video_path: str,
    output_dir: str,
    output_width: int = 64,
    output_height: int = 128,
    max_frames: int = 150,
    yolo_update_interval: int = 10,
    sam_imgsz: int = 512,
    target_class: str = "person"
) -> List[str]:
    """
    Extract silhouettes from video using YOLO + SAM2.
    
    Returns:
        List of paths to saved silhouette images
    """
    try:
        from ultralytics import SAM, YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")
    
    os.makedirs(output_dir, exist_ok=True)

    target_class_id, target_class_name = get_object_id_from_name(target_class)
    print(f"Tracking object class '{target_class_name}' (COCO id {target_class_id})")
    
    # Look for models in preprocessing/ folder first, download there if missing
    script_dir = Path(__file__).parent
    preprocessing_dir = script_dir / "preprocessing"
    preprocessing_dir.mkdir(exist_ok=True)
    
    yolo_path = preprocessing_dir / "yolov8n.pt"
    sam2_path = preprocessing_dir / "sam2_t.pt"
    
    print("Loading YOLO and SAM2 models...")
    if yolo_path.exists():
        print(f"  Using cached YOLO: {yolo_path}")
        yolo = YOLO(str(yolo_path))
    else:
        print(f"  Downloading YOLO to: {yolo_path}")
        yolo = YOLO("yolov8n.pt")
        # Move downloaded model to preprocessing folder
        if Path("yolov8n.pt").exists():
            shutil.move("yolov8n.pt", yolo_path)
    
    if sam2_path.exists():
        print(f"  Using cached SAM2: {sam2_path}")
        sam_model = SAM(str(sam2_path))
    else:
        print(f"  Downloading SAM2 to: {sam2_path}")
        sam_model = SAM("sam2_t.pt")
        # Move downloaded model to preprocessing folder
        if Path("sam2_t.pt").exists():
            shutil.move("sam2_t.pt", sam2_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    
    # Detect person in first frame
    yolo_result = yolo(first_frame, verbose=False)[0]
    obj_box = None
    for box in yolo_result.boxes:
        if int(box.cls) == target_class_id:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_box = [int(x1), int(y1), int(x2), int(y2)]
            break
    
    if obj_box is None:
        cap.release()
        raise ValueError(f"No {target_class_name} detected in first frame!")
    
    silhouette_paths = []
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting silhouettes (sample rate: every {sample_rate} frames)...")
    pbar = tqdm(total=min(max_frames, total_frames // sample_rate))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if sample_rate > 1 and frame_count % sample_rate != 0:
            continue
        
        # Re-detect with YOLO periodically
        if frame_count % yolo_update_interval == 0:
            yolo_result = yolo(frame, verbose=False)[0]
            detections = []
            for box in yolo_result.boxes:
                if int(box.cls) == target_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
            if detections:
                best_match = find_best_matching_box(obj_box, detections, iou_threshold=0.3)
                if best_match:
                    obj_box = best_match
        
        # Run SAM with bounding box
        result = sam_model(frame, imgsz=sam_imgsz, bboxes=obj_box, verbose=False)[0]
        
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

                # Crop tightly to the detected subject to avoid large empty borders
                bbox = get_bbox_from_mask(mask, padding=5)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x2 = max(x1 + 1, x2)
                    y2 = max(y1 + 1, y2)
                    silhouette = silhouette[y1:y2, x1:x2]

                # Resize to CASIA-B format without stretching (letterbox)
                sil_h, sil_w = silhouette.shape
                if sil_h == 0 or sil_w == 0:
                    continue

                target_aspect = output_width / output_height
                sil_aspect = sil_w / sil_h

                if sil_aspect > target_aspect:
                    new_w = output_width
                    new_h = max(1, int(round(new_w / sil_aspect)))
                else:
                    new_h = output_height
                    new_w = max(1, int(round(new_h * sil_aspect)))

                resized = cv2.resize(
                    silhouette,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                )

                padded = np.zeros((output_height, output_width), dtype=np.uint8)
                y_offset = (output_height - new_h) // 2
                x_offset = (output_width - new_w) // 2
                padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                silhouette = padded
                
                # Save
                output_path = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
                cv2.imwrite(output_path, silhouette)
                silhouette_paths.append(output_path)
                saved_count += 1
                pbar.update(1)
                
                # Update bounding box
                new_box = get_bbox_from_mask(mask)
                if new_box:
                    new_box[2] = min(frame.shape[1], new_box[2])
                    new_box[3] = min(frame.shape[0], new_box[3])
                    obj_box = new_box
        print(f"Saved {saved_count} silhouettes to {output_dir}")
        if saved_count >= max_frames:
            break
    
    pbar.close()
    cap.release()
    
    print(f"✅ Extracted {saved_count} silhouettes")
    return silhouette_paths


# =============================================================================
# GEI GENERATION
# =============================================================================

def preprocess_silhouette(silhouette: np.ndarray) -> np.ndarray:
    """Preprocess silhouette frame."""
    _, binary = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == largest_component, 255, 0).astype(np.uint8)
    
    return binary


def normalize_silhouette(silhouette: np.ndarray, target_size: Tuple[int, int] = (64, 128)) -> np.ndarray:
    """Normalize silhouette to target size with centering."""
    coords = np.column_stack(np.where(silhouette > 0))
    if len(coords) == 0:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop to bounding box
    cropped = silhouette[y_min:y_max+1, x_min:x_max+1]
    
    # Resize maintaining aspect ratio
    h_crop, w_crop = cropped.shape
    aspect = w_crop / h_crop
    target_aspect = target_size[0] / target_size[1]
    
    if aspect > target_aspect:
        new_w = target_size[0]
        new_h = max(1, int(new_w / aspect))
    else:
        new_h = target_size[1]
        new_w = max(1, int(new_h * aspect))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Center in frame
    normalized = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return normalized


def compute_gei(silhouette_paths: List[str], normalize: bool = True) -> np.ndarray:
    """
    Compute Gait Energy Image from silhouette frames.
    
    Args:
        silhouette_paths: List of paths to silhouette images
        normalize: Whether to normalize silhouette size/position
    
    Returns:
        GEI array (H, W) with values 0-255
    """
    if not silhouette_paths:
        raise ValueError("No silhouette paths provided")
    
    def _load_silhouette(path: str) -> Optional[np.ndarray]:
        """Load a silhouette frame, forcing grayscale if needed."""
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            return None
        
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 1:
                return frame[:, :, 0]
            if channels == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        
        raise ValueError(f"Unsupported silhouette shape: {frame.shape} from {path}")
    
    # Read first frame to get dimensions
    first_frame = _load_silhouette(silhouette_paths[0])
    if first_frame is None:
        raise ValueError(f"Unable to read silhouette: {silhouette_paths[0]}")
    
    H, W = first_frame.shape[:2]
    
    # Accumulator
    gei = np.zeros((H, W), dtype=np.float64)
    valid_frames = 0
    
    for frame_path in silhouette_paths:
        frame = _load_silhouette(frame_path)
        if frame is None:
            print(f"Skipping unreadable silhouette: {frame_path}")
            continue
        
        binary = preprocess_silhouette(frame)
        
        if normalize:
            binary = normalize_silhouette(binary, target_size=(W, H))
        
        gei += binary.astype(np.float64) / 255.0
        valid_frames += 1
    
    if valid_frames == 0:
        raise ValueError("No valid frames to compute GEI")
    
    gei = gei / valid_frames
    gei_uint8 = (gei * 255).astype(np.uint8)
    
    return gei_uint8


def group_frames_into_cycles(
    silhouette_paths: List[str],
    frames_per_cycle: int = 30
) -> List[List[str]]:
    """
    Group silhouette frames into gait cycles.
    
    Args:
        silhouette_paths: List of paths to silhouette images
        frames_per_cycle: Number of frames per gait cycle (~30 for typical walking)
    
    Returns:
        List of frame path groups, each representing one gait cycle
    """
    n_frames = len(silhouette_paths)
    if n_frames < frames_per_cycle:
        # If we don't have enough frames, use all of them as one cycle
        return [silhouette_paths]
    
    groups = []
    for i in range(0, n_frames - frames_per_cycle + 1, frames_per_cycle // 2):
        # 50% overlap between cycles
        group = silhouette_paths[i:i + frames_per_cycle]
        if len(group) >= frames_per_cycle // 2:
            groups.append(group)
    
    return groups if groups else [silhouette_paths]


# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

class GaitAbnormalityDetector:
    """Main class for gait abnormality detection."""
    
    # Available models and their characteristics
    AVAILABLE_MODELS = {
        'exp1_vae': {
            'type': 'vae',
            'desc': 'Zero-shot transfer (CASIA-B trained) - Best for binary detection',
            'binary_acc': 0.969
        },
        'exp1_contrastive': {
            'type': 'contrastive',
            'desc': 'Zero-shot transfer with contrastive learning',
            'binary_acc': 0.953
        },
        'exp3_vae': {
            'type': 'vae',
            'desc': 'Trained from scratch on pathology data',
            'binary_acc': 0.957
        },
        'exp3_contrastive': {
            'type': 'contrastive',
            'desc': 'Best multi-class accuracy (89.5%)',
            'binary_acc': 0.942
        }
    }
    
    def __init__(self, checkpoint_dir: str, model_name: str = 'exp1_vae'):
        """
        Initialize the detector.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            model_name: Which model to use (exp1_vae, exp1_contrastive, exp3_vae, exp3_contrastive)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.model = None
        self.device = DEVICE
        
        # Classification thresholds (learned from training data)
        # These are based on reconstruction error - higher error = more likely abnormal
        self.reconstruction_threshold = 300.0  # Tuned from experiments
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified model."""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {self.model_name}. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        model_info = self.AVAILABLE_MODELS[self.model_name]
        
        # Determine checkpoint path
        checkpoint_mapping = {
            'exp1_vae': 'exp1_vae_casia.pth',
            'exp1_contrastive': 'exp1_contrastive_casia.pth',
            'exp3_vae': 'exp3_vae_pathology.pth',
            'exp3_contrastive': 'exp3_contrastive_pathology.pth'
        }
        
        checkpoint_path = self.checkpoint_dir / checkpoint_mapping[self.model_name]
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model: {self.model_name}")
        print(f"  Description: {model_info['desc']}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        # Create model
        if model_info['type'] == 'vae':
            self.model = create_vae(latent_dim=128)
        else:
            self.model = create_contrastive_vae(latent_dim=128, projection_dim=128)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✅ Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    def preprocess_gei(self, gei: np.ndarray) -> torch.Tensor:
        """
        Preprocess GEI for model input.
        
        Args:
            gei: GEI array (H, W) or (128, 64)
        
        Returns:
            Tensor of shape (1, 1, 128, 64)
        """
        # Ensure correct shape (128 height, 64 width)
        if gei.shape != (128, 64):
            gei = cv2.resize(gei, (64, 128), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        gei_float = gei.astype(np.float32) / 255.0
        
        # Convert to tensor (1, 1, H, W)
        tensor = torch.from_numpy(gei_float).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def extract_embedding(self, gei_tensor: torch.Tensor) -> np.ndarray:
        """Extract embedding from GEI tensor."""
        with torch.no_grad():
            if isinstance(self.model, ContrastiveVAE):
                embedding = self.model.get_embedding(gei_tensor)
            else:
                mu, _ = self.model.encode(gei_tensor)
                embedding = mu
        return embedding.cpu().numpy()
    
    def compute_reconstruction_error(self, gei_tensor: torch.Tensor) -> float:
        """Compute reconstruction error for anomaly detection."""
        with torch.no_grad():
            if isinstance(self.model, ContrastiveVAE):
                recon, _, _ = self.model(gei_tensor, return_projection=False)
            else:
                recon, _, _ = self.model(gei_tensor)
            
            mse = nn.functional.mse_loss(recon, gei_tensor, reduction='sum')
            return mse.item()
    
    def predict_single(self, gei: np.ndarray) -> Dict:
        """
        Predict abnormality for a single GEI.
        
        Args:
            gei: GEI array (H, W)
        
        Returns:
            Dict with prediction results
        """
        gei_tensor = self.preprocess_gei(gei)
        
        # Get reconstruction error
        recon_error = self.compute_reconstruction_error(gei_tensor)
        
        # Get embedding
        embedding = self.extract_embedding(gei_tensor)
        
        # Binary prediction based on reconstruction error
        # Higher reconstruction error = more likely abnormal
        # (The model was trained on normal gaits, so abnormal gaits have higher error)
        is_abnormal = recon_error > self.reconstruction_threshold
        
        # Confidence based on distance from threshold
        distance_from_threshold = abs(recon_error - self.reconstruction_threshold)
        confidence = min(0.99, 0.5 + (distance_from_threshold / self.reconstruction_threshold) * 0.5)
        
        return {
            'prediction': 'ABNORMAL' if is_abnormal else 'NORMAL',
            'is_abnormal': is_abnormal,
            'reconstruction_error': recon_error,
            'confidence': confidence,
            'embedding': embedding[0]  # (128,)
        }
    
    def predict_video(
        self,
        video_path: str,
        temp_dir: Optional[str] = None,
        frames_per_cycle: int = 30,
        visualize: bool = False,
        target_class: str = "person"
    ) -> Dict:
        """
        Complete pipeline: video -> prediction.
        
        Args:
            video_path: Path to input video
            temp_dir: Directory for temporary files (auto-cleaned if None)
            frames_per_cycle: Frames per gait cycle
            visualize: Whether to save visualization
            target_class: Which COCO class to segment (e.g., 'person', 'dog')
        
        Returns:
            Dict with overall prediction and per-cycle results
        """
        # Create temp directory
        cleanup_temp = temp_dir is None
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix='gait_pipeline_')
        
        silhouette_dir = os.path.join(temp_dir, 'silhouettes')
        gei_dir = os.path.join(temp_dir, 'geis')
        os.makedirs(silhouette_dir, exist_ok=True)
        os.makedirs(gei_dir, exist_ok=True)
        
        try:
            # Step 1: Extract silhouettes
            print("\n" + "="*60)
            print("STEP 1: Extracting Silhouettes")
            print("="*60)
            silhouette_paths = extract_silhouettes_from_video(
                video_path, 
                silhouette_dir,
                output_width=64,
                output_height=128,
                max_frames=150,
                target_class=target_class
            )
            
            if len(silhouette_paths) < 10:
                raise ValueError(f"Too few silhouettes extracted: {len(silhouette_paths)}")
            
            # Step 2: Group into cycles
            print("\n" + "="*60)
            print("STEP 2: Grouping into Gait Cycles")
            print("="*60)
            frame_groups = group_frames_into_cycles(silhouette_paths, frames_per_cycle)
            print(f"Created {len(frame_groups)} gait cycles")
            
            # Step 3: Generate GEIs
            print("\n" + "="*60)
            print("STEP 3: Generating GEIs")
            print("="*60)
            geis = []
            for i, group in enumerate(frame_groups):
                gei = compute_gei(group, normalize=True)
                geis.append(gei)
                if visualize:
                    cv2.imwrite(os.path.join(gei_dir, f'gei_cycle_{i}.png'), gei)
            print(f"Generated {len(geis)} GEIs")
            
            # Step 4: Run predictions
            print("\n" + "="*60)
            print("STEP 4: Running Inference")
            print("="*60)
            cycle_results = []
            for i, gei in enumerate(geis):
                result = self.predict_single(gei)
                result['cycle_idx'] = i
                cycle_results.append(result)
                print(f"  Cycle {i}: {result['prediction']} "
                      f"(recon_error={result['reconstruction_error']:.2f}, "
                      f"confidence={result['confidence']:.2%})")
            
            # Step 5: Aggregate results
            print("\n" + "="*60)
            print("STEP 5: Final Prediction")
            print("="*60)
            
            # Majority voting with confidence weighting
            abnormal_votes = sum(r['is_abnormal'] for r in cycle_results)
            normal_votes = len(cycle_results) - abnormal_votes
            
            # Final prediction
            final_is_abnormal = abnormal_votes > normal_votes
            
            # Best cycle is the one with highest confidence that matches final prediction
            best_cycle = max(
                [r for r in cycle_results if r['is_abnormal'] == final_is_abnormal],
                key=lambda x: x['confidence']
            )
            
            # Overall confidence
            avg_confidence = np.mean([r['confidence'] for r in cycle_results])
            vote_confidence = max(abnormal_votes, normal_votes) / len(cycle_results)
            overall_confidence = (avg_confidence + vote_confidence) / 2
            
            final_result = {
                'video_path': video_path,
                'prediction': 'ABNORMAL' if final_is_abnormal else 'NORMAL',
                'is_abnormal': final_is_abnormal,
                'confidence': overall_confidence,
                'abnormal_votes': abnormal_votes,
                'normal_votes': normal_votes,
                'total_cycles': len(cycle_results),
                'best_cycle': best_cycle,
                'cycle_results': cycle_results,
                'model_used': self.model_name
            }
            
            print(f"\n{'='*60}")
            print(f"FINAL RESULT: {final_result['prediction']}")
            print(f"Confidence: {final_result['confidence']:.2%}")
            print(f"Votes: {abnormal_votes} abnormal, {normal_votes} normal")
            print(f"Best cycle: #{best_cycle['cycle_idx']} "
                  f"(error={best_cycle['reconstruction_error']:.2f})")
            print(f"{'='*60}")
            
            return final_result
            
        finally:
            # Cleanup temp directory
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gait Abnormality Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses exp1_vae model by default)
  python gait_abnormality_pipeline.py walking_video.mp4
  
  # Specify model and output directory
  python gait_abnormality_pipeline.py walking_video.mp4 --model exp3_contrastive --output results/
  
  # With visualization
  python gait_abnormality_pipeline.py walking_video.mp4 --visualize --output results/

Available models:
  - exp1_vae: Zero-shot transfer (CASIA-B) - Best for binary detection (96.9% acc)
  - exp1_contrastive: Zero-shot with contrastive learning (95.3% acc)
  - exp3_vae: Trained on pathology data (95.7% acc)
  - exp3_contrastive: Best multi-class accuracy (94.2% binary acc)
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments_final/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--model', type=str, default='exp1_vae',
                       choices=['exp1_vae', 'exp1_contrastive', 'exp3_vae', 'exp3_contrastive'],
                       help='Which model to use for detection')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--frames-per-cycle', type=int, default=30,
                       help='Number of frames per gait cycle (default: 30)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--object-class', type=str, default='person',
                        choices=sorted(SUPPORTED_OBJECT_CLASSES.keys()),
                        help='COCO object class to segment (default: person)')
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Setup output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        temp_dir = args.output
    else:
        temp_dir = None
    
    try:
        # Initialize detector
        detector = GaitAbnormalityDetector(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model
        )
        
        # Run prediction
        result = detector.predict_video(
            args.video,
            temp_dir=temp_dir if args.visualize else None,
            frames_per_cycle=args.frames_per_cycle,
            visualize=args.visualize,
            target_class=args.object_class
        )
        
        # Output results
        if args.json:
            # Convert numpy arrays to lists for JSON serialization
            result_json = {k: v for k, v in result.items() if k != 'best_cycle'}
            result_json['best_cycle'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in result['best_cycle'].items()
            }
            result_json['cycle_results'] = [
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
                for r in result['cycle_results']
            ]
            print(json.dumps(result_json, indent=2))
        
        # Save results if output directory specified
        if args.output:
            result_path = os.path.join(args.output, 'prediction_result.json')
            result_json = {k: v for k, v in result.items() if k != 'best_cycle'}
            result_json['best_cycle'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in result['best_cycle'].items()
            }
            result_json['cycle_results'] = [
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
                for r in result['cycle_results']
            ]
            with open(result_path, 'w') as f:
                json.dump(result_json, f, indent=2)
            print(f"\n✅ Results saved to: {result_path}")
        
        # Exit code based on prediction
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()