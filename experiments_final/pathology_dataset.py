"""
Image-based dataset loader for pathology GEI images.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict


class PathologyGEIDataset(Dataset):
    """
    Dataset for loading GEI images from pathology dataset.
    
    Expected structure:
        data_root/
            condition1/
                subject_01/
                    image_001.jpg
                    image_002.jpg
            condition2/
                subject_01/
                    ...
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 target_size: Tuple[int, int] = (128, 64),
                 transform=None, random_seed: int = 42):
        """
        Args:
            data_root: Root directory with condition folders
            split: 'train', 'val', or 'test'
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            target_size: (height, width) for resizing images
            transform: Optional transform
            random_seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # Collect all image paths and labels
        all_samples = []
        self.condition_to_idx = {}
        self.idx_to_condition = {}
        self.subject_to_idx = {}
        self.idx_to_subject = {}
        
        condition_idx = 0
        subject_idx = 0
        
        # Iterate through condition directories
        for cond_dir in sorted(self.data_root.iterdir()):
            if not cond_dir.is_dir():
                continue
            
            condition_name = cond_dir.name
            self.condition_to_idx[condition_name] = condition_idx
            self.idx_to_condition[condition_idx] = condition_name
            
            # Iterate through subject directories
            for subj_dir in sorted(cond_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                
                subject_id = f"{condition_name}_{subj_dir.name}"
                
                if subject_id not in self.subject_to_idx:
                    self.subject_to_idx[subject_id] = subject_idx
                    self.idx_to_subject[subject_idx] = subject_id
                    subject_idx += 1
                
                # Collect all image files
                for img_path in sorted(subj_dir.glob('*.jpg')) + sorted(subj_dir.glob('*.png')):
                    all_samples.append({
                        'path': img_path,
                        'condition': condition_name,
                        'condition_idx': condition_idx,
                        'subject_id': subject_id,
                        'subject_idx': self.subject_to_idx[subject_id]
                    })
            
            condition_idx += 1
        
        if not all_samples:
            raise ValueError(f"No images found in {data_root}")
        
        # Split data by subject (to avoid data leakage)
        unique_subjects = list(self.subject_to_idx.keys())
        
        # Set random seed for reproducibility
        import random
        random.seed(random_seed)
        random.shuffle(unique_subjects)
        
        # Calculate split indices
        n_subjects = len(unique_subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        if split == 'train':
            selected_subjects = set(unique_subjects[:n_train])
        elif split == 'val':
            selected_subjects = set(unique_subjects[n_train:n_train+n_val])
        elif split == 'test':
            selected_subjects = set(unique_subjects[n_train+n_val:])
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Filter samples for this split
        self.samples = [s for s in all_samples if s['subject_id'] in selected_subjects]
        
        print(f"{split.upper()} split: {len(self.samples)} images from {len(selected_subjects)} subjects")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns:
            Tuple of (image_tensor, subject_label, metadata)
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('L')  # Grayscale
        
        # Resize to target size
        img = img.resize((self.target_size[1], self.target_size[0]), Image.LANCZOS)
        
        # Convert to numpy array [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor (1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Apply transform if provided
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        # Label is subject index (for contrastive learning)
        label = sample['subject_idx']
        
        # Metadata
        metadata = {
            'condition': sample['condition'],
            'condition_idx': sample['condition_idx'],
            'subject_id': sample['subject_id'],
            'path': str(sample['path'])
        }
        
        return img_tensor, label, metadata
    
    def get_num_classes(self) -> int:
        """Get number of unique subjects"""
        return len(self.subject_to_idx)
    
    def get_num_conditions(self) -> int:
        """Get number of unique conditions"""
        return len(self.condition_to_idx)
