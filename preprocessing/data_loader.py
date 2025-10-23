"""
Data Loaders for Preprocessed Gait Data

Utilities for loading preprocessed data during model training.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from torch.utils.data import Dataset
import torch


class GaitDataset(Dataset):
    """PyTorch Dataset for preprocessed gait data"""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 data_type: str = 'gei', transform=None):
        """
        Initialize gait dataset.
        
        Args:
            data_root: Root directory containing preprocessed data
            split: Which split to load ('train', 'val', or 'test')
            data_type: Type of data to load ('gei', 'silhouettes', or 'pose')
            transform: Optional transform to apply to data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.data_type = data_type
        self.transform = transform
        
        # Load data splits
        splits_path = self.data_root / 'data_splits.json'
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            self.subject_ids = splits[split]
        else:
            # If no splits file, use all subjects
            subject_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
            self.subject_ids = sorted([d.name for d in subject_dirs])
        
        # Build file list
        self.samples = self._build_sample_list()
        
        # Create subject to index mapping
        unique_subjects = sorted(set(s['subject_id'] for s in self.samples))
        self.subject_to_idx = {subj: idx for idx, subj in enumerate(unique_subjects)}
        
    def _build_sample_list(self) -> List[Dict]:
        """Build list of all samples in the split"""
        samples = []
        
        for subject_id in self.subject_ids:
            subject_dir = self.data_root / subject_id
            if not subject_dir.exists():
                continue
            
            # Find all preprocessed files for this subject
            if self.data_type == 'gei':
                pattern = f"*_gei.npy"
            elif self.data_type == 'silhouettes':
                pattern = f"*_silhouettes.npy"
            elif self.data_type == 'pose':
                pattern = f"*_pose.npy"
            else:
                raise ValueError(f"Unknown data type: {self.data_type}")
            
            for data_path in subject_dir.glob(pattern):
                # Parse filename to extract metadata
                filename = data_path.stem
                parts = filename.replace(f'_{self.data_type}', '').split('_')
                
                if len(parts) >= 3:
                    samples.append({
                        'path': data_path,
                        'subject_id': parts[0],
                        'sequence_id': parts[1],
                        'view_angle': parts[2]
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (data, label, metadata)
        """
        sample = self.samples[idx]
        
        # Load data
        data = np.load(sample['path'])
        
        # Convert to tensor
        if self.data_type == 'gei':
            # GEI: (H, W) -> (1, H, W)
            data = torch.from_numpy(data).float().unsqueeze(0) / 255.0
        elif self.data_type == 'silhouettes':
            # Silhouettes: (T, H, W) -> (T, 1, H, W)
            data = torch.from_numpy(data).float().unsqueeze(1) / 255.0
        elif self.data_type == 'pose':
            # Pose: (T, 33, 3)
            data = torch.from_numpy(data).float()
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)
        
        # Get label (subject index)
        label = self.subject_to_idx[sample['subject_id']]
        
        # Metadata
        metadata = {
            'subject_id': sample['subject_id'],
            'sequence_id': sample['sequence_id'],
            'view_angle': sample['view_angle']
        }
        
        return data, label, metadata
    
    def get_num_classes(self) -> int:
        """Get number of unique subjects (classes)"""
        return len(self.subject_to_idx)


class GaitDataLoader:
    """Utility class for creating data loaders"""
    
    @staticmethod
    def create_loaders(data_root: str, data_type: str = 'gei',
                      batch_size: int = 32, num_workers: int = 4,
                      transform=None) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            data_root: Root directory containing preprocessed data
            data_type: Type of data to load ('gei', 'silhouettes', or 'pose')
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            transform: Optional transform to apply to data
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            dataset = GaitDataset(
                data_root=data_root,
                split=split,
                data_type=data_type,
                transform=transform
            )
            
            loaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
        
        return loaders


class SequenceDataset(Dataset):
    """Dataset for sequence data (silhouettes or pose trajectories)"""
    
    def __init__(self, data_root: str, split: str = 'train',
                 data_type: str = 'silhouettes', sequence_length: Optional[int] = None,
                 transform=None):
        """
        Initialize sequence dataset.
        
        Args:
            data_root: Root directory containing preprocessed data
            split: Which split to load ('train', 'val', or 'test')
            data_type: Type of sequence data ('silhouettes' or 'pose')
            sequence_length: If provided, sequences will be truncated/padded to this length
            transform: Optional transform to apply to data
        """
        self.sequence_length = sequence_length
        self.base_dataset = GaitDataset(data_root, split, data_type, transform)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        data, label, metadata = self.base_dataset[idx]
        
        # Handle sequence length
        if self.sequence_length is not None:
            current_length = data.shape[0]
            
            if current_length > self.sequence_length:
                # Truncate
                data = data[:self.sequence_length]
            elif current_length < self.sequence_length:
                # Pad with zeros
                pad_length = self.sequence_length - current_length
                if len(data.shape) == 4:  # Silhouettes (T, 1, H, W)
                    padding = torch.zeros(pad_length, *data.shape[1:])
                else:  # Pose (T, 33, 3)
                    padding = torch.zeros(pad_length, *data.shape[1:])
                data = torch.cat([data, padding], dim=0)
        
        return data, label, metadata


def load_gei_sample(data_root: str, subject_id: str, 
                   sequence_id: str, view_angle: str) -> Optional[np.ndarray]:
    """
    Load a specific GEI sample.
    
    Args:
        data_root: Root directory containing preprocessed data
        subject_id: Subject ID
        sequence_id: Sequence ID
        view_angle: View angle
        
    Returns:
        GEI array or None if not found
    """
    data_path = Path(data_root) / subject_id / f"{subject_id}_{sequence_id}_{view_angle}_gei.npy"
    
    if data_path.exists():
        return np.load(data_path)
    return None


def load_pose_sample(data_root: str, subject_id: str,
                    sequence_id: str, view_angle: str) -> Optional[np.ndarray]:
    """
    Load a specific pose trajectory sample.
    
    Args:
        data_root: Root directory containing preprocessed data
        subject_id: Subject ID
        sequence_id: Sequence ID
        view_angle: View angle
        
    Returns:
        Pose trajectory array (T, 33, 3) or None if not found
    """
    data_path = Path(data_root) / subject_id / f"{subject_id}_{sequence_id}_{view_angle}_pose.npy"
    
    if data_path.exists():
        return np.load(data_path)
    return None


def get_dataset_info(data_root: str) -> Dict:
    """
    Get information about the preprocessed dataset.
    
    Args:
        data_root: Root directory containing preprocessed data
        
    Returns:
        Dictionary containing dataset information
    """
    data_root = Path(data_root)
    
    info = {
        'has_splits': (data_root / 'data_splits.json').exists(),
        'has_statistics': (data_root / 'dataset_statistics.json').exists(),
        'n_subjects': 0,
        'n_sequences': 0
    }
    
    # Count subjects and sequences
    subject_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    info['n_subjects'] = len(subject_dirs)
    
    for subject_dir in subject_dirs:
        gei_files = list(subject_dir.glob("*_gei.npy"))
        info['n_sequences'] += len(gei_files)
    
    # Load splits if available
    if info['has_splits']:
        with open(data_root / 'data_splits.json', 'r') as f:
            splits = json.load(f)
        info['splits'] = {k: len(v) for k, v in splits.items()}
    
    # Load statistics if available
    if info['has_statistics']:
        with open(data_root / 'dataset_statistics.json', 'r') as f:
            info['statistics'] = json.load(f)
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Data loader utilities for gait recognition")
    print("\nExample usage:")
    print("""
    from data_loader import GaitDataLoader
    
    # Create data loaders
    loaders = GaitDataLoader.create_loaders(
        data_root='path/to/preprocessed/data',
        data_type='gei',
        batch_size=32
    )
    
    # Use in training loop
    for batch_data, batch_labels, batch_metadata in loaders['train']:
        # Train your model
        pass
    """)
