"""
CASIA-B Dataset Loader and Batch Processor

CASIA-B Dataset Structure:
- 124 subjects
- 11 view angles (0°, 18°, 36°, ..., 180°)
- 10 sequences per subject:
  - nm-01 to nm-06: normal walking
  - bg-01 to bg-02: walking with bag
  - cl-01 to cl-02: walking in coat
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from gait_preprocessor import GaitPreprocessor, GaitSequence


class CASIABLoader:
    """Loader for CASIA-B gait dataset"""
    
    # CASIA-B dataset specifications
    VIEW_ANGLES = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    SEQUENCE_TYPES = {
        'nm': list(range(1, 7)),  # Normal walking: nm-01 to nm-06
        'bg': list(range(1, 3)),  # Bag: bg-01 to bg-02
        'cl': list(range(1, 3))   # Coat: cl-01 to cl-02
    }
    
    def __init__(self, dataset_root: str):
        """
        Initialize CASIA-B dataset loader.
        
        Args:
            dataset_root: Root directory of CASIA-B dataset
        """
        self.dataset_root = Path(dataset_root)
        self._validate_dataset()
        
    def _validate_dataset(self):
        """Validate that dataset directory exists and has expected structure"""
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        
        print(f"CASIA-B dataset loaded from: {self.dataset_root}")
        
    def get_subject_ids(self) -> List[str]:
        """
        Get list of all subject IDs in dataset.
        
        Returns:
            List of subject ID strings (e.g., ['001', '002', ...])
        """
        # CASIA-B has subjects numbered 001-124
        subject_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        subject_ids = sorted([d.name for d in subject_dirs if d.name.isdigit()])
        return subject_ids
    
    def get_sequence_path(self, subject_id: str, sequence_type: str, 
                         sequence_num: int, view_angle: str) -> Optional[Path]:
        """
        Get path to a specific sequence.
        
        Args:
            subject_id: Subject ID (e.g., '001')
            sequence_type: Type of sequence ('nm', 'bg', or 'cl')
            sequence_num: Sequence number (1-6 for 'nm', 1-2 for 'bg'/'cl')
            view_angle: View angle (e.g., '090')
            
        Returns:
            Path to sequence directory or None if not found
        """
        # CASIA-B path structure: root/subject_id/sequence_type-sequence_num/view_angle/
        sequence_id = f"{sequence_type}-{sequence_num:02d}"
        sequence_path = self.dataset_root / subject_id / sequence_id / view_angle
        
        if sequence_path.exists():
            return sequence_path
        return None
    
    def load_sequence_frames(self, sequence_path: Path) -> List[np.ndarray]:
        """
        Load all frames from a sequence directory.
        
        Args:
            sequence_path: Path to sequence directory containing frames
            
        Returns:
            List of frames (BGR images)
        """
        # CASIA-B frames are typically named as frame numbers (001.png, 002.png, etc.)
        frame_paths = sorted(sequence_path.glob("*.png")) + sorted(sequence_path.glob("*.jpg"))
        
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def get_all_sequences(self, subject_ids: Optional[List[str]] = None,
                         view_angles: Optional[List[str]] = None,
                         sequence_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Get metadata for all sequences matching the filter criteria.
        
        Args:
            subject_ids: List of subject IDs to include (None = all)
            view_angles: List of view angles to include (None = all)
            sequence_types: List of sequence types to include (None = all)
            
        Returns:
            List of dictionaries containing sequence metadata
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()
        if view_angles is None:
            view_angles = self.VIEW_ANGLES
        if sequence_types is None:
            sequence_types = list(self.SEQUENCE_TYPES.keys())
        
        sequences = []
        
        for subject_id in subject_ids:
            for seq_type in sequence_types:
                for seq_num in self.SEQUENCE_TYPES[seq_type]:
                    for view_angle in view_angles:
                        seq_path = self.get_sequence_path(subject_id, seq_type, seq_num, view_angle)
                        if seq_path is not None:
                            sequences.append({
                                'subject_id': subject_id,
                                'sequence_type': seq_type,
                                'sequence_num': seq_num,
                                'sequence_id': f"{seq_type}-{seq_num:02d}",
                                'view_angle': view_angle,
                                'path': seq_path
                            })
        
        return sequences


class CASIABPreprocessor:
    """Batch preprocessor for CASIA-B dataset"""
    
    def __init__(self, dataset_root: str, output_root: str, 
                 silhouette_size: Tuple[int, int] = (64, 128)):
        """
        Initialize CASIA-B preprocessor.
        
        Args:
            dataset_root: Root directory of CASIA-B dataset
            output_root: Root directory for preprocessed output
            silhouette_size: Target size for normalized silhouettes
        """
        self.loader = CASIABLoader(dataset_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.preprocessor = GaitPreprocessor(silhouette_size=silhouette_size)
        
    def process_sequence(self, sequence_info: Dict) -> Optional[GaitSequence]:
        """
        Process a single sequence.
        
        Args:
            sequence_info: Dictionary containing sequence metadata
            
        Returns:
            Processed GaitSequence or None if processing failed
        """
        try:
            # Load frames
            frames = self.loader.load_sequence_frames(sequence_info['path'])
            
            if len(frames) == 0:
                print(f"Warning: No frames found for {sequence_info['sequence_id']}")
                return None
            
            # Process sequence
            gait_sequence = self.preprocessor.process(
                frames=frames,
                subject_id=sequence_info['subject_id'],
                sequence_id=sequence_info['sequence_id'],
                view_angle=sequence_info['view_angle']
            )
            
            return gait_sequence
            
        except Exception as e:
            print(f"Error processing {sequence_info['sequence_id']}: {str(e)}")
            return None
    
    def process_dataset(self, subject_ids: Optional[List[str]] = None,
                       view_angles: Optional[List[str]] = None,
                       sequence_types: Optional[List[str]] = None,
                       save_output: bool = True) -> List[GaitSequence]:
        """
        Process multiple sequences from the dataset.
        
        Args:
            subject_ids: List of subject IDs to process (None = all)
            view_angles: List of view angles to process (None = all)
            sequence_types: List of sequence types to process (None = all)
            save_output: Whether to save processed data to disk
            
        Returns:
            List of processed GaitSequence objects
        """
        # Get all sequences matching criteria
        sequences = self.loader.get_all_sequences(
            subject_ids=subject_ids,
            view_angles=view_angles,
            sequence_types=sequence_types
        )
        
        print(f"Processing {len(sequences)} sequences...")
        
        processed_sequences = []
        
        # Process each sequence with progress bar
        for seq_info in tqdm(sequences, desc="Processing sequences"):
            gait_seq = self.process_sequence(seq_info)
            
            if gait_seq is not None:
                processed_sequences.append(gait_seq)
                
                if save_output:
                    # Organize output by subject
                    subject_output_dir = self.output_root / gait_seq.subject_id
                    self.preprocessor.save(gait_seq, str(subject_output_dir))
        
        print(f"Successfully processed {len(processed_sequences)}/{len(sequences)} sequences")
        
        return processed_sequences
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
        """
        Create train/val/test splits by subject ID.
        
        Args:
            train_ratio: Proportion of subjects for training
            val_ratio: Proportion of subjects for validation
            test_ratio: Proportion of subjects for testing
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing subject IDs
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Get all subject IDs
        subject_ids = self.loader.get_subject_ids()
        n_subjects = len(subject_ids)
        
        # Shuffle subjects
        np.random.seed(seed)
        shuffled_subjects = np.random.permutation(subject_ids)
        
        # Calculate split indices
        train_end = int(n_subjects * train_ratio)
        val_end = train_end + int(n_subjects * val_ratio)
        
        splits = {
            'train': shuffled_subjects[:train_end].tolist(),
            'val': shuffled_subjects[train_end:val_end].tolist(),
            'test': shuffled_subjects[val_end:].tolist()
        }
        
        # Save splits to file
        splits_path = self.output_root / 'data_splits.json'
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Data splits created: {len(splits['train'])} train, "
              f"{len(splits['val'])} val, {len(splits['test'])} test")
        print(f"Splits saved to: {splits_path}")
        
        return splits
    
    def generate_statistics(self, processed_sequences: List[GaitSequence]) -> Dict:
        """
        Generate statistics about the processed dataset.
        
        Args:
            processed_sequences: List of processed sequences
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'n_sequences': len(processed_sequences),
            'n_subjects': len(set(seq.subject_id for seq in processed_sequences)),
            'n_views': len(set(seq.view_angle for seq in processed_sequences)),
            'avg_frames_per_sequence': np.mean([seq.metadata['n_frames'] 
                                               for seq in processed_sequences]),
            'avg_valid_poses': np.mean([seq.metadata['n_valid_poses'] 
                                       for seq in processed_sequences]),
            'sequence_types': {}
        }
        
        # Count sequences by type
        for seq in processed_sequences:
            seq_type = seq.sequence_id.split('-')[0]
            stats['sequence_types'][seq_type] = stats['sequence_types'].get(seq_type, 0) + 1
        
        return stats
    
    def close(self):
        """Release resources"""
        self.preprocessor.close()


def main():
    """Example usage for preprocessing CASIA-B dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CASIA-B Gait Dataset')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to CASIA-B dataset root directory')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Path to output directory for preprocessed data')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='Specific subject IDs to process (default: all)')
    parser.add_argument('--views', type=str, nargs='+', default=None,
                       help='Specific view angles to process (default: all)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                       choices=['nm', 'bg', 'cl'],
                       help='Specific sequence types to process (default: all)')
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = CASIABPreprocessor(
        dataset_root=args.dataset_root,
        output_root=args.output_root
    )
    
    # Process dataset
    processed = preprocessor.process_dataset(
        subject_ids=args.subjects,
        view_angles=args.views,
        sequence_types=args.sequences
    )
    
    # Generate and save statistics
    stats = preprocessor.generate_statistics(processed)
    stats_path = Path(args.output_root) / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDataset statistics saved to: {stats_path}")
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # Create data splits if requested
    if args.create_splits:
        preprocessor.create_splits()
    
    preprocessor.close()
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
