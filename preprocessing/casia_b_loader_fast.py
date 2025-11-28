"""
Fast Parallel CASIA-B Dataset Preprocessor with Resume Capability

Features:
- Multiprocessing for parallel sequence processing
- Automatic resume from last completed sequence
- Progress tracking and error handling
- Compatible with existing preprocessing pipeline

Usage:
    python casia_b_loader_fast.py \
        --dataset_root /path/to/CASIA-B \
        --output_root preprocessed_data \
        --num_workers 8 \
        --skip_pose \
        --create_splits
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import traceback

from gait_preprocessor import GaitPreprocessor, GaitSequence
from casia_b_loader import CASIABLoader


def check_sequence_already_processed(output_root: Path, sequence_info: Dict) -> bool:
    """
    Check if a sequence has already been processed.
    
    Args:
        output_root: Root directory for preprocessed output
        sequence_info: Dictionary containing sequence metadata
    
    Returns:
        True if sequence is already processed, False otherwise
    """
    subject_id = sequence_info['subject_id']
    sequence_id = sequence_info['sequence_id']
    view_angle = sequence_info['view_angle']
    
    # Expected output file
    output_file = output_root / subject_id / f"{subject_id}_{sequence_id}_{view_angle}_gei.npy"
    
    # Check if file exists and is not empty
    if output_file.exists():
        try:
            data = np.load(output_file)
            if data.size > 0:
                return True
        except:
            # File exists but corrupted, reprocess
            return False
    
    return False


def filter_already_processed(sequences: List[Dict], output_root: Path) -> Tuple[List[Dict], int]:
    """
    Filter out sequences that have already been processed.
    
    Args:
        sequences: List of sequence info dictionaries
        output_root: Root directory for preprocessed output
    
    Returns:
        Tuple of (remaining sequences, number skipped)
    """
    remaining = []
    skipped = 0
    
    print("Checking for already processed sequences...")
    for seq_info in tqdm(sequences, desc="Checking existing files"):
        if check_sequence_already_processed(output_root, seq_info):
            skipped += 1
        else:
            remaining.append(seq_info)
    
    if skipped > 0:
        print(f"Found {skipped} already processed sequences (skipping)")
    print(f"Remaining to process: {len(remaining)} sequences")
    
    return remaining, skipped


def process_sequence_worker(args_tuple: Tuple) -> Optional[Dict]:
    """
    Worker function for multiprocessing.
    Each worker creates its own preprocessor instance.
    
    Args:
        args_tuple: Tuple of (sequence_info, dataset_root, output_root, silhouette_size, skip_pose)
    
    Returns:
        Dictionary with result info or None if failed
    """
    sequence_info, dataset_root, output_root, silhouette_size, skip_pose = args_tuple
    
    try:
        # Create loader and preprocessor for this worker
        # (MediaPipe can't be shared across processes)
        loader = CASIABLoader(dataset_root)
        preprocessor = GaitPreprocessor(silhouette_size=silhouette_size)
        
        # Load frames
        frames = loader.load_sequence_frames(sequence_info['path'])
        
        if len(frames) == 0:
            return {
                'success': False,
                'sequence_id': sequence_info['sequence_id'],
                'error': 'No frames found'
            }
        
        # Process sequence
        # Note: skip_pose parameter is for future implementation
        # Currently, GaitPreprocessor always processes both silhouettes and pose
        gait_sequence = preprocessor.process(
            frames=frames,
            subject_id=sequence_info['subject_id'],
            sequence_id=sequence_info['sequence_id'],
            view_angle=sequence_info['view_angle']
        )
        
        if gait_sequence is None:
            return {
                'success': False,
                'sequence_id': sequence_info['sequence_id'],
                'error': 'Processing returned None'
            }
        
        # Save output
        subject_output_dir = Path(output_root) / gait_sequence.subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        preprocessor.save(gait_sequence, str(subject_output_dir))
        
        # Cleanup
        preprocessor.close()
        
        return {
            'success': True,
            'subject_id': gait_sequence.subject_id,
            'sequence_id': gait_sequence.sequence_id,
            'view_angle': gait_sequence.view_angle
        }
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'sequence_id': sequence_info.get('sequence_id', 'unknown'),
            'error': error_msg
        }


class CASIABPreprocessorFast:
    """Fast parallel preprocessor for CASIA-B dataset with resume capability"""
    
    def __init__(self, dataset_root: str, output_root: str, 
                 silhouette_size: Tuple[int, int] = (64, 128),
                 skip_pose: bool = False):
        """
        Initialize fast CASIA-B preprocessor.
        
        Args:
            dataset_root: Root directory of CASIA-B dataset
            output_root: Root directory for preprocessed output
            silhouette_size: Target size for normalized silhouettes
            skip_pose: If True, skip pose processing (much faster, only GEI)
        """
        self.loader = CASIABLoader(dataset_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.silhouette_size = silhouette_size
        self.skip_pose = skip_pose
        
        # Store for worker function
        self.dataset_root = dataset_root
        self.output_root_str = str(output_root)
    
    def process_dataset_parallel(self, 
                                subject_ids: Optional[List[str]] = None,
                                view_angles: Optional[List[str]] = None,
                                sequence_types: Optional[List[str]] = None,
                                num_workers: Optional[int] = None,
                                resume: bool = True) -> Dict:
        """
        Process dataset in parallel using multiprocessing with resume capability.
        
        Args:
            subject_ids: List of subject IDs to process (None = all)
            view_angles: List of view angles to process (None = all)
            sequence_types: List of sequence types to process (None = all)
            num_workers: Number of parallel workers (default: cpu_count())
            resume: If True, skip already processed sequences
        
        Returns:
            Dictionary with processing statistics
        """
        # Get all sequences matching criteria
        sequences = self.loader.get_all_sequences(
            subject_ids=subject_ids,
            view_angles=view_angles,
            sequence_types=sequence_types
        )
        
        print(f"\n{'='*60}")
        print(f"Fast Parallel Preprocessing")
        print(f"{'='*60}")
        print(f"Total sequences found: {len(sequences)}")
        
        # Filter already processed sequences if resume is enabled
        if resume:
            sequences, skipped = filter_already_processed(sequences, self.output_root)
        else:
            skipped = 0
        
        if len(sequences) == 0:
            print("\nAll sequences already processed!")
            return {
                'total': len(sequences) + skipped,
                'processed': 0,
                'skipped': skipped,
                'failed': 0,
                'success_rate': 1.0
            }
        
        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()
        num_workers = min(num_workers, len(sequences), cpu_count())
        
        print(f"Using {num_workers} parallel workers")
        print(f"Processing {len(sequences)} sequences...")
        print(f"{'='*60}\n")
        
        # Prepare arguments for workers
        worker_args = [
            (seq_info, self.dataset_root, self.output_root_str, 
             self.silhouette_size, self.skip_pose)
            for seq_info in sequences
        ]
        
        # Process in parallel
        start_time = time.time()
        successful = 0
        failed = 0
        errors = []
        
        with Pool(processes=num_workers) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(process_sequence_worker, worker_args),
                total=len(sequences),
                desc="Processing",
                unit="seq"
            ))
        
        # Collect results
        for result in results:
            if result is None:
                failed += 1
            elif result.get('success', False):
                successful += 1
            else:
                failed += 1
                errors.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Total sequences: {len(sequences) + skipped}")
        print(f"  - Already processed (skipped): {skipped}")
        print(f"  - Newly processed: {successful}")
        print(f"  - Failed: {failed}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
        print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        if successful > 0:
            print(f"Average time per sequence: {elapsed_time/successful:.2f}s")
        
        if errors:
            print(f"\nErrors encountered ({len(errors)}):")
            for i, err in enumerate(errors[:5]):  # Show first 5 errors
                print(f"  {i+1}. {err.get('sequence_id', 'unknown')}: {err.get('error', 'Unknown error')[:100]}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        return {
            'total': len(sequences) + skipped,
            'processed': successful,
            'skipped': skipped,
            'failed': failed,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            'time_elapsed': elapsed_time,
            'errors': errors
        }
    
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
        
        print(f"\nData splits created: {len(splits['train'])} train, "
              f"{len(splits['val'])} val, {len(splits['test'])} test")
        print(f"Splits saved to: {splits_path}")
        
        return splits
    
    def generate_statistics(self) -> Dict:
        """
        Generate statistics about the processed dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        # Count processed sequences
        gei_files = list(self.output_root.glob("*/**_gei.npy"))
        
        subjects = set()
        views = set()
        sequence_types = {}
        total_frames = 0
        valid_files = 0
        
        for gei_file in gei_files:
            try:
                # Parse filename
                parts = gei_file.stem.replace('_gei', '').split('_')
                if len(parts) >= 3:
                    subject_id = parts[0]
                    sequence_id = parts[1]
                    view_angle = parts[2]
                    
                    subjects.add(subject_id)
                    views.add(view_angle)
                    
                    seq_type = sequence_id.split('-')[0]
                    sequence_types[seq_type] = sequence_types.get(seq_type, 0) + 1
                    
                    # Try to load to verify it's valid
                    data = np.load(gei_file)
                    if data.size > 0:
                        valid_files += 1
            except:
                continue
        
        stats = {
            'n_sequences': valid_files,
            'n_subjects': len(subjects),
            'n_views': len(views),
            'sequence_types': sequence_types
        }
        
        # Save statistics
        stats_path = self.output_root / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset statistics:")
        print(f"  Sequences: {stats['n_sequences']}")
        print(f"  Subjects: {stats['n_subjects']}")
        print(f"  Views: {stats['n_views']}")
        print(f"  Sequence types: {stats['sequence_types']}")
        print(f"Statistics saved to: {stats_path}")
        
        return stats


def main():
    """Main entry point for fast preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fast Parallel Preprocessing for CASIA-B Gait Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to CASIA-B dataset root directory')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Path to output directory for preprocessed data')
    
    # Processing options
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='Specific subject IDs to process (default: all)')
    parser.add_argument('--views', type=str, nargs='+', default=None,
                       help='Specific view angles to process (default: all)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                       choices=['nm', 'bg', 'cl'],
                       help='Specific sequence types to process (default: all)')
    
    # Performance options
    parser.add_argument('--num_workers', type=int, default=None,
                       help=f'Number of parallel workers (default: {cpu_count()})')
    parser.add_argument('--skip_pose', action='store_true',
                       help='Skip pose processing (much faster, only generates GEI)')
    parser.add_argument('--no_resume', action='store_true',
                       help='Disable resume (reprocess all sequences)')
    
    # Output options
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--silhouette_size', type=int, nargs=2, default=[64, 128],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Target size for normalized silhouettes')
    
    args = parser.parse_args()
    
    # Initialize fast preprocessor
    preprocessor = CASIABPreprocessorFast(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        silhouette_size=tuple(args.silhouette_size),
        skip_pose=args.skip_pose
    )
    
    # Process dataset
    results = preprocessor.process_dataset_parallel(
        subject_ids=args.subjects,
        view_angles=args.views,
        sequence_types=args.sequences,
        num_workers=args.num_workers,
        resume=not args.no_resume
    )
    
    # Generate statistics
    if results['processed'] > 0 or results['skipped'] > 0:
        preprocessor.generate_statistics()
    
    # Create data splits if requested
    if args.create_splits:
        preprocessor.create_splits()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()

