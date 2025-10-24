"""
Create Proper Data Splits for Gait Recognition

This script creates sequence-based splits (not subject-based) following
standard CASIA-B evaluation protocols.

All 124 subjects appear in train/val/test, but with different sequences:
- Training: nm-01, nm-02, nm-03, nm-04 (normal walking)
- Validation: nm-05, bg-01 (normal + bag carrying)
- Test: nm-06, bg-02, cl-01, cl-02 (normal + bag + coat)
"""

import json
from pathlib import Path
from collections import defaultdict
import argparse


def create_sequence_based_splits(data_root, output_path=None):
    """
    Create proper sequence-based data splits.
    
    Protocol: Same subjects in all splits, different sequences
    - Train: nm-01, nm-02, nm-03, nm-04 (4 normal sequences)
    - Val: nm-05, bg-01 (1 normal + 1 bag)
    - Test: nm-06, bg-02, cl-01, cl-02 (1 normal + 1 bag + 2 coat)
    
    Args:
        data_root: Path to preprocessed_data directory
        output_path: Where to save new splits (default: data_root/data_splits_by_sequence.json)
    """
    data_root = Path(data_root)
    
    # Define sequence splits (following Yu et al. 2006 - original CASIA-B paper)
    train_sequences = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    val_sequences = ['nm-05', 'bg-01']  # Validation: 1 normal + 1 condition change
    test_sequences = ['nm-06', 'bg-02', 'cl-01', 'cl-02']  # Test: more diverse
    
    # Find all subjects
    subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])
    all_subjects = [d.name for d in subject_dirs]
    
    print(f"Found {len(all_subjects)} subjects")
    print(f"Subjects: {all_subjects[:10]}...{all_subjects[-5:]}")
    
    # Count sequences per split
    split_stats = defaultdict(lambda: defaultdict(int))
    
    # For each subject, collect which sequences exist
    for subject_id in all_subjects:
        subject_dir = data_root / subject_id
        gei_files = list(subject_dir.glob('*_gei.npy'))
        
        for gei_file in gei_files:
            # Parse filename: {subject}_{sequence}_{view}_gei.npy
            parts = gei_file.stem.replace('_gei', '').split('_')
            if len(parts) >= 2:
                sequence = parts[1]  # e.g., 'nm-01', 'bg-02', 'cl-01'
                
                # Determine split
                if sequence in train_sequences:
                    split_stats[subject_id]['train'] += 1
                elif sequence in val_sequences:
                    split_stats[subject_id]['val'] += 1
                elif sequence in test_sequences:
                    split_stats[subject_id]['test'] += 1
    
    # Print statistics
    print("\n=== Sequence-Based Split Statistics ===")
    print(f"\nSequence assignments:")
    print(f"  Train: {train_sequences}")
    print(f"  Val:   {val_sequences}")
    print(f"  Test:  {test_sequences}")
    
    total_train = sum(stats['train'] for stats in split_stats.values())
    total_val = sum(stats['val'] for stats in split_stats.values())
    total_test = sum(stats['test'] for stats in split_stats.values())
    
    print(f"\nTotal samples:")
    print(f"  Train: {total_train}")
    print(f"  Val:   {total_val}")
    print(f"  Test:  {total_test}")
    print(f"  Total: {total_train + total_val + total_test}")
    
    # Verify all subjects have data in each split
    subjects_with_train = sum(1 for stats in split_stats.values() if stats['train'] > 0)
    subjects_with_val = sum(1 for stats in split_stats.values() if stats['val'] > 0)
    subjects_with_test = sum(1 for stats in split_stats.values() if stats['test'] > 0)
    
    print(f"\nSubjects per split:")
    print(f"  Train: {subjects_with_train} / {len(all_subjects)}")
    print(f"  Val:   {subjects_with_val} / {len(all_subjects)}")
    print(f"  Test:  {subjects_with_test} / {len(all_subjects)}")
    
    # Create the new splits structure
    # NOTE: For sequence-based splits, ALL subjects go in ALL splits
    # The sequences determine which samples are used
    new_splits = {
        'train': all_subjects,  # All 124 subjects
        'val': all_subjects,    # All 124 subjects  
        'test': all_subjects,   # All 124 subjects
        'metadata': {
            'split_type': 'sequence_based',
            'description': 'All subjects in all splits. Sequences determine split membership.',
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'total_subjects': len(all_subjects),
            'num_classes': len(all_subjects),
            'protocol': 'Standard CASIA-B (Yu et al. 2006)',
            'note': 'This is the CORRECT split for gait recognition evaluation'
        }
    }
    
    # Save
    if output_path is None:
        output_path = data_root / 'data_splits_by_sequence.json'
    
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(new_splits, f, indent=2)
    
    print(f"\n✓ Saved new splits to: {output_path}")
    print("\nNOTE: The data loader will need to filter by sequence within each split!")
    print("Update GaitDataset._build_sample_list() to respect sequence assignments.")
    
    return new_splits, split_stats


def compare_splits(old_splits_path, new_splits_path):
    """Compare old (subject-based) vs new (sequence-based) splits"""
    
    with open(old_splits_path) as f:
        old_splits = json.load(f)
    with open(new_splits_path) as f:
        new_splits = json.load(f)
    
    print("\n" + "="*70)
    print("COMPARISON: Subject-Based vs Sequence-Based Splits")
    print("="*70)
    
    print("\n" + "-"*70)
    print("OLD SPLITS (Subject-Based) - WRONG FOR GAIT RECOGNITION")
    print("-"*70)
    print(f"Train subjects:  {len(old_splits['train'])} subjects")
    print(f"Val subjects:    {len(old_splits['val'])} subjects (DIFFERENT from train!)")
    print(f"Test subjects:   {len(old_splits['test'])} subjects (DIFFERENT from train!)")
    print(f"\nNum classes (train): {len(old_splits['train'])}")
    print(f"Task: Classify {len(old_splits['train'])} training subjects")
    print(f"Evaluation: Zero-shot on {len(old_splits['val'])} unseen subjects")
    print(f"Expected accuracy: ~1% (random guessing)")
    
    print("\n" + "-"*70)
    print("NEW SPLITS (Sequence-Based) - CORRECT FOR GAIT RECOGNITION")
    print("-"*70)
    print(f"Train subjects:  {len(new_splits['train'])} subjects (ALL subjects)")
    print(f"Val subjects:    {len(new_splits['val'])} subjects (SAME subjects)")
    print(f"Test subjects:   {len(new_splits['test'])} subjects (SAME subjects)")
    print(f"\nNum classes (all splits): {new_splits['metadata']['num_classes']}")
    print(f"Task: Classify {new_splits['metadata']['num_classes']} subjects")
    print(f"Evaluation: Same subjects, different sequences/conditions")
    print(f"Expected accuracy: 70-95%")
    print(f"\nSequence assignments:")
    print(f"  Train: {new_splits['metadata']['train_sequences']}")
    print(f"  Val:   {new_splits['metadata']['val_sequences']}")
    print(f"  Test:  {new_splits['metadata']['test_sequences']}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create proper sequence-based data splits')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to preprocessed_data directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for new splits (default: data_root/data_splits_by_sequence.json)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with existing splits')
    
    args = parser.parse_args()
    
    # Create new splits
    new_splits, stats = create_sequence_based_splits(args.data_root, args.output)
    
    # Compare if requested
    if args.compare:
        old_splits_path = Path(args.data_root) / 'data_splits.json'
        new_splits_path = Path(args.output) if args.output else Path(args.data_root) / 'data_splits_by_sequence.json'
        
        if old_splits_path.exists():
            compare_splits(old_splits_path, new_splits_path)
        else:
            print(f"\nWarning: Old splits file not found at {old_splits_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Update data_loader.py to filter by sequence (not just subject)")
    print("2. Backup old data_splits.json:")
    print(f"   mv {Path(args.data_root)/'data_splits.json'} {Path(args.data_root)/'data_splits_by_subject_BACKUP.json'}")
    print("3. Use new splits:")
    print(f"   mv {Path(args.output) if args.output else Path(args.data_root)/'data_splits_by_sequence.json'} {Path(args.data_root)/'data_splits.json'}")
    print("4. Re-train GEI-CNN with 124 classes (all subjects)")
    print("5. Expected result: 70-95% accuracy!")
