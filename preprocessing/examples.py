"""
Example Usage Script

Demonstrates basic usage of the gait preprocessing pipeline.
"""

import numpy as np
from pathlib import Path
from gait_preprocessor import GaitPreprocessor
from casia_b_loader import CASIABPreprocessor
from visualization import visualize_sample, compare_geis
from data_loader import GaitDataLoader, get_dataset_info


def example_1_preprocess_single_video():
    """Example 1: Preprocess a single video file"""
    print("=" * 60)
    print("Example 1: Preprocessing a Single Video")
    print("=" * 60)
    
    preprocessor = GaitPreprocessor(silhouette_size=(64, 128))
    
    # Load video (replace with your video path)
    video_path = "path/to/your/video.mp4"
    # frames = preprocessor.load_video(video_path)
    
    # Or load image sequence
    # frames = preprocessor.load_image_sequence("path/to/frames", pattern="*.png")
    
    # Process
    # gait_data = preprocessor.process(
    #     frames=frames,
    #     subject_id="001",
    #     sequence_id="nm-01",
    #     view_angle="090"
    # )
    
    # Save results
    # preprocessor.save(gait_data, "output/single_video")
    
    # Access processed data
    # print(f"GEI shape: {gait_data.gei.shape}")
    # print(f"Silhouettes shape: {gait_data.silhouettes.shape}")
    # print(f"Pose trajectories shape: {gait_data.pose_trajectories.shape}")
    
    preprocessor.close()
    print("\n✓ Example 1 complete\n")


def example_2_preprocess_casia_b():
    """Example 2: Preprocess CASIA-B dataset"""
    print("=" * 60)
    print("Example 2: Preprocessing CASIA-B Dataset")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = CASIABPreprocessor(
        dataset_root='/path/to/CASIA-B',
        output_root='preprocessed_data'
    )
    
    # Process a subset (for testing)
    processed = preprocessor.process_dataset(
        subject_ids=['001', '002'],  # Process first 2 subjects
        view_angles=['090'],          # Only 90-degree view
        sequence_types=['nm']         # Only normal walking
    )
    
    print(f"\nProcessed {len(processed)} sequences")
    
    # Create train/val/test splits
    splits = preprocessor.create_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print(f"Train subjects: {len(splits['train'])}")
    print(f"Val subjects: {len(splits['val'])}")
    print(f"Test subjects: {len(splits['test'])}")
    
    # Generate statistics
    stats = preprocessor.generate_statistics(processed)
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {stats['n_sequences']}")
    print(f"  Unique subjects: {stats['n_subjects']}")
    print(f"  Avg frames/sequence: {stats['avg_frames_per_sequence']:.1f}")
    
    preprocessor.close()
    print("\n✓ Example 2 complete\n")


def example_3_load_for_training():
    """Example 3: Load preprocessed data for model training"""
    print("=" * 60)
    print("Example 3: Loading Data for Training")
    print("=" * 60)
    
    # Get dataset information
    info = get_dataset_info('preprocessed_data')
    print(f"Dataset Info:")
    print(f"  Subjects: {info['n_subjects']}")
    print(f"  Sequences: {info['n_sequences']}")
    print(f"  Has splits: {info['has_splits']}")
    
    # Create DataLoaders for GEI-based training
    loaders = GaitDataLoader.create_loaders(
        data_root='preprocessed_data',
        data_type='gei',
        batch_size=16,
        num_workers=4
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches: {len(loaders['val'])}")
    print(f"  Test batches: {len(loaders['test'])}")
    
    # Example training loop
    print("\nExample batch from training set:")
    for batch_data, batch_labels, batch_metadata in loaders['train']:
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  First subject ID: {batch_metadata['subject_id'][0]}")
        break
    
    print("\n✓ Example 3 complete\n")


def example_4_visualize():
    """Example 4: Visualize preprocessed data"""
    print("=" * 60)
    print("Example 4: Visualizing Preprocessed Data")
    print("=" * 60)
    
    # Visualize a single sample
    visualize_sample(
        data_root='preprocessed_data',
        subject_id='001',
        sequence_id='nm-01',
        view_angle='090',
        save_dir='visualizations'
    )
    
    print("\n✓ Example 4 complete\n")


def example_5_compare_subjects():
    """Example 5: Compare GEIs from different subjects"""
    print("=" * 60)
    print("Example 5: Comparing Different Subjects")
    print("=" * 60)
    
    # Load GEIs from multiple subjects
    gei1 = np.load('preprocessed_data/001/001_nm-01_090_gei.npy')
    gei2 = np.load('preprocessed_data/002/002_nm-01_090_gei.npy')
    
    # Compare side by side
    compare_geis(
        [gei1, gei2],
        ['Subject 001', 'Subject 002'],
        save_path='visualizations/comparison.png'
    )
    
    print("\n✓ Example 5 complete\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GAIT PREPROCESSING PIPELINE - EXAMPLES")
    print("=" * 60 + "\n")
    
    print("Available examples:")
    print("  1. Preprocess single video")
    print("  2. Preprocess CASIA-B dataset (subset)")
    print("  3. Load data for training")
    print("  4. Visualize preprocessed data")
    print("  5. Compare subjects")
    print()
    
    # Uncomment the examples you want to run
    # example_1_preprocess_single_video()
    # example_2_preprocess_casia_b()
    # example_3_load_for_training()
    # example_4_visualize()
    # example_5_compare_subjects()
    
    print("=" * 60)
    print("To run examples, uncomment them in the main() function")
    print("Make sure to update paths to match your data location")
    print("=" * 60)


if __name__ == "__main__":
    main()
