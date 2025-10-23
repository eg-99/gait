"""
Test Script - Verify Installation and Basic Functionality

Run this script to make sure everything is installed correctly.
"""

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("  ✓ OpenCV")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("  ✓ MediaPipe")
    except ImportError as e:
        print(f"  ✗ MediaPipe: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ NumPy")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ Matplotlib")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False
    
    try:
        import torch
        print("  ✓ PyTorch")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    try:
        import imageio
        print("  ✓ imageio")
    except ImportError as e:
        print(f"  ✗ imageio: {e}")
        return False
    
    return True


def test_modules():
    """Test that our modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from gait_preprocessor import GaitPreprocessor, SilhouetteProcessor, PoseProcessor
        print("  ✓ gait_preprocessor")
    except ImportError as e:
        print(f"  ✗ gait_preprocessor: {e}")
        return False
    
    try:
        from casia_b_loader import CASIABLoader, CASIABPreprocessor
        print("  ✓ casia_b_loader")
    except ImportError as e:
        print(f"  ✗ casia_b_loader: {e}")
        return False
    
    try:
        from data_loader import GaitDataset, GaitDataLoader, SequenceDataset
        print("  ✓ data_loader")
    except ImportError as e:
        print(f"  ✗ data_loader: {e}")
        return False
    
    try:
        from visualization import visualize_gei, visualize_silhouette_sequence
        print("  ✓ visualization")
    except ImportError as e:
        print(f"  ✗ visualization: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic preprocessing functionality with dummy data"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        import cv2
        from gait_preprocessor import GaitPreprocessor
        
        # Create preprocessor
        preprocessor = GaitPreprocessor()
        print("  ✓ GaitPreprocessor initialized")
        
        # Create dummy frames (10 frames of 240x320 images)
        dummy_frames = [np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) 
                       for _ in range(10)]
        print("  ✓ Created dummy frames")
        
        # Try to process
        result = preprocessor.process(
            frames=dummy_frames,
            subject_id="test",
            sequence_id="test-01",
            view_angle="090"
        )
        print("  ✓ Processing completed")
        
        # Check outputs
        assert result.silhouettes is not None, "Silhouettes not generated"
        assert result.gei is not None, "GEI not generated"
        assert result.pose_trajectories is not None, "Pose not generated"
        print("  ✓ All outputs generated")
        
        print(f"    - Silhouettes shape: {result.silhouettes.shape}")
        print(f"    - GEI shape: {result.gei.shape}")
        print(f"    - Pose shape: {result.pose_trajectories.shape}")
        
        preprocessor.close()
        print("  ✓ Preprocessor closed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_silhouette_processing():
    """Test silhouette processing specifically"""
    print("\nTesting silhouette processing...")
    
    try:
        import numpy as np
        import cv2
        from gait_preprocessor import SilhouetteProcessor
        
        processor = SilhouetteProcessor(target_size=(64, 128))
        
        # Create a dummy frame with a white blob
        frame = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (200, 200), 255, -1)
        
        # Extract silhouette
        silhouette = processor.extract_silhouette(frame)
        assert silhouette.shape == (240, 320), "Silhouette shape incorrect"
        print("  ✓ Silhouette extraction")
        
        # Normalize
        normalized = processor.normalize_silhouette(silhouette)
        assert normalized.shape == (128, 64), "Normalized shape incorrect"
        print("  ✓ Silhouette normalization")
        
        # Generate GEI
        silhouettes = np.stack([normalized] * 10)
        gei = processor.generate_gei(silhouettes)
        assert gei.shape == (128, 64), "GEI shape incorrect"
        print("  ✓ GEI generation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during silhouette test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_processing():
    """Test pose processing"""
    print("\nTesting pose processing...")
    
    try:
        import numpy as np
        from gait_preprocessor import PoseProcessor
        
        processor = PoseProcessor()
        print("  ✓ PoseProcessor initialized")
        
        # Create dummy frame
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        
        # Try to extract pose (might fail on random image, that's ok)
        pose = processor.extract_pose(frame)
        print("  ✓ Pose extraction attempted")
        
        if pose is not None:
            assert pose.shape == (33, 3), "Pose shape incorrect"
            print("  ✓ Pose shape correct")
            
            # Test angle computation
            angles = processor.compute_joint_angles(pose)
            print(f"  ✓ Joint angles computed: {list(angles.keys())}")
        else:
            print("  ℹ No pose detected (expected with random image)")
        
        processor.close()
        print("  ✓ PoseProcessor closed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during pose test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("GAIT PREPROCESSING - INSTALLATION TEST")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Package Imports", test_imports()))
    
    # Test 2: Modules
    if results[0][1]:  # Only if imports passed
        results.append(("Custom Modules", test_modules()))
    else:
        print("\nSkipping module tests due to import failures")
        print("Please install missing packages: pip install -r requirements.txt")
        return
    
    # Test 3: Basic functionality
    if results[1][1]:  # Only if modules passed
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("Silhouette Processing", test_silhouette_processing()))
        results.append(("Pose Processing", test_pose_processing()))
    else:
        print("\nSkipping functionality tests due to module import failures")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready to start preprocessing!")
        print("Next steps:")
        print("  1. Download CASIA-B dataset")
        print("  2. Run: python casia_b_loader.py --dataset_root /path/to/CASIA-B --output_root output")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Common fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Make sure you're in the gait_preprocessing directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
