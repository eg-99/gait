#!/usr/bin/env python3
"""
Video Segmentation Script
Extracts object silhouettes from video using YOLO + SAM2
"""

import argparse
import cv2
import numpy as np
import os
from ultralytics import SAM, YOLO

# COCO class names to IDs mapping (common objects)
COCO_CLASS_NAMES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79,
}


def get_object_id_from_name(object_name):
    """
    Get COCO class ID from object name.
    
    Args:
        object_name: Name of the object (case-insensitive)
    
    Returns:
        tuple: (class_id, class_name) or (None, None) if not found
    """
    object_name_lower = object_name.lower().strip()
    
    # Direct lookup
    if object_name_lower in COCO_CLASS_NAMES:
        return COCO_CLASS_NAMES[object_name_lower], object_name_lower
    
    # Try to find partial matches
    for name, class_id in COCO_CLASS_NAMES.items():
        if object_name_lower in name or name in object_name_lower:
            return class_id, name
    
    return None, None


def get_bbox_from_mask(mask, padding=10):
    """Get bounding box from mask with optional padding"""
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
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
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


def process_video(
    video_path,
    output_dir,
    object_id=0,
    object_name="person",
    output_width=None,
    output_height=None,
    display=False,
    yolo_update_interval=10,
    sam_imgsz=512
):
    """
    Process video and extract object silhouettes
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save silhouettes
        object_id: COCO class ID (0=person, 16=dog, etc.)
        object_name: Name of object for logging
        output_width: Width of output images (None = original)
        output_height: Height of output images (None = original)
        display: Whether to show live preview
        yolo_update_interval: Re-detect with YOLO every N frames
        sam_imgsz: SAM processing image size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    yolo = YOLO("yolov8n.pt")
    sam_model = SAM("sam2_t.pt")
    
    # Get initial bounding box from first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    
    # Get original frame dimensions
    original_height, original_width = first_frame.shape[:2]
    print(f"Video dimensions: {original_width}x{original_height}")
    
    # Determine output dimensions
    if output_width is None and output_height is None:
        # Keep original size
        output_width, output_height = original_width, original_height
        resize_output = False
    elif output_width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = original_width / original_height
        output_width = int(output_height * aspect_ratio)
        resize_output = True
    elif output_height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = original_height / original_width
        output_height = int(output_width * aspect_ratio)
        resize_output = True
    else:
        resize_output = True
    
    print(f"Output dimensions: {output_width}x{output_height}")
    
    # Detect object in first frame
    yolo_result = yolo(first_frame, verbose=False)[0]
    obj_box = None
    
    for box in yolo_result.boxes:
        if int(box.cls) == object_id:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_box = [int(x1), int(y1), int(x2), int(y2)]
            print(f"Found {object_name} at bounding box: {obj_box}")
            break
    
    if obj_box is None:
        print(f"Error: No {object_name} detected in first frame!")
        cap.release()
        return
    
    # Process video frame by frame
    frame_count = 0
    print("Starting tracking and saving silhouettes...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Re-detect with YOLO periodically - match to closest detection
        if frame_count % yolo_update_interval == 0:
            yolo_result = yolo(frame, verbose=False)[0]
            # Collect all detections of the target class
            detections = []
            for box in yolo_result.boxes:
                if int(box.cls) == object_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
        
            # Find the best matching detection (closest to current box)
            if detections:
                best_match = find_best_matching_box(obj_box, detections, iou_threshold=0.3)
                if best_match:
                    obj_box = best_match
                # If no good match found, keep using current box (prevents jumping)
        
        # Run SAM with bounding box
        result = sam_model(frame, imgsz=sam_imgsz, bboxes=obj_box, verbose=False)[0]
        
        # Process mask and save silhouette
        if result.masks is not None and len(result.masks.data) > 0:
            mask = result.masks.data[0].cpu().numpy()
            if len(mask.shape) == 2:
                # Resize mask if needed
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                
                # Create white silhouette on black background
                silhouette = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                silhouette[mask > 0] = 255
                
                # Resize silhouette if needed
                if resize_output:
                    silhouette = cv2.resize(
                        silhouette,
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Save silhouette
                output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(output_path, silhouette)
                
                # Update bounding box from mask for next frame
                new_box = get_bbox_from_mask(mask)
                if new_box:
                    new_box[2] = min(frame.shape[1], new_box[2])
                    new_box[3] = min(frame.shape[0], new_box[3])
                    obj_box = new_box
        
        # Display (if enabled)
        if display:
            img = frame.copy()
            if result.masks is not None and len(result.masks.data) > 0:
                mask = result.masks.data[0].cpu().numpy()
                if len(mask.shape) == 2:
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    overlay = img.copy()
                    overlay[mask > 0] = [0, 255, 0]
                    img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
                    
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) > 0:
                        cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
            
            if obj_box:
                cv2.rectangle(
                    img,
                    (obj_box[0], obj_box[1]),
                    (obj_box[2], obj_box[3]),
                    (255, 0, 0),
                    2
                )
            
            cv2.putText(
                img,
                f"Frame {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            cv2.imshow('Live Tracking', img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Stopped by user")
                cap.release()
                if display:
                    cv2.destroyAllWindows()
                return
    
    print(f"Saved {frame_count} silhouettes to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract object silhouettes from video using YOLO + SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings (person, original size, no display)
  python segment_video.py video.mp4
  
  # Process dog - automatically detects class ID from name
  python segment_video.py video.mp4 --object-name dog
  
  # Process with custom output size and display
  python segment_video.py video.mp4 --output-dir output --width 512 --height 512 --display
  
  # Process cat with custom size
  python segment_video.py video.mp4 --object-name cat --width 256
  
  # Process with only width specified (height auto-calculated)
  python segment_video.py video.mp4 --object-name car --width 640 --display
        """
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for silhouettes (default: output_silhouettes_<video_name>)"
    )
    
    parser.add_argument(
        "--object-name",
        type=str,
        default="person",
        help="Name of object to detect (e.g., 'dog', 'person', 'cat', 'car'). Automatically infers COCO class ID. (default: person)"
    )
    
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="COCO class ID (0=person, 16=dog, etc.). If not specified, will be inferred from --object-name"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output image width (maintains aspect ratio if height not specified)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height (maintains aspect ratio if width not specified)"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable live preview display"
    )
    
    parser.add_argument(
        "--yolo-interval",
        type=int,
        default=10,
        help="Re-detect with YOLO every N frames (default: 10)"
    )
    
    parser.add_argument(
        "--sam-imgsz",
        type=int,
        default=512,
        help="SAM processing image size (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Infer object ID from name if not explicitly provided
    if args.object_id is None:
        object_id, detected_name = get_object_id_from_name(args.object_name)
        if object_id is None:
            print(f"Error: Could not find COCO class for '{args.object_name}'")
            print(f"Available objects: {', '.join(sorted(COCO_CLASS_NAMES.keys()))}")
            return
        args.object_id = object_id
        args.object_name = detected_name  # Use canonical name
        print(f"Detected object: '{detected_name}' (COCO class ID: {object_id})")
    else:
        # If ID is explicitly provided, validate the name exists
        if args.object_name.lower() not in COCO_CLASS_NAMES:
            print(f"Warning: '{args.object_name}' may not be a valid COCO class name")
            print(f"Using provided object ID: {args.object_id}")
    
    # Set default output directory
    if args.output_dir is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = f"output_silhouettes_{video_name}"
    
    # Process video
    process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        object_id=args.object_id,
        object_name=args.object_name,
        output_width=args.width,
        output_height=args.height,
        display=args.display,
        yolo_update_interval=args.yolo_interval,
        sam_imgsz=args.sam_imgsz
    )


if __name__ == "__main__":
    main()

