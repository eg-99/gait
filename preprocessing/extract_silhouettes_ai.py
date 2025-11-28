"""
Robust Silhouette Extractor using AI Background Removal

Works on ANY background - no assumptions about color or motion.
Uses rembg's AI model to detect and extract the person.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from rembg import remove
from PIL import Image
import io


def extract_silhouettes_ai(video_path, output_dir, target_size=(64, 128), save_every_n=1):
    """
    Extract silhouettes using AI background removal.
    Works on ANY background.
    
    Args:
        video_path: Path to video
        output_dir: Where to save silhouettes
        target_size: (width, height) for output
        save_every_n: Save every Nth frame (1 = all frames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0
    
    print(f"Processing {Path(video_path).name} with AI background removal...")
    print("This will be slower but works on ANY background!")
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame if specified
        if frame_idx % save_every_n == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_frame)
            
            # Remove background using AI
            output = remove(pil_img)
            
            # Convert back to numpy
            output_np = np.array(output)
            
            # Extract alpha channel (the mask)
            if output_np.shape[2] == 4:
                alpha = output_np[:, :, 3]
            else:
                # Fallback if no alpha
                gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
                _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find largest contour
            contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                
                # Create clean silhouette
                silhouette = np.zeros_like(alpha)
                cv2.drawContours(silhouette, [largest], -1, 255, thickness=cv2.FILLED)
                
                # Get bounding box with padding
                x, y, w, h = cv2.boundingRect(largest)
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                
                # Crop
                cropped = silhouette[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    # Resize maintaining aspect ratio
                    h_crop, w_crop = cropped.shape
                    aspect = w_crop / h_crop
                    target_aspect = target_size[0] / target_size[1]
                    
                    if aspect > target_aspect:
                        new_w = target_size[0]
                        new_h = int(new_w / aspect)
                    else:
                        new_h = target_size[1]
                        new_w = int(new_h * aspect)
                    
                    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Center in canvas
                    final = np.zeros(target_size[::-1], dtype=np.uint8)
                    y_offset = (target_size[1] - new_h) // 2
                    x_offset = (target_size[0] - new_w) // 2
                    final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    
                    # Save
                    output_path = output_dir / f"silhouette_{saved:05d}.png"
                    cv2.imwrite(str(output_path), final)
                    saved += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"\n✓ Saved {saved} silhouettes to {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract silhouettes using AI - works on ANY background")
    parser.add_argument("--input", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", default="64x128", help="Output size WxH (default: 64x128)")
    parser.add_argument("--every", type=int, default=1, help="Save every Nth frame (default: 1 = all)")
    args = parser.parse_args()
    
    # Parse size
    w, h = map(int, args.size.split('x'))
    
    extract_silhouettes_ai(args.input, args.output, target_size=(w, h), save_every_n=args.every)


if __name__ == "__main__":
    main()
