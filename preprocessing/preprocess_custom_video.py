#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video → black-and-white walking silhouettes (CASIA-B style)

- Extracts stable binary silhouettes using background subtraction + morphology
- Keeps only the largest connected component (the person)
- Optional center-crop around the subject with temporal smoothing
- Saves per-frame PNGs with white foreground (255) on black background (0)

Usage:
	python extract_silhouettes.py --input /path/to/video_or_dir --out out_dir
	# optional args:
	# --size 320x240 --pad 1.6 --history 800 --varth 16 --warmup 150 --lr 0.001
	# --minarea 2500 --kernel 5 --no-crop

Notes:
- Works best with mostly static cameras (as in CASIA-B).
- For moving cameras, first stabilize or skip cropping (--no-crop).
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def parse_size(s):
	w, h = s.lower().split("x")
	return int(w), int(h)

def ensure_dir(p):
	Path(p).mkdir(parents=True, exist_ok=True)

def largest_cc(binary):
	# binary: 0/255 uint8
	num, labels, stats, _ = cv2.connectedComponentsWithStats((binary>0).astype(np.uint8), connectivity=8)
	if num <= 1:
		return None, None  # no FG
	# skip label 0 (background)
	areas = stats[1:, cv2.CC_STAT_AREA]
	idx = int(np.argmax(areas)) + 1
	mask = (labels == idx).astype(np.uint8) * 255
	bbox = stats[idx, :4]  # x,y,w,h
	return mask, bbox

def smooth_bbox(prev, cur, alpha=0.2):
	if prev is None: return cur
	px, py, pw, ph = prev
	cx, cy, cw, ch = cur
	return (
		int((1-alpha)*px + alpha*cx),
		int((1-alpha)*py + alpha*cy),
		int((1-alpha)*pw + alpha*cw),
		int((1-alpha)*ph + alpha*ch),
	)

def pad_and_square_crop(frame, bbox, pad_scale=1.6):
	h, w = frame.shape[:2]
	x, y, bw, bh = bbox
	# expand bbox by pad_scale around center
	cx, cy = x + bw/2.0, y + bh/2.0
	side = int(max(bw, bh) * pad_scale)
	x1 = int(cx - side/2)
	y1 = int(cy - side/2)
	x2 = x1 + side
	y2 = y1 + side
	# clip to frame
	x1 = max(0, x1); y1 = max(0, y1)
	x2 = min(w, x2); y2 = min(h, y2)
	return x1, y1, max(1, x2-x1), max(1, y2-y1)

def binarize_foreground(fg, kernel_size=5, min_area=2500):
	# Initial hard threshold (MOG2 may already be near-binary if detectShadows=False)
	_, th = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# Morphology: open (remove specks) then close (fill gaps)
	k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
	th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
	# Keep largest connected component only (the person)
	mask, bbox = largest_cc(th)
	if mask is None:
		return None, None
	# Reject tiny detections
	if cv2.countNonZero(mask) < min_area:
		return None, None
	# Hole filling via contour fill
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	fill = np.zeros_like(mask)
	cv2.drawContours(fill, contours, -1, 255, thickness=cv2.FILLED)
	return fill, bbox

def process_video(vid_path, out_dir, out_size=(320,240), history=800, var_th=16, warmup=150, lr=0.001,
				  kernel=5, min_area=2500, pad=1.6, crop=True):
	ensure_dir(out_dir)
	cap = cv2.VideoCapture(vid_path)
	if not cap.isOpened():
		print(f"[WARN] Could not open: {vid_path}", file=sys.stderr)
		return 0
	bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_th, detectShadows=False)

	# Warmup background with learning
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
	warmup = min(warmup, max(0, total//5)) if total > 0 else warmup

	for _ in range(warmup):
		ret, frame = cap.read()
		if not ret: break
		bg.apply(frame, learningRate=0.5)

	# Reset to start
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	frame_idx = 0
	sm_bbox = None
	w_out, h_out = out_size
	saved = 0

	pbar_total = total if total > 0 else None
	pbar = tqdm(total=pbar_total, desc=Path(vid_path).name[:40], ncols=100)

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		fg = bg.apply(frame, learningRate=lr)
		sil, bbox = binarize_foreground(fg, kernel_size=kernel, min_area=min_area)
		if sil is not None and bbox is not None:
			if crop:
				sm_bbox = smooth_bbox(sm_bbox, bbox, alpha=0.2)
				cx, cy, cw, ch = pad_and_square_crop(frame, sm_bbox, pad_scale=pad)
				crop_sil = sil[cy:cy+ch, cx:cx+cw]
				if crop_sil.size == 0:
					res = sil
				else:
					res = cv2.resize(crop_sil, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
			else:
				res = cv2.resize(sil, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

			# Ensure binary 0/255 uint8
			res = (res > 0).astype(np.uint8) * 255
			cv2.imwrite(str(Path(out_dir) / f"frame_{frame_idx:06d}.png"), res)
			saved += 1

		frame_idx += 1
		if pbar_total:
			pbar.update(1)

	pbar.close()
	cap.release()
	return saved

def collect_videos(input_path):
	p = Path(input_path)
	if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
		return [str(p)]
	if p.is_dir():
		vids = []
		for ext in ("*.mp4","*.avi","*.mov","*.mkv","*.m4v"):
			vids.extend(glob.glob(str(p / "**" / ext), recursive=True))
		return sorted(vids)
	return []

def main():
	parser = argparse.ArgumentParser(description="Turn walking videos into CASIA-B style binary silhouettes.")
	parser.add_argument("--input", required=True, help="Path to a video file or a directory of videos.")
	parser.add_argument("--out", required=True, help="Output directory. Each video gets its own subfolder.")
	parser.add_argument("--size", default="64x128", help="Output size WxH (default: 64x128 to match CASIA-B).")
	parser.add_argument("--history", type=int, default=800, help="MOG2 background model history (default: 800).")
	parser.add_argument("--varth", type=float, default=16.0, help="MOG2 varThreshold (default: 16).")
	parser.add_argument("--warmup", type=int, default=150, help="Warmup frames to learn background (default: 150).")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate per frame (default: 0.001).")
	parser.add_argument("--kernel", type=int, default=5, help="Morph kernel size (odd, default: 5).")
	parser.add_argument("--minarea", type=int, default=2500, help="Min FG area to accept (default: 2500).")
	parser.add_argument("--pad", type=float, default=1.6, help="Crop padding scale around person (default: 1.6).")
	parser.add_argument("--no-crop", action="store_true", help="Disable subject-centered cropping.")
	args = parser.parse_args()

	out_size = parse_size(args.size)
	vids = collect_videos(args.input)
	if not vids:
		print("[ERR] No videos found.", file=sys.stderr)
		sys.exit(1)

	total_saved = 0
	for v in vids:
		name = Path(v).stem
		out_dir = Path(args.out) / name
		ensure_dir(out_dir)
		saved = process_video(
			v, str(out_dir), out_size=out_size,
			history=args.history, var_th=args.varth, warmup=args.warmup, lr=args.lr,
			kernel=args.kernel, min_area=args.minarea, pad=args.pad, crop=not args.no_crop
		)
		print(f"[OK] {name}: saved {saved} silhouette frames to {out_dir}")
		total_saved += saved

	if total_saved == 0:
		print("[WARN] No silhouettes saved. Try lowering --minarea, increasing --pad, or using --no-crop.", file=sys.stderr)

if __name__ == "__main__":
	main()