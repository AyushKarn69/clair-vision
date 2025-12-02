"""
pick_best.py
ClairVision — Best Photo Selection script (Phase 3)

This script selects the "best" image from each duplicate group.

Changes in this revision:
- Removed unconditional sys.exit(0) calls that caused SystemExit in normal runs.
- When groups_dir is missing, the script creates it and either continues (if --create_sample) or
  runs main() which will print instructions and return (no SystemExit).
- run_tests no longer calls sys.exit(); it returns after completing tests.
- Added inline comments to explain complex parts.

Usage:
  python pick_best.py --groups_dir output/duplicates --dry_run
  python pick_best.py --groups_dir output/duplicates --create_sample
  python pick_best.py --run_tests

Requirements (minimal):
- Python 3.8+
- numpy, pillow, opencv-python, tqdm
- facenet-pytorch (optional) for better face detection

"""

import os
import argparse
import json
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import cv2
import sys
import random

# Optional MTCNN face detector; fallback to Haar cascade if not installed
USE_MTCNN = False
try:
    from facenet_pytorch import MTCNN
    import torch
    USE_MTCNN = True
except Exception:
    USE_MTCNN = False


# ---------------------- Utility functions ----------------------

def variance_of_laplacian_cv(img_cv):
    """Return variance of Laplacian (a sharpness metric) for a grayscale OpenCV image."""
    return float(cv2.Laplacian(img_cv, cv2.CV_64F).var())


def pil_to_cv2(img_pil: Image.Image):
    """Convert a PIL image to an OpenCV BGR numpy array."""
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def compute_basic_metrics(image_path):
    """Compute sharpness, area, and brightness score for an image.

    The brightness_score is 1.0 when mean brightness ~= 127.5 and decreases as it deviates.
    """
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    area = w * h

    cv_img = pil_to_cv2(img)
    if cv_img.size == 0:
        return {'sharpness': 0.0, 'area': 0, 'brightness_score': 0.0, 'width': w, 'height': h}

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian_cv(gray)

    mean_brightness = float(np.mean(gray))
    brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5

    return {
        'sharpness': float(sharpness),
        'area': int(area),
        'brightness_score': float(brightness_score),
        'width': int(w),
        'height': int(h)
    }


# ---------------------- Face detection helpers ----------------------

class FaceDetectorFallback:
    """Fallback face detector using OpenCV Haar cascade.

    If the cascade file is not available, detector is set to None and detect() returns [].
    """
    def __init__(self):
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_path):
            self.detector = None
            return
        self.detector = cv2.CascadeClassifier(haar_path)

    def detect(self, img_cv):
        if self.detector is None:
            return []
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        results = []
        for (x, y, w, h) in faces:
            results.append({'box': [x, y, x+w, y+h], 'confidence': 0.5})
        return results


class FaceAnalyzer:
    """Wrapper to detect faces and compute face-level quality metrics.

    Uses MTCNN if available; otherwise uses Haar fallback.
    """
    def __init__(self, device=None):
        self.device = device
        if USE_MTCNN:
            try:
                self.mtcnn = MTCNN(keep_all=True, device=device)
            except Exception:
                # if MTCNN fails to initialize, silently use fallback
                self.mtcnn = None
        else:
            self.fallback = FaceDetectorFallback()

    def detect_and_analyze(self, image_path):
        """Return (face_count, avg_face_sharpness, avg_face_area_ratio).

        - avg_face_sharpness: mean Laplacian variance over face crops
        - avg_face_area_ratio: mean(face_area / image_area)
        """
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img_cv = pil_to_cv2(img)
        if img_cv.size == 0:
            return 0, 0.0, 0.0
        h, w = img_cv.shape[:2]

        # Use MTCNN when available and initialized
        if USE_MTCNN and getattr(self, 'mtcnn', None) is not None:
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None:
                return 0, 0.0, 0.0
            sharp_vals = []
            area_ratios = []
            for box in boxes:
                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img_cv[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp_vals.append(variance_of_laplacian_cv(gray))
                area_ratios.append((crop.shape[0] * crop.shape[1]) / (w * h))
            if len(sharp_vals) == 0:
                return 0, 0.0, 0.0
            return len(sharp_vals), float(np.mean(sharp_vals)), float(np.mean(area_ratios))

        # Use Haar cascade fallback
        faces = self.fallback.detect(img_cv)
        if len(faces) == 0:
            return 0, 0.0, 0.0
        sharp_vals = []
        area_ratios = []
        for f in faces:
            x1, y1, x2, y2 = f['box']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp_vals.append(variance_of_laplacian_cv(gray))
            area_ratios.append((crop.shape[0] * crop.shape[1]) / (w * h))
        if len(sharp_vals) == 0:
            return 0, 0.0, 0.0
        return len(sharp_vals), float(np.mean(sharp_vals)), float(np.mean(area_ratios))


# ---------------------- Normalization helper ----------------------

def normalize_array(arr):
    """Min-max normalize a 1D iterable to [0,1]. Returns numpy array."""
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    minv = arr.min()
    maxv = arr.max()
    if maxv - minv < 1e-9:
        return np.ones_like(arr)
    return (arr - minv) / (maxv - minv)


# ---------------------- Core: process a single group ----------------------

def process_group(group_folder: Path, face_analyzer: FaceAnalyzer, weights=None, dry_run=True, move_best=True, out_best=None, out_rest=None):
    """Pick the best image from a group folder and optionally move files.

    Returns the selected best image path (string) or None if folder empty.
    """
    img_paths = [p for p in sorted(group_folder.iterdir()) if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if len(img_paths) == 0:
        return None

    # compute per-image basic metrics
    metrics = {}
    for p in img_paths:
        metrics[str(p)] = compute_basic_metrics(str(p))

    # compute face-level metrics
    face_counts = []
    face_sharpness = []
    face_area_ratios = []
    for p in img_paths:
        c, fs, far = face_analyzer.detect_and_analyze(str(p))
        face_counts.append(c)
        face_sharpness.append(fs)
        face_area_ratios.append(far)
        metrics[str(p)]['face_count'] = c
        metrics[str(p)]['face_sharpness'] = fs
        metrics[str(p)]['face_area_ratio'] = far

    # normalize features so they can be combined
    sharp_arr = normalize_array([metrics[str(p)]['sharpness'] for p in img_paths])
    area_arr = normalize_array([metrics[str(p)]['area'] for p in img_paths])
    bright_arr = normalize_array([metrics[str(p)]['brightness_score'] for p in img_paths])
    face_sharp_arr = normalize_array(face_sharpness)
    face_area_arr = normalize_array(face_area_ratios)

    face_quality = 0.5 * face_sharp_arr + 0.5 * face_area_arr

    if weights is None:
        weights = {'sharpness': 0.35, 'face': 0.40, 'aesthetic': 0.25}

    scores = []
    for i, p in enumerate(img_paths):
        if face_counts[i] > 0:
            score = (weights['sharpness'] * sharp_arr[i] +
                     weights['face'] * face_quality[i] +
                     weights['aesthetic'] * bright_arr[i])
        else:
            # redistribute face weight when absent
            w_sharp = weights['sharpness'] + weights['face'] * 0.6
            w_aes = weights['aesthetic'] + weights['face'] * 0.4
            score = w_sharp * sharp_arr[i] + w_aes * bright_arr[i]
        scores.append(score)
        metrics[str(p)]['composite_score'] = float(score)

    best_idx = int(np.argmax(scores))
    best_path = img_paths[best_idx]

    # move files if not dry run
    if not dry_run:
        out_best = Path(out_best) if out_best else Path('output/selected_best')
        out_rest = Path(out_rest) if out_rest else Path('output/duplicates_kept')
        out_best.mkdir(parents=True, exist_ok=True)
        out_rest.mkdir(parents=True, exist_ok=True)

        if move_best:
            shutil.move(str(best_path), str(out_best / best_path.name))
        for i, p in enumerate(img_paths):
            if i == best_idx:
                continue
            dest = out_rest / p.name
            shutil.move(str(p), str(dest))

    return str(best_path)


# ---------------------- Main logic ----------------------

def main(args):
    groups_dir = Path(args.groups_dir)
    out_best = args.out_best
    out_rest = args.out_rest
    dry_run = args.dry_run

    # If the directory doesn't exist, create it and notify the user.
    if not groups_dir.exists():
        print(f"Note: groups directory {groups_dir} does not exist. Creating it now.")
        groups_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {groups_dir}. Add subfolders (group_001, group_002, ...) with images and re-run the script, or run with --create_sample to generate demo data.")
        return

    group_folders = [p for p in sorted(groups_dir.iterdir()) if p.is_dir()]

    if len(group_folders) == 0:
        print(f"No group subfolders found inside {groups_dir}.")
        print("Create subfolders (one per duplicate group) with image files, or run: --create_sample to auto-generate a demo dataset.")
        return

    # initialize face analyzer
    face_analyzer = FaceAnalyzer(device='cuda' if (USE_MTCNN and torch.cuda.is_available()) else 'cpu')

    print(f"Processing {len(group_folders)} groups from {groups_dir}")

    results = []
    for g in tqdm(group_folders, desc='Groups'):
        selected = process_group(g, face_analyzer, dry_run=dry_run, move_best=not dry_run, out_best=out_best, out_rest=out_rest)
        results.append((str(g), selected))

    # write summary JSON
    summary = []
    print('\nSummary:')
    for grp, sel in results:
        print(f"Group: {grp}  -> Selected: {sel}")
        summary.append({'group': grp, 'selected': sel})

    out_summary = Path('output/pick_best_summary.json')
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {out_summary}")


# ---------------------- Sample dataset utility ----------------------

def create_sample_groups(base_dir: Path, groups=3, imgs_per_group=3):
    base_dir.mkdir(parents=True, exist_ok=True)
    colors = [(200,30,30), (30,200,30), (30,30,200), (200,200,30), (200,30,200)]
    for gi in range(1, groups+1):
        gdir = base_dir / f'group_{gi:03d}'
        gdir.mkdir(parents=True, exist_ok=True)
        for ii in range(1, imgs_per_group+1):
            w, h = random.choice([(800,600),(1024,768),(640,480)])
            img = Image.new('RGB', (w, h), color=random.choice(colors))
            draw = ImageDraw.Draw(img)
            draw.text((10,10), f'Group {gi} Img {ii}', fill=(255,255,255))
            if ii == 2:
                img = img.filter(ImageFilter.GaussianBlur(radius=5))
            path = gdir / f'img_{ii}.jpg'
            img.save(path, quality=90)


# ---------------------- Internal tests ----------------------

def run_internal_tests(tmp_root: Path):
    print('Running internal tests...')
    sample_dir = tmp_root / 'sample_duplicates'
    create_sample_groups(sample_dir, groups=2, imgs_per_group=3)
    class Args:
        groups_dir = str(sample_dir)
        out_best = 'test_out_best'
        out_rest = 'test_out_rest'
        dry_run = True
        create_sample = False
        run_tests = False
    args = Args()
    main(args)
    out_summary = Path('output/pick_best_summary.json')
    if not out_summary.exists():
        raise AssertionError('Internal test failed: summary file not written')
    print('Internal tests finished — summary generated.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pick Best Photo from Duplicate Groups")

    parser.add_argument('--groups_dir', type=str, default="duplicates",
                        help='Path to duplicate groups folder')
    parser.add_argument('--out_best', type=str, default="selected_best",
                        help='Where best photos will be stored')
    parser.add_argument('--out_rest', type=str, default="others",
                        help='Where non-selected images will go')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run without moving files')
    parser.add_argument('--create_sample', action='store_true',
                        help='Generate a small test dataset')
    parser.add_argument('--run_tests', action='store_true',
                        help='Run internal self-checks')

    args = parser.parse_args()

    if args.create_sample:
        print("\n📌 Creating sample dataset...")
        create_sample_groups(Path(args.groups_dir))
        print("✅ Sample data created.\n")

    elif args.run_tests:
        run_internal_tests(Path("tmp_test"))

    else:
        main(args)
