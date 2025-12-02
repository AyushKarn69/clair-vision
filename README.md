# ClairVision v0.11 - Image Quality & Duplicate Detection Suite

A comprehensive computer vision toolkit for image quality assessment, blur detection, and duplicate image identification using OpenCV, PyTorch, and deep learning models.

## Project Overview

ClairVision provides multiple tools for image processing and analysis:

1. **Blur Detection** - Identify blurry vs. sharp images using Laplacian variance and Resnet50
2. **Duplicate Detection** - Find and group similar/duplicate images using CLIP embeddings
3. **Image Quality Assessment** - Evaluate image quality using neural networks

## Project Structure

```
clairvisionV0.11/
├── BlurDetection2/              # Blur detection module
│   ├── blur_detection/
│   │   ├── detection.py         # Core blur detection algorithm
│   │   ├── best_blur_detector.pth  # Pre-trained blur detection model
│   │   ├── resenet50.ipynb      # ResNet50 model notebook
│   │   └── images_test/         # Test images for blur detection
│   ├── process.py               # Main blur detection processing script
│   ├── sort_blur.py             # Script to sort images into sharp/blurry folders
│   └── results.json             # Output file with blur detection results
│
├── model3/                      # Duplicate detection module
│   ├── duplicate_detection.py   # CLIP-based duplicate detection
│   ├── dup_detc_adv.py          # Advanced duplicate detection (commented - future enhancement)
│   ├── input_images/            # Input images for duplicate detection
│   ├── duplicates_sorted/       # Output: grouped duplicates
│   ├── duplicateeeeeeee_sorted/ # Output: alternative grouping results
│   └── results/                 # Duplicate detection results
│
├── best_pick/                   # Additional utilities
│   ├── scripts/                 # Processing scripts
│   └── face_detect/             # Face detection utilities
│
├── .gitignore                   # Git ignore rules
├── .venv/                       # Virtual environment (excluded from git)
└── README.md                    # This file
```

## Features

### 1. Blur Detection (BlurDetection2)

**Algorithm:** Laplacian variance-based blur detection

- **Method:** Computes variance of Laplacian edges in image
- **Default Threshold:** 100 (configurable)
- **Optimal Threshold (for reference dataset):** 207.95
- **Input:** Single images or directory of images
- **Output:** JSON file with blur scores and classification

**Key Files:**
- `detection.py`: Contains `estimate_blur()` function using OpenCV
- `process.py`: Batch processing script with CLI arguments
- `sort_blur.py`: Organizes results into sharp/blurry directories

**Usage:**

```bash
# Basic blur detection
python process.py -i path/to/images -s results.json

# With custom threshold
python process.py -i path/to/images -t 207.95 -s results.json

# With verbose output
python process.py -i path/to/images -t 207.95 -v

# Sort results into directories
python sort_blur.py
```

**Output Structure:**
```json
{
  "images": ["path/to/images"],
  "threshold": 207.95,
  "fix_size": true,
  "results": [
    {
      "input_path": "image.jpg",
      "score": 250.5,
      "blurry": false
    }
  ]
}
```

### 2. Duplicate Detection (model3)

**Algorithm:** CLIP-based semantic similarity with cosine distance

- **Model:** Vision Transformer (ViT-B/32) via OpenCV CLIP
- **Similarity Threshold:** 0.93 (configurable)
- **Method:** Embeddings extraction + cosine similarity matrix
- **Input:** Directory of images
- **Output:** Grouped duplicates in separate folders

**Key Files:**
- `duplicate_detection.py`: Main duplicate detection using CLIP
- `dup_detc_adv.py`: Advanced multi-model approach (hybrid CLIP + EfficientNet - commented)

**Usage:**

```bash
# Run duplicate detection
python duplicate_detection.py
# Results saved to: duplicateeeeeeee_sorted/
```

**Configuration (in duplicate_detection.py):**
```python
input_folder = "input_images"           # Input image directory
output_folder = "duplicateeeeeeee_sorted"
similarity_threshold = 0.93             # 0.9-0.95 recommended
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 3. Image Quality Assessment

Neural network-based image quality evaluation (NIMA-style models)
- Multiple architecture support (InceptionResNet, MobileNet, NASNet)
- Feature extraction and scoring pipeline
- Training and evaluation scripts available

## Requirements

### Dependencies

```
opencv-python
numpy
torch
torchvision
clip
scikit-learn
pillow
tqdm
```

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Threshold Optimization Guide

### Blur Detection Threshold

The optimal threshold depends on your dataset characteristics:

**Reference Dataset Analysis (700 images):**
- Score Range: 14.77 - 9300.36
- Mean: 430.24
- Median: 207.06
- Standard Deviation: 718.83
- **Percentiles:**
  - 10th: 58.69
  - 25th: 104.94
  - 50th: 207.06
  - 75th: 466.62
  - 90th: 1038.21

**Threshold Selection:**
- For ~50% sharp/blur split: Use **207.95** (median)
- For stricter quality: Increase threshold (e.g., 400+)
- For more lenient: Decrease threshold (e.g., 100-150)

**Formula:** Images with score ≥ threshold = SHARP; score < threshold = BLURRY

## Dataset Processing Workflow

### Step 1: Process Images for Blur
```bash
python BlurDetection2/process.py -i input_folder -t 207.95 -s results.json
```

### Step 2: Sort by Blur Classification
```bash
python BlurDetection2/sort_blur.py
# Output: sorted_output/sharp/ and sorted_output/blurry/
```

### Step 3: Detect Duplicates (Optional)
```bash
# Copy sharp images to model3/input_images/
python model3/duplicate_detection.py
# Output: duplicates grouped in duplicateeeeeeee_sorted/
```

## Model Files

### Pre-trained Models

| File | Size | Purpose | Location |
|------|------|---------|----------|
| `resenet50.ipynb` | ~50KB | ResNet50 model weights notebook | BlurDetection2/blur_detection/ |
| `best_blur_detector.pth` | ~244MB | Pre-trained blur detection model | BlurDetection2/blur_detection/ |

**Note:** Model weight files (*.h5, *.pth, *.pt) are excluded from git except ResNet50 weights (tracked via .gitignore exception).

## Performance Notes

### Blur Detection
- **Speed:** ~10-50ms per image (CPU)
- **Speed:** ~5-20ms per image (GPU)
- **Memory:** Minimal (~50MB)

### Duplicate Detection
- **Speed:** Highly dependent on number of images and GPU availability
- **Bottleneck:** CLIP embedding extraction (~100-200ms per image)
- **Memory:** ~2GB for 1000 images (GPU)

### Optimization Tips
1. Use `--variable-size` flag to skip image normalization (faster but less consistent scoring)
2. Enable GPU for duplicate detection (30-50x speedup)
3. Process images in batches for better resource utilization

## Example Results

### Blur Detection Output (700 images)
```
Processing complete:
- Sharp images: 179 (25.6%)
- Blurry images: 176 (25.1%)
- Total processed: 355 images
- Threshold used: 207.95
- Score range: 14.77 - 9300.36
```

### Duplicate Detection Output
```
Processing complete:
- Total images processed: 350
- Duplicate groups found: 12
- Largest group: 5 images
- Average group size: 2.3 images
```

## Advanced Configuration

### Custom Blur Detection Function

```python
from blur_detection.detection import estimate_blur
import cv2

image = cv2.imread('test.jpg')
blur_map, score, is_blurry = estimate_blur(image, threshold=207.95)

print(f"Blur score: {score}, Blurry: {is_blurry}")
```

### Batch Processing with Custom Thresholds

```python
import json
from pathlib import Path

thresholds = [100, 150, 200, 250, 300]

for threshold in thresholds:
    os.system(f'python process.py -i images/ -t {threshold} -s results_{threshold}.json')
```

## Troubleshooting

### Common Issues

**"CUDA out of memory" during duplicate detection**
- Reduce batch size or process fewer images at once
- Use CPU mode: Set `device = "cpu"` in duplicate_detection.py

**"No module named 'clip'"**
- Install: `pip install git+https://github.com/openai/CLIP.git`

**Low blur detection accuracy**
- Dataset-specific thresholds needed
- Run analyze_threshold.py to find optimal threshold for your data

**Images not being copied in sort_blur.py**
- Verify image paths in results.json are correct
- Check file permissions and directory existence

## Future Enhancements

1. Advanced duplicate detection combining CLIP + EfficientNet (see dup_detc_adv.py)
2. Face detection and quality assessment
3. Multi-model ensemble for blur detection
4. Real-time video stream processing
5. Web API interface for remote processing


