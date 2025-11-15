# Complete Guide: Underwater Fish Detection System

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Webcam/camera connected to your computer
- At least 4GB RAM (8GB recommended for training)
- GPU recommended but not required

### Check Python Version
```bash
python --version
# Should show Python 3.8+
```

## Step 1: Environment Setup

### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv fish_detection_env

# Activate
# Windows:
fish_detection_env\Scripts\activate
# macOS/Linux:
source fish_detection_env/bin/activate
```

### Install Required Packages
```bash
pip install ultralytics opencv-python torch torchvision
```

## Step 2: Get the Code

### Save the Code
1. Copy the fish detection code from the artifact above
2. Save it as `fish_detection.py` in your project folder

### File Structure Should Look Like:
```
your_project_folder/
└── fish_detection.py
```

## Step 3: Quick Test - Webcam Detection

### Run Basic Detection
```bash
python fish_detection.py
```

### What You'll See:
```
Underwater Fish Detection System
========================================
Dataset Structure Expected:
dataset_root/
├── train/
│   ├── images/ (409 .jpg files)
│   └── labels/ (.txt files)
├── valid/
│   ├── images/ (118 .jpg files)
│   └── labels/ (.txt files)
└── test/
    ├── images/ (59 .jpg files)
    └── labels/ (.txt files)
========================================

Options:
1. Run real-time detection with webcam
2. Train custom model (requires dataset)

Enter your choice (1-2):
```

### Choose Option 1
- Type `1` and press Enter
- Your webcam will start
- You'll see a window showing:
  - Live video feed at 640×640 resolution
  - FPS counter in top-left
  - Detection count
  - Bounding boxes around detected objects

### Webcam Controls:
- **'q'** - Quit the application
- **'s'** - Save current frame with timestamp

## Step 4: Download the Fish Dataset

### Get the Dataset
1. Go to: https://data.mendeley.com/datasets/7w45jx35hd/1
2. Click "Download" button
3. Extract the downloaded ZIP file

### Verify Dataset Structure
After extraction, you should have:
```
fish_dataset/
├── train/
│   ├── images/     # 409 .jpg files (640×640 pixels)
│   └── labels/     # 409 .txt files (YOLO format)
├── valid/
│   ├── images/     # 118 .jpg files
│   └── labels/     # 118 .txt files
└── test/
    ├── images/     # 59 .jpg files
    └── labels/     # 59 .txt files
```

## Step 5: Train Custom Fish Model

### Start Training
```bash
python fish_detection.py
```

### Choose Option 2
- Type `2` and press Enter
- Enter the path to your dataset folder (e.g., `C:\Users\YourName\fish_dataset`)

### Training Process:
```
Enter path to dataset directory: /path/to/fish_dataset
✓ Found train images: 409 files
✓ Found valid images: 118 files
✓ Found test images: 59 files
Enter number of epochs (default 100): 50
```

### What Happens During Training:
1. **Model Download**: Downloads YOLOv8 pretrained weights (~6MB)
2. **Dataset Validation**: Checks all images and labels
3. **Training Loop**: Shows progress, loss values, accuracy metrics
4. **Model Saving**: Saves best model automatically

### Training Output:
```
Training custom fish detection model...
Ultralytics YOLOv8.0.0 🚀 Python-3.9.0 torch-1.13.0 CUDA:0
Model summary: 225 layers, 3011433 parameters

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      1.2G      1.234      0.567      1.123         45        640
  2/50      1.2G      1.156      0.489      1.067         38        640
...
```

### Training Complete:
- Best model saved as `runs/detect/fish_detection/weights/best.pt`
- Training results in `runs/detect/fish_detection/`

## Step 6: Use Trained Model

### Update Code to Use Custom Model
After training, modify the detector initialization:

```python
# In the main() function, replace:
detector = UnderwaterFishDetector()

# With:
detector = UnderwaterFishDetector("runs/detect/fish_detection/weights/best_modeel-2.pt")
```

### Run with Custom Model
```bash
python fish_detection.py
```
- Choose option 1
- Now it uses your trained fish model!

## Understanding the Results

### Bounding Box Format
Each detection shows:
- **Green rectangle**: Detected fish boundary
- **Label**: "fish: 0.85" (class name + confidence score)
- **Coordinates**: Based on normalized YOLO format from dataset

### Performance Metrics
- **FPS**: Frames per second processing speed
- **Detections**: Number of fish detected in current frame
- **Confidence**: How certain the model is (0.0-1.0)

### Confidence Threshold
You can adjust detection sensitivity:
```python
# In __init__ method, modify:
self.confidence_threshold = 0.3  # Lower = more detections, higher = fewer but more certain
```

## Troubleshooting

### Common Issues and Solutions:

#### 1. Webcam Not Working
```
Error: Could not open webcam
```
**Solution:**
- Check if camera is connected
- Close other applications using camera
- Try different camera index:
```python
cap = cv2.VideoCapture(1)  # Try 1, 2, 3 instead of 0
```

#### 2. Package Installation Error
```
ERROR: Could not find a version that satisfies the requirement ultralytics
```
**Solution:**
```bash
pip install --upgrade pip
pip install ultralytics --no-cache-dir
```

#### 3. CUDA Out of Memory (During Training)
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size in training:
```python
results = model.train(
    batch=8,  # Reduce from 16 to 8 or 4
    # ... other parameters
)
```

#### 4. Dataset Path Error
```
❌ Dataset structure doesn't match expected format!
```
**Solution:**
- Verify folder structure matches exactly
- Check for hidden files or incorrect naming
- Ensure images are .jpg and labels are .txt

#### 5. Poor Detection Performance
**Solutions:**
- Lower confidence threshold (0.3-0.5)
- Train for more epochs (100-200)
- Use better lighting conditions
- Clean camera lens

### Advanced Options

#### Modify Image Enhancement
```python
# In preprocess_frame(), adjust parameters:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))  # Less aggressive
```

#### Change Model Size
```python
# For faster inference:
self.model = YOLO('yolov8n.pt')  # nano (fastest)

# For better accuracy:
self.model = YOLO('yolov8m.pt')  # medium (slower but more accurate)
```

#### Save Detection Results
```python
# Add to detect_fish method:
if detections:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"detections_{timestamp}.txt"
    with open(results_file, 'w') as f:
        for det in detections:
            f.write(f"{det['class']}: {det['confidence']:.3f} at {det['bbox']}\n")
```

## Expected Performance

### With Pretrained Model:
- **Speed**: 15-30 FPS on CPU, 60+ FPS on GPU
- **Accuracy**: Limited (not trained on fish)
- **Use**: Testing and development

### With Custom Fish Model:
- **Speed**: 10-25 FPS on CPU, 40+ FPS on GPU  
- **Accuracy**: High for Orange Chromide fish
- **Use**: Production deployment

This system is designed for the exact conditions in the research paper: turbid water, occlusion, varying lighting, and high fish density per frame in South Indian pond environments.
