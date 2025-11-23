# Underwater Object Detection 🐠🌊

A comprehensive underwater object detection system using YOLOv8 for real-time identification of marine life, underwater debris, and other aquatic objects. This project combines computer vision, deep learning, and edge deployment capabilities for practical underwater monitoring applications.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.9.2-02569B?logo=flutter)](https://flutter.dev/)
[![Dart](https://img.shields.io/badge/Dart-3.9.2-0175C2?logo=dart)](https://dart.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)

## 🌟 Features

- **Real-time Object Detection**: Live detection using YOLOv8 models optimized for underwater environments
- **Multi-Dataset Support**: Training on diverse underwater datasets including fish species and marine debris
- **Mobile Deployment**: Flutter-based cross-platform mobile application for on-device inference
- **REST API**: FastAPI-based backend with endpoints for image, video, and real-time detection
- **Model Variants**: Multiple model sizes (Nano, Small, Medium) for different hardware constraints
- **Image Enhancement**: Specialized preprocessing for turbid water, low-light conditions, and color correction
- **Cross-Platform**: Works on Android, iOS, Windows, macOS, and Linux

## 📁 Project Structure

```
undwerwater-object-detection/
├── API/                          # FastAPI REST API server
│   ├── main.py                   # API endpoints and logic
│   └── requirements.txt          # Python dependencies
├── AppFiles/                     # Flutter mobile application
│   ├── main.dart                 # Flutter app main file
│   ├── pubspec.yaml             # Flutter dependencies
│   └── README.md                # App-specific documentation
├── Dataset1/                     # Primary underwater dataset
├── Dataset2/                     # Secondary underwater dataset  
├── Merged_Trash_Creature/        # Combined dataset for debris & marine life
├── Nano Model/                   # YOLOv8 Nano model implementation
│   └── README.md                # Detailed training guide
├── Final-Project/                # Production-ready implementation
├── Initial-Test/                 # Experimental and testing code
├── Test/                         # Test scripts and validation
├── Documentation/                # Project documentation and guides
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Flutter**: 3.9.2 or higher (for mobile app)
- **CUDA**: GPU with CUDA support (recommended, optional for CPU inference)
- **Hardware**: Webcam/camera for real-time detection
- **Memory**: 8GB+ RAM for training (4GB for inference only)

---

## 🔧 API Setup & Usage

### Installation

1. **Navigate to API directory**
   ```bash
   cd API
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (choose based on your system)
   ```bash
   # For CPU-only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Place your trained model**
   ```bash
   # Copy your best.pt model to the parent directory
   # or update MODEL_PATH in main.py
   cp path/to/your/best-50.pt ../
   ```

### Running the API Server

```bash
# From the API directory
python main.py

# Server will start on http://0.0.0.0:8000
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```
Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": 2
}
```

#### 2. Get Class Labels
```bash
curl http://localhost:8000/labels
```
Response:
```json
{
  "0": "fish",
  "1": "trash"
}
```

#### 3. Detect Objects in Image
```bash
curl -X POST "http://localhost:8000/detect/image?conf=0.5&iou=0.45" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@underwater_image.jpg"
```
Response:
```json
{
  "detections": [
    {
      "box": {"x1": 120.5, "y1": 80.3, "x2": 340.2, "y2": 280.7},
      "confidence": 0.87,
      "class_id": 0,
      "label": "fish",
      "is_fish": true
    }
  ],
  "num_fish": 1,
  "num_objects": 1,
  "conf_threshold": 0.5,
  "iou_threshold": 0.45
}
```

#### 4. Detect with Annotated Image
```bash
curl -X POST "http://localhost:8000/detect/image/annotated?conf=0.5&iou=0.45" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@underwater_image.jpg" \
  --output annotated_result.png
```
Returns: PNG image with bounding boxes drawn

#### 5. Detect Objects in Video
```bash
curl -X POST "http://localhost:8000/detect/video?conf=0.5&iou=0.45" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@underwater_video.mp4" \
  --output annotated_video.mp4
```
Returns: MP4 video with detections annotated on each frame

### API Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `conf` | float | 0.25 | 0.0-1.0 | Confidence threshold for detections |
| `iou` | float | 0.45 | 0.0-1.0 | IoU threshold for NMS |

### Python Client Example

```python
import requests

# Image detection
url = "http://localhost:8000/detect/image"
params = {"conf": 0.5, "iou": 0.45}
files = {"file": open("underwater_image.jpg", "rb")}

response = requests.post(url, params=params, files=files)
result = response.json()

print(f"Detected {result['num_fish']} fish")
print(f"Total objects: {result['num_objects']}")

for detection in result['detections']:
    print(f"{detection['label']}: {detection['confidence']:.2%}")
```

### API Configuration

Edit `API/main.py` to customize:

```python
# Model path
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'best-50.pt'))

# Default thresholds
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# CORS settings (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📱 Flutter Mobile App

### Prerequisites

1. **Install Flutter SDK**
   - Download from [flutter.dev](https://flutter.dev/docs/get-started/install)
   - Add Flutter to PATH
   - Verify installation: `flutter doctor`

2. **Setup IDE**
   - Android Studio (for Android development)
   - Xcode (for iOS development on macOS)
   - VS Code with Flutter extension

### Installation

1. **Navigate to app directory**
   ```bash
   cd AppFiles
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Configure API endpoint**
   
   Edit `main.dart` and update the API URL:
   ```dart
   class ApiConfig {
     // Update with your API server IP and port
     static const String baseUrl = 'http://192.168.1.100:8000';
     
     // For Android Emulator, use: 'http://10.0.2.2:8000'
     // For iOS Simulator on same machine: 'http://localhost:8000'
     // For physical device: Use your computer's local IP
   }
   ```

### Running the App

```bash
# Check connected devices
flutter devices

# Run on connected device/emulator
flutter run

# Run on specific device
flutter run -d <device-id>

# Run in release mode (optimized)
flutter run --release
```

### Platform-Specific Setup

#### Android
```bash
# Enable developer options and USB debugging on device
# Connect device via USB or start emulator
flutter run
```

#### iOS (macOS only)
```bash
# Open iOS simulator
open -a Simulator

# Run app
flutter run
```

#### Windows
```bash
# Build and run
flutter run -d windows
```

### App Features

The Flutter app provides three main modes:

#### 1. **Live Camera Detection**
- Real-time camera preview
- Capture and detect on-demand
- Auto-detection mode (every 3 seconds)
- Front/back camera switching
- Flash control
- Results displayed in side panel

#### 2. **Upload Image**
- Pick from gallery or camera
- Adjustable confidence threshold
- Displays detections with confidence scores
- Color-coded results (fish vs. other objects)

#### 3. **Upload Video**
- Select video files from device
- Upload to API for processing
- Download annotated video
- Open processed video directly

### App Dependencies

Key packages used (see `pubspec.yaml`):

```yaml
dependencies:
  camera: ^0.10.5+5           # Camera access
  http: ^1.2.0                # API communication
  image_picker: ^1.0.7        # Gallery/camera picker
  file_picker: ^8.0.0         # Video file picker
  video_player: ^2.8.6        # Video playback
  path_provider: ^2.1.2       # File storage
  image_gallery_saver: ^2.0.3 # Save to gallery
  open_filex: ^4.4.0          # Open files
```

### Building for Production

#### Android APK
```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

#### Android App Bundle (for Play Store)
```bash
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

#### iOS
```bash
flutter build ios --release
# Then open in Xcode to archive and upload
```

#### Windows
```bash
flutter build windows --release
# Output: build/windows/runner/Release/
```

### Troubleshooting App

**1. Cannot connect to API**
```
✗ Cannot connect to API
Check IP/port 8000 and firewall
```
**Solution:**
- Verify API server is running
- Check firewall allows port 8000
- For Android emulator, use `10.0.2.2` instead of `localhost`
- For physical device, use computer's local IP (e.g., `192.168.1.100`)
- Ensure both devices are on same network

**2. Camera not working**
```
Permission denied
```
**Solution:**
- Android: Add permissions to `AndroidManifest.xml`
- iOS: Add descriptions to `Info.plist`
- Grant camera permissions in device settings

**3. Build errors**
```bash
# Clean build and reinstall
flutter clean
flutter pub get
flutter run
```

---

## 🎓 Model Training

### Quick Start Training

Navigate to the Nano Model directory:
```bash
cd "Nano Model"
python fish_detection.py
```

Choose option 2 for training and follow the prompts.

For detailed training instructions, see [Nano Model/README.md](Nano%20Model/README.md)

### Basic Training Code

```python
from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.45,
    name='fish_detection'
)

# Best model saved at: runs/detect/fish_detection/weights/best.pt
```

## 📊 Datasets

### Dataset 1: Orange Chromide Fish
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/7w45jx35hd/1)
- **Images**: 586 total (409 train / 118 valid / 59 test)
- **Resolution**: 640×640 pixels
- **Format**: YOLO format annotations
- **Conditions**: Turbid water, high density, occlusion
- **Environment**: South Indian pond environments

### Dataset 2: Marine Debris & Creatures
- Combined dataset for multi-class underwater object detection
- Includes various fish species and underwater trash items
- Optimized for debris detection and classification

## 🎯 Model Performance

### YOLOv8 Nano Model
| Metric | Value |
|--------|-------|
| **Speed (CPU)** | 10-25 FPS |
| **Speed (GPU)** | 40+ FPS |
| **mAP@50** | ~85% |
| **Model Size** | ~6 MB |
| **Input Size** | 640×640 |
| **Parameters** | 3.01M |

### Detection Classes
- Fish species (Orange Chromide, etc.)
- Underwater debris/trash
- Marine creatures
- Custom trained classes

## 🛠️ Advanced Configuration

### Image Preprocessing

The system includes specialized underwater image enhancement:

```python
# CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

# Color correction for underwater scenes
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l = clahe.apply(l)
enhanced = cv2.merge([l, a, b])
frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

### Model Selection

Choose model based on your hardware:

```python
# Fastest - Best for mobile/edge devices
model = YOLO('yolov8n.pt')  # 6 MB

# Balanced - Good accuracy and speed
model = YOLO('yolov8s.pt')  # 22 MB

# Highest accuracy - Requires GPU
model = YOLO('yolov8m.pt')  # 52 MB
```

### Confidence Threshold Tuning

```python
# Adjust detection sensitivity
results = model.predict(
    source=image,
    conf=0.3,  # Lower = more detections (more false positives)
    iou=0.45   # IoU threshold for NMS
)
```

## 📖 Documentation

Detailed documentation available:
- **API Guide**: Complete REST API documentation above
- **Flutter App Guide**: Mobile app setup and usage above
- **Training Guide**: [Nano Model/README.md](Nano%20Model/README.md)
- **Dataset Preparation**: YOLOv8 format requirements
- **Model Optimization**: Quantization and pruning techniques
- **Deployment Strategies**: Edge devices and cloud deployment

## 🔧 Common Issues & Solutions

### API Issues

**1. Model file not found**
```
RuntimeError: Model file not found at /path/to/best-50.pt
```
**Solution:** Update `MODEL_PATH` in `main.py` or place model in correct location

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Use CPU-only PyTorch or reduce batch size

### App Issues

**3. API connection timeout**
**Solution:** 
- Increase timeout in Flutter app
- Check network connectivity
- Verify firewall settings

**4. Video processing slow**
**Solution:**
- Use smaller video files
- Reduce resolution before uploading
- Use GPU-enabled API server

## 🧪 Testing

### API Testing with cURL
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test detection
curl -X POST "http://localhost:8000/detect/image?conf=0.5" \
  -F "file=@test_image.jpg" \
  -o result.json
```

### API Testing with Swagger UI
Navigate to: `http://localhost:8000/docs`

Interactive API documentation with test interface.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📧 Contact

**Bodhdipta Roy**
- GitHub: [@bodhdipta-roy](https://github.com/bodhdipta-roy)
- Repository: [undwerwater-object-detection](https://github.com/bodhdipta-roy/undwerwater-object-detection)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API framework
- [Flutter](https://flutter.dev/) for cross-platform mobile development
- [OpenCV](https://opencv.org/) for image processing capabilities
- Dataset contributors and researchers in underwater computer vision
- South Indian pond fish monitoring research community

## 🔗 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flutter Documentation](https://docs.flutter.dev/)
- [Orange Chromide Dataset](https://data.mendeley.com/datasets/7w45jx35hd/1)
- Underwater image enhancement techniques for turbid water conditions
- Real-time object detection in challenging aquatic environments

---

## 🚀 Quick Command Reference

```bash
# API Server
cd API && pip install -r requirements.txt
python main.py  # Starts on port 8000

# Flutter App
cd AppFiles && flutter pub get
flutter run

# Training
cd "Nano Model"
python fish_detection.py

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/detect/image -F "file=@image.jpg"
```

---


**Keywords**: underwater object detection, YOLOv8, fish detection, marine debris, computer vision, deep learning, real-time detection, edge AI, mobile deployment, FastAPI, Flutter, cross-platform, REST API, computer vision API
