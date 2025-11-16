"""
Underwater Creature Detection Model Training Script
Optimized for GTX 1650 Super (4GB VRAM) + i5 10th Gen
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import torch

# Configuration
class TrainingConfig:
    """
    GPU-Optimized Training Configuration
    GTX 1650 Super: 4GB VRAM
    CPU: i5 10th Gen
    """

    # ============ MODEL SELECTION ============
    # For GTX 1650 Super (4GB VRAM):
    # 'yolov8n.pt' - Nano: 6MB, fastest, fits easily ← For maximum speed
    # 'yolov8s.pt' - Small: 22MB, balanced ← RECOMMENDED for 4GB VRAM
    # 'yolov8m.pt' - Medium: 52MB, accurate ← Possible but tight
    # 'yolov8l.pt' - Large: 87MB ← Will run out of VRAM

    MODEL_SIZE = 'yolov8n.pt'  # Perfect for GTX 1650 Super

    # ============ DATASET PATH ============
    DATASET_PATH = r'D:\Projects\UnderWaterObjectDetection\Dataset-2'

    # ============ GPU SETTINGS ============
    # GTX 1650 Super optimized settings
    EPOCHS = 30              # GPU trains much faster, so more epochs
    IMG_SIZE = 640            # Standard size (GTX 1650 Super can handle it)
    BATCH_SIZE = 8            # Optimized for 4GB VRAM
    DEVICE = 0                # Use GPU (0 = first GPU)
    WORKERS = 4               # i5 10th gen has 6 cores, use 4 for data loading

    # ============ MEMORY OPTIMIZATION ============
    # These settings prevent OOM (Out of Memory) on 4GB GPU
    CACHE = False             # Don't cache images in RAM (saves VRAM)
    AMP = True                # Automatic Mixed Precision (uses less VRAM)

    # ============ TRAINING PARAMETERS ============
    PATIENCE = 30             # Stop if no improvement for 30 epochs
    SAVE_PERIOD = 10          # Save checkpoint every 10 epochs

    # Augmentation (optimized for underwater)
    AUGMENT = True
    MOSAIC = 1.0              # Mosaic augmentation
    MIXUP = 0.0               # Disabled for speed and VRAM
    DEGREES = 10.0            # Rotation
    TRANSLATE = 0.2           # Translation
    SCALE = 0.5               # Scaling
    FLIPUD = 0.5              # Flip up-down
    FLIPLR = 0.5              # Flip left-right
    HSV_H = 0.015             # Hue (underwater lighting)
    HSV_S = 0.7               # Saturation (underwater color)
    HSV_V = 0.4               # Brightness

    # Close mosaic augmentation in last N epochs (improves final accuracy)
    CLOSE_MOSAIC = 10


def print_gpu_info():
    """Display GPU information"""
    print("\n" + "="*70)
    print("           GPU INFORMATION")
    print("="*70)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n✅ GPU Detected: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.2f} GB")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        print(f"🐍 PyTorch Version: {torch.__version__}")

        # Check if it's GTX 1650 Super
        if "1650" in gpu_name:
            print(f"\n✅ GTX 1650 Super detected!")
            print(f"💡 Optimized settings applied for 4GB VRAM")

        # Memory recommendation
        if gpu_memory < 4:
            print(f"\n⚠️  Low VRAM detected!")
            print(f"   Recommended: Reduce batch size to 4")
        elif gpu_memory < 6:
            print(f"\n✅ Good VRAM for YOLOv8s")
            print(f"   Batch size 8 is optimal")
        else:
            print(f"\n✅ Plenty of VRAM!")
            print(f"   Can increase batch size to 16")

    else:
        print("\n❌ No GPU detected!")
        print("   Training will use CPU (very slow)")
        print("\n💡 Make sure:")
        print("   1. NVIDIA GPU drivers are installed")
        print("   2. CUDA toolkit is installed")
        print("   3. PyTorch with CUDA is installed")

    print("="*70)


def print_system_info():
    """Display system information"""
    print("\n" + "="*70)
    print("           SYSTEM INFORMATION")
    print("="*70)

    import platform
    import psutil

    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"⚡ CPU: {platform.processor()}")
    print(f"⚡ CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"💾 RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")

    print("="*70)


def print_training_time_estimate(config):
    """Estimate training time on GPU"""
    print("\n" + "="*70)
    print("           TRAINING TIME ESTIMATE (GTX 1650 Super)")
    print("="*70)

    # Time estimates for GTX 1650 Super
    time_per_epoch = {
        'yolov8n.pt': 1.5,   # ~1.5 min/epoch
        'yolov8s.pt': 2.5,   # ~2.5 min/epoch
        'yolov8m.pt': 4.5,   # ~4.5 min/epoch
    }

    model_name = config.MODEL_SIZE
    time = time_per_epoch.get(model_name, 2.5)

    total_minutes = time * config.EPOCHS
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)

    print(f"\n📊 Model: {model_name}")
    print(f"⏱️  Time per epoch: ~{time:.1f} minutes")
    print(f"📈 Total epochs: {config.EPOCHS}")
    print(f"⏰ Estimated total time: {hours}h {minutes}m")

    print("\n💡 Comparison:")
    print(f"   GPU (GTX 1650 Super): ~{hours}h {minutes}m")
    print(f"   CPU (i5 10th gen):    ~25-30 hours")
    print(f"   Speed improvement:    ~10-12x faster! 🚀")

    print("="*70)


def detect_classes_from_labels(labels_path):
    """Automatically detect class IDs from label files"""
    print(f"\n🔍 Analyzing labels in: {labels_path}")

    class_ids = set()
    label_files = list(Path(labels_path).glob('*.txt'))

    if not label_files:
        print(f"⚠️  Warning: No label files found")
        return []

    # Sample labels to detect classes
    sample_size = min(100, len(label_files))
    for label_file in label_files[:sample_size]:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:  # Valid box annotation
                        class_ids.add(int(parts[0]))
        except Exception as e:
            continue

    print(f"✓ Found {len(class_ids)} unique classes: {sorted(class_ids)}")
    return sorted(class_ids)


def verify_dataset_structure(dataset_path):
    """Verify and display dataset structure"""
    dataset_path = Path(dataset_path).absolute()

    print("\n" + "="*70)
    print("           DATASET VERIFICATION")
    print("="*70)

    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")

    print(f"\n📁 Dataset: {dataset_path}")

    # Check folders
    train_images = dataset_path / 'train' / 'images'
    train_labels = dataset_path / 'train' / 'labels'
    valid_images = dataset_path / 'valid' / 'images'
    valid_labels = dataset_path / 'valid' / 'labels'

    # Count files
    train_img = len(list(train_images.glob('*.[jp][pn]g'))) if train_images.exists() else 0
    train_lbl = len(list(train_labels.glob('*.txt'))) if train_labels.exists() else 0
    valid_img = len(list(valid_images.glob('*.[jp][pn]g'))) if valid_images.exists() else 0
    valid_lbl = len(list(valid_labels.glob('*.txt'))) if valid_labels.exists() else 0

    print(f"\n📊 Dataset Statistics:")
    print(f"   Train: {train_img} images, {train_lbl} labels")
    print(f"   Valid: {valid_img} images, {valid_lbl} labels")
    print(f"   Total: {train_img + valid_img} images")

    # Check data.yaml
    yaml_file = dataset_path / 'data.yaml'
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if 'nc' in data and 'names' in data:
                print(f"\n🏷️  Classes ({data['nc']} total):")
                for i, name in enumerate(data['names']):
                    print(f"   {i}: {name}")
                return True

    return False


def create_dataset_yaml(dataset_path, output_file='underwater_data.yaml'):
    """Create YAML configuration for dataset"""

    dataset_path = Path(dataset_path).absolute()

    # Check for existing data.yaml
    existing_yaml = dataset_path / 'data.yaml'
    if existing_yaml.exists():
        print(f"\n✅ Found existing data.yaml")
        try:
            with open(existing_yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Update paths
            yaml_data['path'] = str(dataset_path)
            yaml_data['train'] = 'train/images'
            yaml_data['val'] = 'valid/images'

            # Save updated version
            with open(output_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            print(f"✓ Classes: {yaml_data.get('names', [])}")
            return str(Path(output_file).absolute())
        except Exception as e:
            print(f"⚠️  Error reading data.yaml: {e}")

    # Auto-detect classes
    print("\n⚠️  Creating data.yaml from labels...")

    train_labels = dataset_path / 'train' / 'labels'
    class_ids = detect_classes_from_labels(train_labels)

    if not class_ids:
        raise ValueError("❌ Could not detect classes!")

    num_classes = max(class_ids) + 1

    # Default class names
    default_names = [
        'fish', 'jellyfish', 'penguin', 'puffin', 'shark',
        'starfish', 'stingray', 'octopus', 'turtle', 'crab'
    ]

    class_names = default_names[:num_classes]

    # Create YAML
    data_yaml = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': num_classes,
        'names': class_names
    }

    with open(output_file, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Created: {output_file}")
    print(f"✓ Classes: {class_names}")

    return str(Path(output_file).absolute())


def train_model(config):
    """Train the YOLOv8 model on GPU"""

    print("\n" + "="*70)
    print("       UNDERWATER OBJECT DETECTION - GPU TRAINING")
    print("="*70)

    # System info
    print_system_info()
    print_gpu_info()

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n❌ ERROR: No GPU detected!")
        print("\n💡 Solutions:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA toolkit")
        print("   3. Reinstall PyTorch with CUDA:")
        print("      pip uninstall torch torchvision")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return None

    # Verify dataset
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {config.DATASET_PATH}")

    verify_dataset_structure(config.DATASET_PATH)

    # Create/load YAML
    data_yaml = create_dataset_yaml(config.DATASET_PATH)

    # Training time estimate
    print_training_time_estimate(config)

    # Configuration summary
    print("\n" + "="*70)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*70)
    print(f"   Model:      {config.MODEL_SIZE}")
    print(f"   Device:     GPU (CUDA:{config.DEVICE})")
    print(f"   Epochs:     {config.EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Image Size: {config.IMG_SIZE}")
    print(f"   Workers:    {config.WORKERS}")
    print(f"   Mixed Precision: {'Enabled' if config.AMP else 'Disabled'}")
    print("="*70)

    # Load model
    print(f"\n📦 Loading {config.MODEL_SIZE}...")
    model = YOLO(config.MODEL_SIZE)

    # Confirm start
    print("\n🚀 Ready to start GPU training!")
    print("💡 This will take approximately 2-4 hours")
    print("💡 You can monitor GPU usage with: nvidia-smi")
    print("\n" + "="*70)
    input("Press ENTER to start training...")

    print("\n🚀 Starting training...\n")

    try:
        # Train with GPU-optimized settings
        results = model.train(
            data=data_yaml,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            device=config.DEVICE,
            workers=config.WORKERS,
            patience=config.PATIENCE,
            save_period=config.SAVE_PERIOD,
            cache=config.CACHE,

            # Augmentation
            augment=config.AUGMENT,
            mosaic=config.MOSAIC,
            mixup=config.MIXUP,
            degrees=config.DEGREES,
            translate=config.TRANSLATE,
            scale=config.SCALE,
            flipud=config.FLIPUD,
            fliplr=config.FLIPLR,
            hsv_h=config.HSV_H,
            hsv_s=config.HSV_S,
            hsv_v=config.HSV_V,
            close_mosaic=config.CLOSE_MOSAIC,

            # Optimization
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            val=True,
            plots=True,
            save=True,
            amp=config.AMP,  # Mixed precision for 4GB VRAM

            # Project
            project='underwater_detection',
            name='training_run',
            exist_ok=True,
        )

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)

        # Validate
        print("\n🔍 Validating model...")
        metrics = model.val()

        print("\n📊 FINAL RESULTS:")
        print("="*70)
        print(f"   mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
        print(f"   mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
        print(f"   Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)")
        print(f"   Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)")

        # Performance rating
        if metrics.box.map50 > 0.80:
            print("\n🎉 Excellent! Top-tier accuracy!")
        elif metrics.box.map50 > 0.70:
            print("\n✅ Great! Model is production-ready!")
        elif metrics.box.map50 > 0.60:
            print("\n⚠️  Good. Consider training longer for better results.")
        else:
            print("\n❌ Needs improvement. Check dataset or train longer.")

        # Model location
        best_model = Path('underwater_detection/training_run/weights/best.pt').absolute()
        print("\n📁 MODEL SAVED:")
        print(f"   {best_model}")
        print("\n💡 Ready for deployment!")
        print("="*70)

        return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU OUT OF MEMORY!")
            print("\n💡 Solutions:")
            print("   1. Reduce batch size:")
            print("      BATCH_SIZE = 4  (current: 8)")
            print("   2. Use smaller model:")
            print("      MODEL_SIZE = 'yolov8n.pt'")
            print("   3. Reduce image size:")
            print("      IMG_SIZE = 512  (current: 640)")
            print("\n   Edit these in the script and try again.")
        else:
            print(f"\n❌ Error: {e}")
        return None

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Checkpoint saved in: underwater_detection/training_run/weights/")
        return None


if __name__ == "__main__":
    # Initialize configuration
    config = TrainingConfig()

    print("\n" + "="*70)
    print("   UNDERWATER OBJECT DETECTION - GPU ACCELERATED")
    print("   Optimized for GTX 1650 Super (4GB) + i5 10th Gen")
    print("="*70)

    # Train
    model = train_model(config)

    if model:
        print("\n" + "="*70)
        print("🎉 SUCCESS! GPU Training Complete!")
        print("="*70)
        print("\n📋 Next steps:")
        print("   1. Check results: underwater_detection/training_run/")
        print("   2. Test model: python underwater_webcam_detection.py")
        print("   3. Model file: underwater_detection/training_run/weights/best.pt")
        print("\n🎯 Your model is ready for real-time detection!")
        print("="*70 + "\n")