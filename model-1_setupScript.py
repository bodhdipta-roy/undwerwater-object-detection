"""
Underwater Creature Detection Model Training Script
This script trains a YOLOv8 model for detecting various underwater creatures
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import torch

# Configuration
class TrainingConfig:
    # Model selection - choose based on your needs:
    # 'yolov8n.pt' - Nano (fastest, least accurate)
    # 'yolov8s.pt' - Small (fast, good balance)
    # 'yolov8m.pt' - Medium (balanced)
    # 'yolov8l.pt' - Large (slower, more accurate)
    # 'yolov8x.pt' - Extra Large (slowest, most accurate)
    MODEL_SIZE = 'yolov8l.pt'

    # Training parameters
    EPOCHS = 10  # Increase for better results (100-300 recommended)
    IMG_SIZE = 640  # Image size for training
    BATCH_SIZE = 16  # Adjust based on your GPU memory

    # Dataset path - YOUR EXACT PATH
    DATASET_PATH = r'D:\Projects\UnderWaterObjectDetection\Dataset-1.0'

    # Training settings
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    WORKERS = 8  # Number of workers for data loading

    # Advanced training parameters
    PATIENCE = 50  # Early stopping patience
    SAVE_PERIOD = 10  # Save checkpoint every N epochs

    # Augmentation (helps with varying underwater conditions)
    AUGMENT = True
    MOSAIC = 1.0  # Mosaic augmentation probability
    MIXUP = 0.1  # Mixup augmentation probability
    DEGREES = 10.0  # Rotation
    TRANSLATE = 0.2  # Translation
    SCALE = 0.9  # Scale
    FLIPUD = 0.5  # Flip up-down probability
    FLIPLR = 0.5  # Flip left-right probability
    HSV_H = 0.015  # HSV-Hue augmentation (important for underwater)
    HSV_S = 0.7  # HSV-Saturation (important for underwater)
    HSV_V = 0.4  # HSV-Value (important for underwater)


def detect_classes_from_labels(labels_path):
    """
    Automatically detect class names from label files
    """
    print(f"\n🔍 Analyzing labels in: {labels_path}")

    class_ids = set()
    label_files = list(Path(labels_path).glob('*.txt'))

    if not label_files:
        print(f"⚠️  Warning: No label files found in {labels_path}")
        return []

    # Read all label files to find unique class IDs
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
        except Exception as e:
            print(f"⚠️  Error reading {label_file}: {e}")

    num_classes = len(class_ids)
    print(f"✓ Found {num_classes} unique classes: {sorted(class_ids)}")

    return sorted(class_ids)


def create_dataset_yaml(dataset_path, output_file='underwater_data.yaml'):
    """
    Create YAML configuration file for the dataset

    Your dataset structure:
    Dataset-1/
    ├── test/
    │   ├── images/
    │   └── labelTxt/
    ├── train/
    │   ├── images/
    │   └── labelTxt/
    ├── valid/
    │   ├── images/
    │   └── labelTxt/
    └── data.yaml
    """

    dataset_path = Path(dataset_path).absolute()

    # Verify dataset structure
    train_images = dataset_path / 'train' / 'images'
    train_labels = dataset_path / 'train' / 'labelTxt'
    valid_images = dataset_path / 'valid' / 'images'
    valid_labels = dataset_path / 'valid' / 'labelTxt'
    test_images = dataset_path / 'test' / 'images'
    test_labels = dataset_path / 'test' / 'labelTxt'

    print("\n📁 Checking dataset structure...")

    if not train_images.exists():
        raise FileNotFoundError(f"❌ Train images not found: {train_images}")
    if not train_labels.exists():
        raise FileNotFoundError(f"❌ Train labels not found: {train_labels}")
    if not valid_images.exists():
        raise FileNotFoundError(f"❌ Valid images not found: {valid_images}")
    if not valid_labels.exists():
        raise FileNotFoundError(f"❌ Valid labels not found: {valid_labels}")

    # Count files
    train_img_count = len(list(train_images.glob('*.[jp][pn]g')) + list(train_images.glob('*.[JP][PN]G')))
    train_label_count = len(list(train_labels.glob('*.txt')))
    valid_img_count = len(list(valid_images.glob('*.[jp][pn]g')) + list(valid_images.glob('*.[JP][PN]G')))
    valid_label_count = len(list(valid_labels.glob('*.txt')))

    print(f"✓ Train: {train_img_count} images, {train_label_count} labels")
    print(f"✓ Valid: {valid_img_count} images, {valid_label_count} labels")

    if test_images.exists():
        test_img_count = len(list(test_images.glob('*.[jp][pn]g')) + list(test_images.glob('*.[JP][PN]G')))
        print(f"✓ Test:  {test_img_count} images")

    # Check if data.yaml already exists
    existing_yaml = dataset_path / 'data.yaml'
    if existing_yaml.exists():
        print(f"\n📄 Found existing data.yaml in dataset folder")
        try:
            with open(existing_yaml, 'r') as f:
                existing_data = yaml.safe_load(f)
                if 'names' in existing_data and 'nc' in existing_data:
                    print(f"✓ Loading class information from existing data.yaml")
                    print(f"✓ Number of classes: {existing_data['nc']}")
                    print(f"✓ Classes: {existing_data['names']}")

                    # Update paths to absolute paths
                    existing_data['path'] = str(dataset_path)
                    existing_data['train'] = 'train/images'
                    existing_data['val'] = 'valid/images'
                    if test_images.exists():
                        existing_data['test'] = 'test/images'

                    # Save updated YAML
                    with open(output_file, 'w') as f:
                        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)

                    print(f"\n✓ Dataset YAML created: {Path(output_file).absolute()}")
                    return str(Path(output_file).absolute())
        except Exception as e:
            print(f"⚠️  Could not read existing data.yaml: {e}")

    # Detect classes from labels if data.yaml doesn't exist
    class_ids = detect_classes_from_labels(train_labels)

    if not class_ids:
        print("\n⚠️  No classes detected. Using default underwater creature classes.")
        print("   Please update the 'names' list in the generated YAML file.")
        class_ids = list(range(20))  # Default to 20 classes

    # Default class names for underwater creatures
    # IMPORTANT: Update this list to match YOUR dataset's actual classes
    default_class_names = [
        'fish',
        'shark',
        'turtle',
        'jellyfish',
        'octopus',
        'crab',
        'lobster',
        'seahorse',
        'starfish',
        'coral',
        'ray',
        'dolphin',
        'whale',
        'seal',
        'urchin',
        'shrimp',
        'squid',
        'eel',
        'clownfish',
        'submarine'
    ]

    # Create class names list based on detected class IDs
    num_classes = max(class_ids) + 1 if class_ids else 20
    class_names = []

    for i in range(num_classes):
        if i < len(default_class_names):
            class_names.append(default_class_names[i])
        else:
            class_names.append(f'class_{i}')

    # Create YAML structure
    data_yaml = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': num_classes,
        'names': class_names
    }

    if test_images.exists():
        data_yaml['test'] = 'test/images'

    # Save to file
    yaml_path = Path(output_file)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Dataset YAML created: {yaml_path.absolute()}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Classes: {class_names}")

    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Please verify the class names above!")
    print("   If they don't match your dataset, edit 'underwater_data.yaml'")
    print("   or update your existing 'data.yaml' in the dataset folder.")
    print("="*70)

    return str(yaml_path.absolute())


def train_model(config):
    """Train the YOLOv8 model"""

    print("="*70)
    print("       UNDERWATER CREATURE DETECTION - MODEL TRAINING")
    print("="*70)

    # Check if dataset exists
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset path not found!")
        print(f"   Looking for: {config.DATASET_PATH}")
        print("\n   Please verify the path is correct.")
        return None

    print(f"\n✓ Dataset found at: {dataset_path}")

    try:
        # Create dataset YAML
        data_yaml = create_dataset_yaml(config.DATASET_PATH)
    except Exception as e:
        print(f"\n❌ Error creating dataset configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Load pre-trained model
    print(f"\n📦 Loading {config.MODEL_SIZE} model...")
    model = YOLO(config.MODEL_SIZE)

    # Display training configuration
    print("\n⚙️  Training Configuration:")
    print(f"   Device: {'GPU' if config.DEVICE == 0 else 'CPU'} ({config.DEVICE})")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Image Size: {config.IMG_SIZE}")
    print(f"   Augmentation: {'Enabled' if config.AUGMENT else 'Disabled'}")
    print(f"   Workers: {config.WORKERS}")

    # Train the model
    print("\n🚀 Starting training...\n")
    print("="*70)

    try:
        results = model.train(
            data=data_yaml,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            device=config.DEVICE,
            workers=config.WORKERS,
            patience=config.PATIENCE,
            save_period=config.SAVE_PERIOD,

            # Augmentation parameters (optimized for underwater)
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

            # Additional settings
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save=True,

            # Project naming
            project='underwater_detection',
            name='training_run',
            exist_ok=True,

            # Learning rate and momentum
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

        print("\n" + "="*70)
        print("✅ Training completed successfully!")
        print("="*70)

        # Show where results are saved
        results_path = Path('underwater_detection/training_run')
        best_model = results_path / 'weights' / 'best_modeel-2.pt'
        last_model = results_path / 'weights' / 'last.pt'

        print(f"\n📁 Results saved in: {results_path.absolute()}")
        print(f"📈 Best model: {best_model.absolute()}")
        print(f"📈 Last model: {last_model.absolute()}")

        # Validate the model
        print("\n🔍 Validating model on validation set...")
        metrics = model.val()

        print("\n📊 Validation Metrics:")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   mAP50:    {metrics.box.map50:.4f}")
        print(f"   mAP75:    {metrics.box.map75:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall:    {metrics.box.mr:.4f}")

        # Interpretation
        print("\n📊 Model Performance:")
        if metrics.box.map50 > 0.8:
            print("   🎉 Excellent! (mAP50 > 0.80)")
        elif metrics.box.map50 > 0.7:
            print("   ✅ Good! (mAP50 > 0.70)")
        elif metrics.box.map50 > 0.6:
            print("   ⚠️  Fair (mAP50 > 0.60) - Consider more training")
        else:
            print("   ❌ Needs improvement - Try more epochs or data")

        # Show plots location
        print(f"\n📊 Training plots saved in: {results_path.absolute()}")
        print("   - results.png (metrics over time)")
        print("   - confusion_matrix.png")
        print("   - F1_curve.png")
        print("   - PR_curve.png")
        print("   - P_curve.png")
        print("   - R_curve.png")

        return model

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_on_sample(model_path, test_image_path):
    """Test the trained model on a sample image"""

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return

    print(f"\n🧪 Testing model on: {test_image_path}")
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=test_image_path,
        conf=0.25,
        iou=0.7,
        save=True,
        project='underwater_detection',
        name='test_results'
    )

    print(f"✅ Test results saved in: underwater_detection/test_results/")


if __name__ == "__main__":
    # Initialize configuration
    config = TrainingConfig()

    print("\n" + "="*70)
    print("  Welcome to Underwater Creature Detection Training Pipeline")
    print("="*70)
    print(f"\n📁 Dataset Path: {config.DATASET_PATH}")
    print(f"🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔧 Model: {config.MODEL_SIZE}")
    print(f"📊 Epochs: {config.EPOCHS}")
    print(f"📦 Batch Size: {config.BATCH_SIZE}")
    print("\n" + "="*70)

    # Train the model
    model = train_model(config)

    if model:
        # Get the best model path
        best_model_path = 'underwater_detection/training_run/weights/best_modeel-2.pt'

        # Optional: Test on sample images from test folder
        test_folder = Path(config.DATASET_PATH) / 'test' / 'images'
        if test_folder.exists():
            test_images = list(test_folder.glob('*.[jp][pn]g'))
            if test_images:
                print(f"\n🧪 Found {len(test_images)} test images")
                test_first = input("Test model on first test image? (y/n): ").strip().lower()
                if test_first == 'y':
                    test_model_on_sample(best_model_path, str(test_images[0]))

        print("\n" + "="*70)
        print("🎉 Training pipeline completed successfully!")
        print("="*70)
        print(f"\n📁 Your trained model is ready at:")
        print(f"   {Path(best_model_path).absolute()}")
        print("\n💡 Next steps:")
        print("   1. Check the training plots in underwater_detection/training_run/")
        print("   2. If mAP50 > 0.7, your model is good!")
        print("   3. Update webcam detection script with model path")
        print("   4. Run: python underwater_webcam_detection.py")
        print("\n📊 Recommended metrics for good model:")
        print("   - mAP50 > 0.70 (70%)")
        print("   - Precision > 0.75")
        print("   - Recall > 0.70")
        print("="*70 + "\n")