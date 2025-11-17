"""
Improved Underwater Trash + Marine Life Detection Training
Optimized to distinguish between similar objects (e.g., plastic bag vs jellyfish)
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================
# CONFIGURATION
# ============================================

class ImprovedTrainingConfig:
    """Enhanced configuration for trash detection"""

    # Paths
    PROJECT_DIR = r"D:\Projects\Underwater Trash Detection"
    DATASET_PATH = os.path.join(PROJECT_DIR, "merged-dataset", "merged_dataset.yaml")

    # Model selection - YOLOv8m recommended for better accuracy
    BASE_MODEL = "yolov8m.pt"  # Medium model - better for distinguishing similar objects

    # Training parameters - OPTIMIZED FOR TRASH DETECTION
    EPOCHS = 150  # More epochs for learning subtle differences
    IMG_SIZE = 640  # Standard size
    BATCH_SIZE = 8  # Adjust based on GPU
    DEVICE = 0  # GPU
    WORKERS = 4  # Data loading threads

    # Advanced training settings
    PATIENCE = 50  # Early stopping patience
    SAVE_PERIOD = 10  # Save checkpoint every N epochs

    # CRITICAL: Augmentation for trash vs marine life distinction
    # These help model learn to distinguish plastic bags from jellyfish
    AUGMENT = True
    MOSAIC = 1.0  # Mosaic augmentation (helps with context)
    MIXUP = 0.15  # Mix images to learn boundaries
    COPY_PASTE = 0.3  # Copy-paste augmentation (good for trash)

    # Geometric augmentation (trash can appear at any angle)
    DEGREES = 15.0  # Rotation range
    TRANSLATE = 0.2  # Translation
    SCALE = 0.7  # Scaling (trash varies in size)
    SHEAR = 0.1  # Shear transformation
    PERSPECTIVE = 0.0005  # Perspective distortion

    # Flip augmentation
    FLIPUD = 0.5  # Vertical flip
    FLIPLR = 0.5  # Horizontal flip

    # Color augmentation (CRITICAL for underwater scenes)
    HSV_H = 0.02  # Hue shift (lighting variations)
    HSV_S = 0.9  # Saturation (underwater color cast)
    HSV_V = 0.5  # Brightness (depth variations)

    # Advanced augmentation for underwater
    BLUR = 0.01  # Motion blur (water current)
    NOISE = 0.02  # Noise (particles in water)

    # Optimizer settings - CRITICAL
    OPTIMIZER = 'AdamW'  # Better for complex distinctions
    LR0 = 0.001  # Initial learning rate
    LRF = 0.001  # Final learning rate (keep learning)
    MOMENTUM = 0.937  # SGD momentum
    WEIGHT_DECAY = 0.0005  # Regularization

    # Loss weights - CRITICAL FOR TRASH DETECTION
    BOX = 7.5  # Box loss weight
    CLS = 1.5  # Class loss weight (INCREASED - important!)
    DFL = 1.5  # Distribution focal loss

    # Learning rate schedule
    WARMUP_EPOCHS = 5  # Warmup period
    WARMUP_MOMENTUM = 0.8  # Warmup momentum
    WARMUP_BIAS_LR = 0.1  # Warmup bias learning rate
    COS_LR = True  # Cosine LR scheduler

    # Label smoothing (helps with similar classes)
    LABEL_SMOOTHING = 0.1  # Smooth labels (helps with jellyfish vs plastic)

    # Multi-scale training
    MULTI_SCALE = True  # Train at different scales

    # Close mosaic in final epochs (improves final accuracy)
    CLOSE_MOSAIC = 15

    # Project naming
    RUN_NAME = "underwater_trash_detection_improved"


def analyze_dataset(dataset_yaml_path):
    """Analyze dataset to understand class distribution"""

    print("\n" + "=" * 70)
    print("           DATASET ANALYSIS")
    print("=" * 70)

    if not os.path.exists(dataset_yaml_path):
        print(f"\n❌ Dataset YAML not found: {dataset_yaml_path}")
        return None

    # Load YAML
    with open(dataset_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"\n📁 Dataset: {data.get('path', 'Unknown')}")
    print(f"📊 Number of classes: {data.get('nc', 'Unknown')}")

    # Display classes
    if 'names' in data:
        print(f"\n🏷️  Classes:")

        # Separate trash from marine life
        trash_classes = []
        marine_classes = []

        for idx, name in enumerate(data['names']):
            class_lower = name.lower()

            # Identify trash-related classes
            if any(word in class_lower for word in [
                'plastic', 'bag', 'bottle', 'can', 'trash', 'waste',
                'debris', 'garbage', 'litter', 'pollution', 'wrapper',
                'container', 'cup', 'straw', 'foam'
            ]):
                trash_classes.append((idx, name))
                print(f"   {idx}: {name} 🗑️  [TRASH]")
            else:
                marine_classes.append((idx, name))
                print(f"   {idx}: {name} 🐠 [MARINE LIFE]")

        print(f"\n📊 Distribution:")
        print(f"   Trash items: {len(trash_classes)}")
        print(f"   Marine life: {len(marine_classes)}")

        # Warn about confusing pairs
        print(f"\n⚠️  Watch out for confusing pairs:")
        jellyfish_idx = [i for i, n in marine_classes if 'jellyfish' in n.lower()]
        plastic_idx = [i for i, n in trash_classes if 'bag' in n.lower() or 'plastic' in n.lower()]

        if jellyfish_idx and plastic_idx:
            print(f"   • Jellyfish vs Plastic bags - CRITICAL distinction!")

        print(f"   • Transparent objects need special attention")
        print(f"   • Similar shapes/colors require strong augmentation")

    return data


def check_dataset_balance(dataset_path):
    """Check if dataset is balanced between classes"""

    print("\n" + "=" * 70)
    print("           CLASS BALANCE CHECK")
    print("=" * 70)

    dataset_root = Path(dataset_path).parent
    train_labels = dataset_root / 'train' / 'labels'

    if not train_labels.exists():
        print("\n⚠️  Cannot find training labels")
        return

    # Count instances per class
    class_counts = {}

    for label_file in train_labels.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

    if class_counts:
        total = sum(class_counts.values())
        print(f"\n📊 Total annotations: {total}")
        print(f"\n📈 Class distribution:")

        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"   Class {class_id}: {count:5d} ({percentage:5.1f}%) {bar}")

        # Check imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count

        print(f"\n⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 10:
            print("   ⚠️  HIGH IMBALANCE! Consider:")
            print("      - Collecting more data for rare classes")
            print("      - Using class weights")
            print("      - Data augmentation for minority classes")
        elif imbalance_ratio > 5:
            print("   ⚠️  Moderate imbalance - augmentation will help")
        else:
            print("   ✅ Good balance!")


def train_improved_model(config):
    """Train with improved settings for trash detection"""

    print("\n" + "=" * 70)
    print("   IMPROVED UNDERWATER TRASH + MARINE LIFE DETECTION")
    print("   Optimized to distinguish similar objects")
    print("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: GPU not available! Training will be VERY slow.")
        print("   Consider using Google Colab for GPU training.")
        config.DEVICE = 'cpu'
        config.BATCH_SIZE = 2
        config.WORKERS = 2
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.2f} GB")

        # Adjust batch size based on VRAM
        if gpu_memory < 4:
            config.BATCH_SIZE = 4
            print("   Batch size adjusted to 4 (low VRAM)")
        elif gpu_memory < 6:
            config.BATCH_SIZE = 8
            print("   Batch size adjusted to 8 (medium VRAM)")

    # Analyze dataset
    dataset_info = analyze_dataset(config.DATASET_PATH)
    if not dataset_info:
        return None

    # Check class balance
    check_dataset_balance(config.DATASET_PATH)

    # Training time estimate
    print("\n" + "=" * 70)
    print("           TRAINING ESTIMATE")
    print("=" * 70)

    if config.DEVICE == 'cpu':
        print("\n⏱️  Estimated time: 50-80 hours (CPU)")
    else:
        if 'yolov8n' in config.BASE_MODEL:
            est_time = config.EPOCHS * 1.5 / 60
        elif 'yolov8s' in config.BASE_MODEL:
            est_time = config.EPOCHS * 2.5 / 60
        elif 'yolov8m' in config.BASE_MODEL:
            est_time = config.EPOCHS * 4.5 / 60
        else:
            est_time = config.EPOCHS * 2 / 60

        print(f"\n⏱️  Estimated time: ~{est_time:.1f} hours (GPU)")

    print(f"📊 Epochs: {config.EPOCHS}")
    print(f"📦 Batch size: {config.BATCH_SIZE}")
    print(f"🖼️  Image size: {config.IMG_SIZE}")

    # Load model
    print(f"\n📦 Loading model: {config.BASE_MODEL}")
    model = YOLO(config.BASE_MODEL)

    # Training configuration
    print("\n⚙️  Training Configuration:")
    print(f"   Optimizer: {config.OPTIMIZER}")
    print(f"   Learning rate: {config.LR0}")
    print(f"   Augmentation: {'Enabled' if config.AUGMENT else 'Disabled'}")
    print(f"   Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"   Class loss weight: {config.CLS} (increased for better distinction)")

    print("\n🚀 Starting training...")
    print("=" * 70 + "\n")

    # Train with optimized parameters
    try:
        results = model.train(
            data=config.DATASET_PATH,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            device=config.DEVICE,
            workers=config.WORKERS,
            patience=config.PATIENCE,
            save_period=config.SAVE_PERIOD,

            # Augmentation - CRITICAL for distinguishing similar objects
            augment=config.AUGMENT,
            mosaic=config.MOSAIC,
            mixup=config.MIXUP,
            copy_paste=config.COPY_PASTE,
            degrees=config.DEGREES,
            translate=config.TRANSLATE,
            scale=config.SCALE,
            shear=config.SHEAR,
            perspective=config.PERSPECTIVE,
            flipud=config.FLIPUD,
            fliplr=config.FLIPLR,
            hsv_h=config.HSV_H,
            hsv_s=config.HSV_S,
            hsv_v=config.HSV_V,
            blur=config.BLUR,
            noise=config.NOISE,

            # Optimizer settings
            optimizer=config.OPTIMIZER,
            lr0=config.LR0,
            lrf=config.LRF,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            warmup_epochs=config.WARMUP_EPOCHS,
            warmup_momentum=config.WARMUP_MOMENTUM,
            warmup_bias_lr=config.WARMUP_BIAS_LR,
            cos_lr=config.COS_LR,

            # Loss weights - IMPORTANT
            box=config.BOX,
            cls=config.CLS,  # Higher class loss for better distinction
            dfl=config.DFL,

            # Label smoothing
            label_smoothing=config.LABEL_SMOOTHING,

            # Multi-scale training
            multi_scale=config.MULTI_SCALE,

            # Close mosaic
            close_mosaic=config.CLOSE_MOSAIC,

            # Other settings
            pretrained=True,
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            resume=False,
            amp=True,  # Mixed precision
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            val=True,
            plots=True,
            save=True,

            # Project naming
            project=os.path.join(config.PROJECT_DIR, "training_runs"),
            name=config.RUN_NAME,
            exist_ok=True,
        )

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)

        # Validate
        print("\n🔍 Validating model...")
        metrics = model.val()

        print("\n📊 FINAL RESULTS:")
        print("=" * 70)
        print(f"   mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.1f}%)")
        print(f"   mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map * 100:.1f}%)")
        print(f"   Precision: {metrics.box.mp:.4f} ({metrics.box.mp * 100:.1f}%)")
        print(f"   Recall:    {metrics.box.mr:.4f} ({metrics.box.mr * 100:.1f}%)")

        # Performance evaluation
        if metrics.box.map50 > 0.80:
            print("\n🎉 Excellent! Model should distinguish trash well!")
        elif metrics.box.map50 > 0.70:
            print("\n✅ Good! Model ready for deployment.")
        elif metrics.box.map50 > 0.60:
            print("\n⚠️  Fair. Consider training longer or using YOLOv8l.")
        else:
            print("\n❌ Low accuracy. Check dataset quality and labels.")

        # Check for confusion matrix
        best_model_dir = Path(config.PROJECT_DIR) / "training_runs" / config.RUN_NAME
        confusion_matrix = best_model_dir / "confusion_matrix.png"

        if confusion_matrix.exists():
            print(f"\n📊 Check confusion matrix to see if model still confuses:")
            print(f"   • Plastic bags with jellyfish")
            print(f"   • Similar colored objects")
            print(f"   File: {confusion_matrix}")

        # Model location
        best_model = best_model_dir / "weights" / "best.pt"
        print(f"\n📁 BEST MODEL SAVED:")
        print(f"   {best_model.absolute()}")

        # Tips for improvement
        if metrics.box.map50 < 0.75:
            print("\n💡 To improve trash detection:")
            print("   1. Train for more epochs (200-300)")
            print("   2. Use YOLOv8l or YOLOv8x model")
            print("   3. Collect more trash images (especially plastic bags)")
            print("   4. Add hard negative examples")
            print("   5. Use focal loss (set cls=2.0)")

        print("=" * 70)

        return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU OUT OF MEMORY!")
            print("\n💡 Solutions:")
            print("   1. Reduce batch_size to 4")
            print("   2. Use yolov8s.pt instead of yolov8m.pt")
            print("   3. Reduce IMG_SIZE to 512")
            print("\n   Edit config and try again.")
        else:
            print(f"\n❌ Error: {e}")
        return None

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Checkpoint saved - can resume training")
        return None


def test_on_confusing_cases(model_path, test_images_dir):
    """Test model on potentially confusing cases"""

    print("\n" + "=" * 70)
    print("           TESTING ON CONFUSING CASES")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return

    if not os.path.exists(test_images_dir):
        print(f"\n⚠️  Test images directory not found: {test_images_dir}")
        return

    model = YOLO(model_path)

    # Test on images
    results = model.predict(
        source=test_images_dir,
        conf=0.25,
        iou=0.7,
        save=True,
        project='confusion_test',
        name='results'
    )

    print(f"\n✅ Test results saved in: confusion_test/results/")
    print("\n💡 Check if model correctly identifies:")
    print("   • Plastic bags vs jellyfish")
    print("   • Bottles vs fish")
    print("   • Other transparent objects")


if __name__ == "__main__":
    # Initialize configuration
    config = ImprovedTrainingConfig()

    print("\n" + "=" * 70)
    print("   IMPROVED TRASH DETECTION TRAINING")
    print("   Optimized for distinguishing similar objects")
    print("=" * 70)

    print("\n💡 Key Improvements:")
    print("   ✅ Higher class loss weight (better distinction)")
    print("   ✅ Label smoothing (reduces overconfidence)")
    print("   ✅ Enhanced augmentation (learns variations)")
    print("   ✅ Copy-paste augmentation (context learning)")
    print("   ✅ Multi-scale training (size variations)")
    print("   ✅ More epochs (150 vs 30)")
    print("   ✅ YOLOv8m model (better accuracy)")

    input("\nPress ENTER to start training...")

    # Train
    model = train_improved_model(config)

    if model:
        print("\n🎉 Training pipeline completed!")
        print("\n📋 Next steps:")
        print("   1. Check confusion matrix in training_runs folder")
        print("   2. Test on validation images")
        print("   3. If still confusing objects, train longer")
        print("   4. Deploy model for real-time detection")