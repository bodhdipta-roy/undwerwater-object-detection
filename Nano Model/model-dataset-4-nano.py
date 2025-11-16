"""
Underwater Creature Detection Training (CPU-Optimized)
Compatible with Dataset-4 structure
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import psutil


# ====================== CONFIGURATION ====================== #
class TrainingConfig:
    MODEL_SIZE = 'yolov8n.pt'  # Use yolov8s.pt for better accuracy (GPU recommended)
    DATASET_PATH = r'D:\Projects\UnderWaterObjectDetection\Dataset-4\Underwater'

    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 8
    WORKERS = 2
    DEVICE = 0

    # Training settings
    PATIENCE = 20
    SAVE_PERIOD = 5

    # Augmentation
    AUGMENT = True
    MOSAIC = 1.0
    MIXUP = 0.0
    DEGREES = 10.0
    TRANSLATE = 0.2
    SCALE = 0.5
    FLIPUD = 0.5
    FLIPLR = 0.5
    HSV_H = 0.015
    HSV_S = 0.7
    HSV_V = 0.4


# ====================== YAML HANDLER ====================== #
def create_dataset_yaml(dataset_path, output_file='underwater_data.yaml'):
    dataset_path = Path(dataset_path).absolute()
    existing_yaml = dataset_path / 'data.yaml'

    if existing_yaml.exists():
        print(f"\n✅ Found existing data.yaml")
        with open(existing_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        yaml_data['path'] = str(dataset_path)
        yaml_data['train'] = 'train/images'
        yaml_data['val'] = 'val/images'

        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Using existing class configuration")
        print(f"✓ Classes: {yaml_data.get('names', [])}")
        return str(Path(output_file).absolute())

    else:
        raise FileNotFoundError("❌ No data.yaml found in dataset path!")


# ====================== TRAINING FUNCTION ====================== #
def train_model(config):
    print("\n" + "=" * 60)
    print("       UNDERWATER OBJECT DETECTION - TRAINING START")
    print("=" * 60)

    print(f"\n💾 Dataset: {config.DATASET_PATH}")
    data_yaml = create_dataset_yaml(config.DATASET_PATH)

    print(f"\n📦 Loading model: {config.MODEL_SIZE}")
    model = YOLO(config.MODEL_SIZE)

    print("\n🚀 Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        patience=config.PATIENCE,
        save_period=config.SAVE_PERIOD,
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
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=42,
        val=True,
        plots=True,
        save=True,
        project='underwater_detection_nano',
        name='training_run',
        exist_ok=True
    )

    print("\n✅ Training complete! Validating model...\n")
    metrics = model.val()
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    print(f"📊 mAP50-95: {metrics.box.map:.4f}")

    best_model = Path('underwater_detection_nano/training_run/weights/best.pt').absolute()
    print(f"\n📁 Model saved at: {best_model}")
    print("=" * 60)
    return model


# ====================== MAIN ====================== #
if __name__ == "__main__":
    config = TrainingConfig()

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\n💻 System RAM: {ram_gb:.1f} GB")

    if ram_gb < 8:
        print("⚠️ Warning: Low RAM (<8GB). Expect slow training on CPU.")

    input("\nPress ENTER to start training... ")

    train_model(config)
