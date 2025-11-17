# ============================================
# Underwater Creatures + Trash Detection — YOLOv8 Training
# ============================================

import os
from ultralytics import YOLO

# === CONFIGURATION ===
# Project and dataset paths
PROJECT_DIR = r"D:\Projects\Underwater Trash Detection"
DATASET_PATH = os.path.join(PROJECT_DIR, "merged-dataset", "merged_dataset.yaml")

# Base YOLO model to start with (choose yolov8s.pt / yolov8m.pt / yolov8l.pt based on GPU)
BASE_MODEL = "yolov8n.pt"

# Training parameters
EPOCHS = 30          # Increase epochs for better accuracy
IMG_SIZE = 640        # Standard image input size
BATCH_SIZE = 4        # Adjust based on GPU VRAM (try 4 if VRAM < 6GB)
DEVICE = 0            # GPU ID (set 'cpu' if CUDA not available)
WORKERS = 2           # Data loader threads

# Save run details under project directory
RUN_NAME = "underwater_creatures_trash_training"
SAVE_DIR = os.path.join(PROJECT_DIR, "training_runs", RUN_NAME)

# ============================================

def train_underwater_trash_model():
    """Train YOLOv8 model to detect both marine life and underwater trash."""

    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Ensure dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset YAML not found at: {DATASET_PATH}")
        return
    else:
        print(f"[INFO] Using dataset config: {DATASET_PATH}")

    print(f"[INFO] Training results will be saved to: {SAVE_DIR}")

    # Begin training
    model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        name=RUN_NAME,
        project=os.path.dirname(SAVE_DIR),
        save=True,
        exist_ok=True,
        patience=25,           # Early stopping if no improvement
        cos_lr=True,           # Cosine learning rate scheduling
        optimizer='AdamW',     # More stable for smaller datasets
        lr0=0.001,             # Initial learning rate
        weight_decay=0.0005,   # Prevent overfitting
        pretrained=True,       # Use pretrained COCO weights
        verbose=True
    )

    print(f"[INFO] ✅ Training completed! Best weights saved in:")
    print(f"     {os.path.join(SAVE_DIR, 'weights', 'best.pt')}")

    # Evaluate trained model
    print("[INFO] Running validation on trained model...")
    results = model.val(data=DATASET_PATH, device=DEVICE)
    print(results)

    # Export final weights
    print("[INFO] Exporting trained model for inference...")
    model.export(format="pt")
    print("[INFO] ✅ Model exported successfully!")

# ============================================

if __name__ == "__main__":
    train_underwater_trash_model()
