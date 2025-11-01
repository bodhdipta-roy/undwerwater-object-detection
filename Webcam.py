import os
import cv2
import time
from ultralytics import YOLO

# === CONFIGURATION ===
DATA_YAML = r"D:\Project\undwerwater-object-detection\fish-dataset\data.yaml"  # path to your dataset yaml
MODEL_PATH = r"D:\Project\undwerwater-object-detection\best.pt"  # model after training will be saved here
VIDEO_PATH = None  # set "2.mp4" for video file or None for webcam
EPOCHS = 50  # training epochs (adjust based on dataset size)
CONF_THRESHOLD = 0.45
# ======================


def train_model(data_yaml, epochs=50):
    """
    Train YOLOv8l model on the underwater dataset.
    """
    print(f"[INFO] Training YOLOv8l on dataset: {data_yaml}")
    model = YOLO("yolov8l.pt")  # use large model for better accuracy
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project="runs/train",
        name="underwater_yolov8l",
        batch=8,
        workers=2,
        device=0  # change to 'cpu' if no GPU
    )
    print("[INFO] Training complete! Best weights saved automatically.")
    return model


def load_model(path):
    """
    Load the trained model or fallback to yolov8l.pt
    """
    if os.path.exists(path):
        print(f"[INFO] Loading trained model from: {path}")
        return YOLO(path)
    else:
        print(f"[WARN] Trained model not found at {path}. Using pretrained yolov8l.pt instead.")
        return YOLO("..\Yolo-Weights\yolov8l.pt")


def run_live_detection(model, conf=0.45, video_path=None):
    """
    Run real-time detection from webcam or video file.
    """
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    source_name = "webcam" if video_path is None else video_path
    print(f"[INFO] Starting live detection from {source_name}...")

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or failed to read frame.")
                break

            # Predict
            results = model.predict(frame, conf=conf, verbose=False)
            annotated = results[0].plot()

            # FPS counter
            frame_count += 1
            if frame_count % 5 == 0:
                now = time.time()
                fps = 5 / (now - prev_time)
                prev_time = now
            else:
                fps = None

            if fps:
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Underwater Object Detection (Press Q to Quit)", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit requested.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection ended.")


if __name__ == "__main__":
    # === TRAIN PHASE ===
    if not os.path.exists(MODEL_PATH):
        model = train_model(DATA_YAML, epochs=EPOCHS)
        # After training, the best weights are usually saved at:
        best_path = os.path.join("runs", "train", "underwater_yolov8l", "weights", "best.pt")
        if os.path.exists(best_path):
            MODEL_PATH = best_path
    else:
        print("[INFO] Model already trained. Skipping training.")

    # === TEST PHASE (LIVE DETECTION) ===
    model = load_model(MODEL_PATH)
    run_live_detection(model, conf=CONF_THRESHOLD, video_path=VIDEO_PATH)
