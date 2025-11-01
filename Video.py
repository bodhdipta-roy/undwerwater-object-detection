# ================================
# Underwater Object Detection — Custom YOLO Model
# Video / Webcam Inference
# ================================

import os
import cv2
import time
from ultralytics import YOLO

# === CONFIGURATION ===
MODEL_PATH = r"D:\Project\undwerwater-object-detection\best.pt"  # your uploaded model
VIDEO_PATH = r"D:\Project\undwerwater-object-detection\UnderWater_Animal_Video\Octopus1.mp4\Octopus1.mp4"  # or None for webcam
CONF_THRESHOLD = 0.45
# ======================


def load_model(path):
    """Load your pre-trained YOLO model."""
    if os.path.exists(path):
        print(f"[INFO] Loading custom trained model from: {path}")
        return YOLO(path)
    else:
        print(f"[ERROR] Model not found at: {path}")
        exit(1)


def run_video_detection(model, conf=0.45, video_path=None):
    """Run YOLOv8 model on a video file or webcam feed."""
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    source_name = "webcam" if video_path is None else video_path
    print(f"[INFO] Starting detection from {source_name}...")

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

            # Run detection
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

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested. Exiting...")
                break
            elif key == ord('s'):
                filename = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[INFO] Saved snapshot: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection ended.")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    run_video_detection(model, conf=CONF_THRESHOLD, video_path=VIDEO_PATH)
