# fixed_detection.py
import os
import cv2
import time
from ultralytics import YOLO

# === CONFIG — change these two paths as needed ===
MODEL_PATH = r"D:\Project\undwerwater-object-detection\best.pt"
# If you want to run on video file instead of webcam:
VIDEO_PATH = "3.mp4"  # or None for webcam
# Image folder to run batch prediction (optional)
TEST_IMAGES_FOLDER = r"D:\Project\undwerwater-object-detection\fish-dataset\fish-dataset\test\images"
# ==================================================

def load_model(path=None):
    """Load YOLO model. If path is None or not found, fallback to pretrained yolov8n.pt"""
    if path and os.path.exists(path):
        print(f"[INFO] Loading model from: {path}")
        model = YOLO(path)
    else:
        if path:
            print(f"[WARN] Model path not found: {path}. Falling back to yolov8n.pt (pretrained).")
        else:
            print("[INFO] No model path provided — using yolov8n.pt (pretrained).")
        model = YOLO("..\Yolo-Weights\yolov8l.pt")
    return model

def predict_on_folder(model, folder_path, conf=0.5, save=True):
    """Run YOLO's built-in predictor on a folder of images and save outputs"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    print(f"[INFO] Running batch prediction on folder: {folder_path}")
    # ultralytics will handle saving under runs/detect/predict or similar
    results = model.predict(source=folder_path, conf=conf, save=save)
    print("[INFO] Prediction completed. Check runs/detect/predict/ for outputs.")
    return results

def run_live_detection(model, video_path=None, conf=0.5):
    """
    Run inference on webcam (if video_path is None) or on a video file.
    Displays live annotated frames.
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Opening video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print("[INFO] Opening default webcam (0)")

    if not cap.isOpened():
        print("[ERROR] Video source could not be opened.")
        return

    prev_time = time.time()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or failed to read frame.")
                break

            # Optional: resize to speed things up (comment out if you want original resolution)
            # frame = cv2.resize(frame, (640, 640))

            # Run model on frame
            # Using model(frame) returns a Results sequence
            results = model(frame, conf=conf)  # returns a sequence; results[0] is a Results object

            # Get annotated image from results
            try:
                annotated = results[0].plot()  # annotated is a numpy array in BGR (suitable for cv2)
            except Exception as e:
                # fallback: if plot() not available, draw boxes manually (less common)
                print(f"[WARN] results[0].plot() failed: {e}")
                annotated = frame

            # Calculate and overlay FPS
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

            # Show
            cv2.imshow("YOLO Live - Press Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested, exiting.")
                break
            elif key == ord('s'):
                # save one frame for debugging
                outname = f"debug_frame_{int(time.time())}.jpg"
                cv2.imwrite(outname, annotated)
                print(f"[INFO] Saved {outname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # load model
    model = load_model(MODEL_PATH)

    # Option A: Batch prediction on folder (saves annotated images)
    if os.path.exists(TEST_IMAGES_FOLDER):
        predict_on_folder(model, TEST_IMAGES_FOLDER, conf=0.45, save=True)
    else:
        print("[INFO] Test images folder not found; skipping batch prediction.")

    # Option B: Live detection (video file if present, else webcam)
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        run_live_detection(model, video_path=VIDEO_PATH, conf=0.45)
    else:
        run_live_detection(model, video_path=None, conf=0.45)
