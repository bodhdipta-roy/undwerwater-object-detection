import io
import os
import tempfile
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image
import cv2

from ultralytics import YOLO

# -------- Config -------- #
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'best-50.pt'))
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

app = FastAPI(title="Underwater Fish Detection API", version="1.0.0")

# Enable CORS for Flutter/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model
_model: Optional[YOLO] = None
_model_classes: Dict[int, str] = {}


def get_model() -> YOLO:
    global _model, _model_classes
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
        # names is a dict mapping class index to label
        _model_classes = _model.model.names if hasattr(_model, 'model') else _model.names
    return _model


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    box: Box
    confidence: float
    class_id: int
    label: str
    is_fish: bool


class ImageDetectResponse(BaseModel):
    detections: List[Detection]
    num_fish: int
    num_objects: int
    conf_threshold: float
    iou_threshold: float


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        mdl = get_model()
        num_classes = len(_model_classes)
        return {"status": "ok", "model_loaded": True, "classes": num_classes}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.get("/labels")
async def labels() -> Dict[int, str]:
    mdl = get_model()
    return {int(k): str(v) for k, v in _model_classes.items()}


def _is_fish_label(label: str) -> bool:
    # Heuristic: labels containing 'fish' are considered fish
    return 'fish' in label.lower()


def _results_to_detections(results, conf_thr: float, iou_thr: float) -> List[Detection]:
    detections: List[Detection] = []
    for r in results:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else np.zeros((0, 4))
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.zeros((0,))
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.zeros((0,), dtype=int)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            score = float(conf[i])
            cid = int(cls[i])
            label = str(_model_classes.get(cid, str(cid)))
            detections.append(Detection(
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=score,
                class_id=cid,
                label=label,
                is_fish=_is_fish_label(label),
            ))
    return detections


@app.post("/detect/image", response_model=ImageDetectResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold")
):
    try:
        mdl = get_model()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(image)

        results = mdl.predict(source=img_np, conf=conf, iou=iou, verbose=False)
        detections = _results_to_detections(results, conf, iou)
        num_fish = sum(1 for d in detections if d.is_fish)
        return ImageDetectResponse(
            detections=detections,
            num_fish=num_fish,
            num_objects=len(detections),
            conf_threshold=conf,
            iou_threshold=iou,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect/image/annotated")
async def detect_image_annotated(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0)
):
    try:
        mdl = get_model()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(image)

        results = mdl.predict(source=img_np, conf=conf, iou=iou, save=False, verbose=False)
        # Use Ultralytics plotting utility
        plotted = results[0].plot()
        # Encode to PNG
        is_ok, buf = cv2.imencode('.png', cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        if not is_ok:
            raise RuntimeError('Failed to encode annotated image')
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0)
):
    try:
        mdl = get_model()
        # Save uploaded video to a temp file
        suffix = os.path.splitext(file.filename or '')[-1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            contents = await file.read()
            tmp_in.write(contents)
            input_path = tmp_in.name

        # Prepare output annotated video path
        out_fd, out_path = tempfile.mkstemp(suffix='.mp4')
        os.close(out_fd)

        # Run prediction with save=True to create annotated video, or manually draw
        results = mdl.predict(source=input_path, conf=conf, iou=iou, save=True, verbose=False, project=os.path.dirname(out_path), name=os.path.basename(out_path).split('.')[0])

        # Ultralytics saves to {project}/{name}/{original_filename}
        save_dir = os.path.join(os.path.dirname(out_path), os.path.basename(out_path).split('.')[0])
        # Try to find the annotated video in the save_dir
        annotated_path = None
        if os.path.isdir(save_dir):
            for fname in os.listdir(save_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    annotated_path = os.path.join(save_dir, fname)
                    break
        if annotated_path is None:
            # Fallback: return original if not created
            annotated_path = input_path

        # Return the annotated video file
        return FileResponse(annotated_path, media_type='video/mp4', filename='annotated.mp4')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # We intentionally do not delete temp files immediately to allow FileResponse to stream.
        # A background cleanup process could be added if needed.
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
