"""
FastAPI Server for YOLO26 Vietnamese Money Detection & Calculator
"""

import io
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates

from app.detector import MoneyDetector

app = FastAPI(
    title="YOLO26 Vietnamese Money Recognition & Calculator API",
    description="Nhận diện các mệnh giá tiền Việt Nam (1.000đ - 500.000đ) và tự động tính tổng tiền bằng YOLO26",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global detector instance
detector = MoneyDetector()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render interactive Web UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_path": str(detector.model_path.name),
            "is_custom_model": detector.is_custom_model
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": detector.model is not None,
        "model_path": str(detector.model_path),
        "is_custom_model": detector.is_custom_model,
        "message": (
            "Using custom trained money recognition model"
            if detector.is_custom_model
            else "Using base YOLO26 model. Place best.pt into models/ for full Vietnamese money recognition."
        )
    }


@app.post("/api/v1/reload-model")
async def reload_model(weights_path: Optional[str] = Query(None, description="Optional path to new weights")):
    """Reload detector weights on demand without restarting the server/container."""
    global detector
    try:
        detector = MoneyDetector(model_path=weights_path)
        return {
            "status": "success",
            "message": f"Successfully reloaded model: {detector.model_path}",
            "is_custom_model": detector.is_custom_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.post("/api/v1/predict")
async def predict_api(
    file: UploadFile = File(..., description="Image file to detect"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.01, le=1.0, description="NMS IoU threshold"),
    imgsz: int = Query(640, description="Inference image resolution")
):
    """
    Run detection and return structured JSON with total VND calculation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP)")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = detector.predict(image, conf_threshold=conf, iou_threshold=iou, imgsz=imgsz)
    return JSONResponse(content=result)


@app.post("/api/v1/predict/visualize")
async def predict_visualize(
    file: UploadFile = File(..., description="Image file to annotate"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.01, le=1.0, description="NMS IoU threshold"),
    imgsz: int = Query(640, description="Inference image resolution")
):
    """
    Run detection and return annotated JPEG image with bounding boxes & summary banner.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = detector.predict(image, conf_threshold=conf, iou_threshold=iou, imgsz=imgsz)
    vis_image = detector.visualize(image, result)

    success, encoded_img = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")

    return Response(
        content=encoded_img.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Total-Amount-VND": str(result["total_amount_vnd"]),
            "X-Total-Notes": str(result["total_notes"])
        }
    )
