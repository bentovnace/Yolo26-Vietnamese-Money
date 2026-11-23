"""
FastAPI Server for YOLO26 Vietnamese Money Detection & Calculator
Supports Cloud-Only Mode (connecting to Render Cloud) and Native Server Mode.
"""

import os
import io
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import httpx
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates

RENDER_CLOUD_URL = os.environ.get("RENDER_URL", "https://yolo26-vietnamese-money.onrender.com").rstrip("/")
IS_ON_RENDER = os.environ.get("RENDER") == "true" or os.environ.get("RENDER_SERVICE_ID") is not None

# When running locally via 'make dev', default to Render Cloud only (no local model loaded)
RENDER_ONLY = (os.environ.get("RENDER_ONLY", "1") == "1") and not IS_ON_RENDER

detector = None
if not RENDER_ONLY:
    try:
        from app.detector import MoneyDetector
        detector = MoneyDetector()
    except Exception as e:
        print(f"[!] Warning: could not initialize local MoneyDetector: {e}")

app = FastAPI(
    title="YOLO26 Vietnamese Money Recognition & Calculator API",
    description="Nhận diện 9 mệnh giá tiền Việt Nam (1.000đ - 500.000đ) và tính tổng tiền bằng YOLO26",
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render interactive Web UI connected to Render Cloud."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "render_cloud_url": RENDER_CLOUD_URL,
            "render_only": RENDER_ONLY,
            "model_path": "Render Cloud (best.pt)" if RENDER_ONLY else (str(detector.model_path.name) if detector else "best.pt"),
            "is_custom_model": True
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if RENDER_ONLY:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.get(f"{RENDER_CLOUD_URL}/health")
                cloud_data = r.json()
        except Exception as e:
            cloud_data = {"error": str(e), "status": "connecting"}
        return {
            "status": "online",
            "mode": "render_cloud_only",
            "render_url": RENDER_CLOUD_URL,
            "cloud_backend": cloud_data
        }

    return {
        "status": "online",
        "model_loaded": detector.model is not None if detector else False,
        "model_path": str(detector.model_path) if detector else "None",
        "is_custom_model": detector.is_custom_model if detector else False,
        "message": "Running on Render Cloud Backend"
    }


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
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if RENDER_ONLY:
        # Proxy request directly to Render Cloud
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{RENDER_CLOUD_URL}/api/v1/predict",
                    files={"file": (file.filename, contents, file.content_type or "image/jpeg")},
                    params={"conf": conf, "iou": iou, "imgsz": imgsz}
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json"
                )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Render Cloud connection error: {str(e)}")

    # Native local inference (when running on Render backend)
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
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if RENDER_ONLY:
        # Proxy request directly to Render Cloud
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{RENDER_CLOUD_URL}/api/v1/predict/visualize",
                    files={"file": (file.filename, contents, file.content_type or "image/jpeg")},
                    params={"conf": conf, "iou": iou, "imgsz": imgsz}
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="image/jpeg",
                    headers={k: v for k, v in resp.headers.items() if k.lower().startswith("x-total-")}
                )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Render Cloud connection error: {str(e)}")

    # Native local inference (when running on Render backend)
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
