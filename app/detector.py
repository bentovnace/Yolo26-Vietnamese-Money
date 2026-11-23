"""
YOLO26 Vietnamese Money Detector Engine
Wraps Ultralytics YOLO26 for detection and automatic currency sum calculation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Standard denominations in Vietnamese Dong (VND)
DENOMINATION_VALUES: Dict[str, int] = {
    "1000": 1_000,
    "2000": 2_000,
    "5000": 5_000,
    "10000": 10_000,
    "20000": 20_000,
    "50000": 50_000,
    "100000": 100_000,
    "200000": 200_000,
    "500000": 500_000,
}

# Color palette (BGR for OpenCV) for distinct visual identification
DENOMINATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "1000": (128, 128, 128),     # Xám / Nâu nhạt
    "2000": (80, 110, 150),      # Nâu sẫm
    "5000": (180, 100, 30),      # Xanh lam đậm
    "10000": (40, 170, 220),     # Vàng đất
    "20000": (200, 160, 50),     # Xanh lơ / dương đậm
    "50000": (180, 80, 220),     # Hồng tím
    "100000": (50, 180, 60),     # Xanh lá cây
    "200000": (50, 70, 220),     # Đỏ cam
    "500000": (220, 140, 30),    # Xanh ngọc / lam sáng
}


def format_vnd(amount: int) -> str:
    """Format integer amount to Vietnamese currency string (e.g. 500,000 VNĐ)."""
    return f"{amount:,.0f} VNĐ".replace(",", ".")


class MoneyDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize detector with model weights.
        Defaults to checking:
        1. Explicit path passed
        2. models/best.pt (custom trained weights from Colab)
        3. models/yolo26n.pt (base pretrained fallback)
        """
        self.model_path = self._resolve_model_path(model_path)
        print(f"[*] Loading YOLO26 model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.is_custom_model = "best.pt" in str(self.model_path)
        print(f"[*] Model loaded successfully. Is custom trained money model: {self.is_custom_model}")

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        if model_path and Path(model_path).exists():
            return Path(model_path)

        custom_weights = base_dir / "models" / "best.pt"
        if custom_weights.exists():
            return custom_weights

        base_weights = base_dir / "models" / "yolo26n.pt"
        if base_weights.exists():
            return base_weights

        # Fallback to current directory or auto-download yolo26n.pt
        return Path("yolo26n.pt")

    def predict(
        self,
        image_input: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Run inference on image (numpy array, PIL Image, or file path).
        Returns structured detection info and total money calculation.
        """
        # Convert PIL to BGR numpy array if needed
        if isinstance(image_input, Image.Image):
            image_np = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image_np = image_input
        elif isinstance(image_input, (str, Path)):
            image_np = cv2.imread(str(image_input))
            if image_np is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        h, w = image_np.shape[:2]

        results = self.model.predict(
            source=image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        breakdown: Dict[str, int] = {}
        total_amount = 0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                # Calculate currency value if class matches denomination
                clean_cls = cls_name.replace("k", "000").replace("K", "000").strip()
                amount_value = DENOMINATION_VALUES.get(clean_cls, 0)

                if amount_value > 0:
                    total_amount += amount_value
                    breakdown[clean_cls] = breakdown.get(clean_cls, 0) + 1

                detections.append({
                    "box": xyxy,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "denomination_str": clean_cls,
                    "confidence": round(conf, 4),
                    "amount_vnd": amount_value,
                    "formatted_amount": format_vnd(amount_value) if amount_value else None,
                })

        # Sort breakdown by descending denomination
        sorted_breakdown = {
            k: breakdown[k]
            for k in sorted(breakdown.keys(), key=lambda x: int(x), reverse=True)
        }

        return {
            "success": True,
            "model_path": str(self.model_path),
            "is_custom_model": self.is_custom_model,
            "image_size": {"width": w, "height": h},
            "total_notes": len(detections),
            "total_amount_vnd": total_amount,
            "total_amount_formatted": format_vnd(total_amount),
            "breakdown": sorted_breakdown,
            "detections": detections,
        }

    def visualize(
        self,
        image_input: Any,
        predict_result: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and summary header on the image.
        Returns BGR numpy array ready for cv2.imwrite or web streaming.
        """
        if isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        elif isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
        else:
            raise TypeError("Unsupported image input type for visualization")

        h, w = image.shape[:2]

        for det in predict_result.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            cls_name = det["denomination_str"]
            conf = det["confidence"]
            color = DENOMINATION_COLORS.get(cls_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Label text
            label_text = f"{cls_name} ({conf*100:.1f}%)"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # Label background badge
            cv2.rectangle(
                image,
                (x1, max(0, y1 - th - baseline - 8)),
                (x1 + tw + 10, y1),
                color,
                -1,
            )
            cv2.putText(
                image,
                label_text,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Top summary overlay banner (use ASCII VND for cv2.putText compatibility)
        total_str = predict_result.get("total_amount_formatted", "0 VNĐ")
        ascii_total = total_str.replace("VNĐ", "VND").replace("đ", "d")
        notes_count = predict_result.get("total_notes", 0)
        banner_text = f"Tong: {ascii_total} | So to: {notes_count}"

        header_h = 44
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        cv2.putText(
            image,
            banner_text,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return image
