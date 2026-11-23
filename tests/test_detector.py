"""
Smoke test for YOLO26 Vietnamese Money Detector
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from app.detector import MoneyDetector, format_vnd


def test_detector_smoke():
    print("[*] Testing MoneyDetector initialization...")
    detector = MoneyDetector()
    assert detector.model is not None, "Model failed to load"
    print(f"[+] Loaded model path: {detector.model_path}")

    # Create synthetic test image (640x640x3)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (100, 100), (400, 300), (200, 200, 200), -1)

    print("[*] Testing prediction inference...")
    result = detector.predict(dummy_img, conf_threshold=0.25)
    assert result["success"] is True
    assert "total_amount_vnd" in result
    assert "total_notes" in result
    assert "breakdown" in result
    print(f"[+] Prediction response structure verified: total={result['total_amount_formatted']}")

    print("[*] Testing visualizer...")
    vis_img = detector.visualize(dummy_img, result)
    assert vis_img.shape == dummy_img.shape
    print("[+] Visualizer output verified successfully.")

    print("\n[SUCCESS] ALL DETECTOR SMOKE TESTS PASSED!")


if __name__ == "__main__":
    test_detector_smoke()
