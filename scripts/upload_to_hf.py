#!/usr/bin/env python3
"""
Upload trained YOLO26 weights and metadata to Hugging Face Model Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

# Retrieve token from environment or local HF cache (~/.cache/huggingface/token)
TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "DangKhoa09/yolo26-vietnamese-money"

MODEL_CARD = """---
language:
- vi
- en
license: mit
tags:
- yolo
- yolo26
- ultralytics
- object-detection
- computer-vision
- vietnamese-money
- currency-recognition
datasets:
- blindtrix/money_vn_regconization-2-cumhs
pipeline_tag: object-detection
---

# 🇻🇳 YOLO26 Vietnamese Money Recognition & Smart Calculator

Mô hình **YOLO26** (kiến trúc NMS-Free End-to-End thế hệ mới nhất từ Ultralytics) nhận diện 9 mệnh giá tiền Việt Nam (1.000đ - 500.000đ) và tự động cộng dồn tổng tiền mặt.

- **GitHub Repository:** [https://github.com/bentovnace/Yolo26-Vietnamese-Money](https://github.com/bentovnace/Yolo26-Vietnamese-Money)
- **Tập dữ liệu Roboflow:** `blindtrix/money_vn_regconization-2-cumhs` (10,116 ảnh)

## 📌 9 Mệnh giá tiền hỗ trợ:
1. `1000`: 1.000 VNĐ
2. `2000`: 2.000 VNĐ
3. `5000`: 5.000 VNĐ
4. `10000`: 10.000 VNĐ
5. `20000`: 20.000 VNĐ
6. `50000`: 50.000 VNĐ
7. `100000`: 100.000 VNĐ
8. `200000`: 200.000 VNĐ
9. `500000`: 500.000 VNĐ

## 💻 Hướng dẫn nạp mô hình trong Python:

```python
from ultralytics import YOLO

# Tải trực tiếp mô hình đã train từ Hugging Face
model = YOLO("https://huggingface.co/DangKhoa09/yolo26-vietnamese-money/resolve/main/best.pt")

# Dự đoán trên ảnh
results = model.predict("anh_tien.jpg", conf=0.25)
results[0].show()
```
"""


def main():
    print(f"[*] Connecting to Hugging Face Model Hub...")
    api = HfApi(token=TOKEN)

    # Ensure repository exists
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"[+] Repository confirmed: https://huggingface.co/{REPO_ID}")

    # Upload Model Card README
    print("[*] Uploading Model Card README.md...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )

    base_dir = Path(__file__).resolve().parent.parent

    # Upload weights
    files_to_check = [
        base_dir / "models" / "best.pt",
        base_dir / "models" / "yolo26n.pt",
        base_dir / "models" / "best.onnx",
    ]

    for p in files_to_check:
        if p.exists():
            print(f"[*] Uploading {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"[+] Uploaded {p.name} successfully!")

    print(f"\n{'='*60}")
    print(f"🎉 HOÀN TẤT SETUP MODEL HUB TRÊN HUGGING FACE!")
    print(f"👉 Link Model: https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
