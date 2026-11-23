#!/usr/bin/env python3
"""
CLI Predictor for YOLO26 Vietnamese Money Recognition & Calculator
Usage:
    python predict_cli.py --source path/to/image.jpg
    python predict_cli.py --source path/to/folder/ --weights models/best.pt --conf 0.3
"""

import argparse
from pathlib import Path
import cv2
from app.detector import MoneyDetector, format_vnd


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Vietnamese Money CLI Predictor")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory, or video/webcam ('0')")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (default: models/best.pt or yolo26n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (default: 640)")
    parser.add_argument("--output-dir", type=str, default="runs/predict", help="Output directory for annotated images")
    parser.add_argument("--no-save", action="store_true", help="Do not save output images to disk")
    args = parser.parse_args()

    detector = MoneyDetector(model_path=args.weights)
    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = [
            p for p in source_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ]
    else:
        print(f"[!] Invalid source path: {args.source}")
        return

    print(f"\n{'='*60}")
    print(f"[*] YOLO26 VIETNAMESE MONEY RECOGNITION CLI")
    print(f"[*] Model: {detector.model_path}")
    print(f"[*] Processing {len(image_files)} image(s)...")
    print(f"{'='*60}\n")

    grand_total_all_images = 0
    total_notes_all_images = 0

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Analyzing: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    [!] Error reading image: {img_path}")
            continue

        result = detector.predict(
            image,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz
        )

        total_amount = result["total_amount_vnd"]
        total_notes = result["total_notes"]
        breakdown = result["breakdown"]

        grand_total_all_images += total_amount
        total_notes_all_images += total_notes

        print(f"    -> Phat hien: {total_notes} to tien")
        print(f"    -> Tong gia tri: {format_vnd(total_amount)}")
        if breakdown:
            breakdown_str = ", ".join([f"{int(k):,}d: {v} to" for k, v in breakdown.items()])
            print(f"    -> Chi tiet: {breakdown_str}")

        if not args.no_save:
            annotated_img = detector.visualize(image, result)
            out_file = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_file), annotated_img)
            print(f"    -> Luu anh tai: {out_file}")
        print()

    print(f"{'='*60}")
    print(f"[SUMMARY] TONG KET TAT CA HINH ANH:")
    print(f"[*] Tong so to tien: {total_notes_all_images} to")
    print(f"[*] TONG SO TIEN:    {format_vnd(grand_total_all_images)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
