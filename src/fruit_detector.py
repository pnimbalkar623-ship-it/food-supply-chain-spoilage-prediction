"""
fruit_detector.py
-----------------
Integrates YOLOv8 object detection for multi-fruit localization and counting.
Supports single and batch image processing, drawing bounding boxes, and
aggregating counts per fruit category.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.config import (
    ARCHIVE_PRETRAINED_PT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFERENCE_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    PROJECT_ROOT,
    YOLO_BACKUP_WEIGHTS,
    YOLO_IMPROVED_WEIGHTS,
    YOLO_MODERATE_WEIGHTS,
    YOLO_TRIAL_WEIGHTS,
)

# Vibrant, high-contrast palette for fruit classes
CLASS_COLORS: dict[str, str] = {
    "apple": "#E63946",       # Red
    "banana": "#E9C46A",      # Golden Yellow
    "grape": "#7209B7",       # Purple
    "orange": "#F77F00",      # Bright Orange
    "pineapple": "#2A9D8F",   # Teal/Greenish
    "watermelon": "#2D6A4F",  # Deep Forest Green
    "default": "#457B9D",     # Steel Blue
}


@dataclass
class DetectionItem:
    """Represents a single detected fruit instance in an image."""
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixels


@dataclass
class DetectionResult:
    """Contains all detections, annotated image, and summary counts for one image."""
    image_name: str
    original_image: Image.Image
    annotated_image: Image.Image
    items: list[DetectionItem] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    error: str | None = None


def resolve_model_weights(preferred_path: str | None = None) -> str:
    """Find the best available YOLO weights file without failing silently."""
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path
    # 1. Primary: Moderate fine-tuned model (1,500 samples, 5 epochs, mAP50 33.4%)
    if os.path.isfile(YOLO_MODERATE_WEIGHTS):
        return YOLO_MODERATE_WEIGHTS
    runs_moderate_pt = os.path.join(
        PROJECT_ROOT, "runs", "detect", "models", "object_detector", "moderate_finetune", "weights", "best.pt"
    )
    if os.path.isfile(runs_moderate_pt):
        return runs_moderate_pt
    # 2. Safe Fallback: Previous improved weights (426 samples, 3 epochs, mAP50 23.7%)
    if os.path.isfile(YOLO_IMPROVED_WEIGHTS):
        return YOLO_IMPROVED_WEIGHTS
    # 3. Safe Fallback: Backup copy of previous weights
    if os.path.isfile(YOLO_BACKUP_WEIGHTS):
        return YOLO_BACKUP_WEIGHTS
    # 4. Fallback: Baseline trial weights
    if os.path.isfile(YOLO_TRIAL_WEIGHTS):
        return YOLO_TRIAL_WEIGHTS
    # 5. Fallback: Pretrained starting weights
    if os.path.isfile(ARCHIVE_PRETRAINED_PT):
        return ARCHIVE_PRETRAINED_PT
    raise FileNotFoundError(
        f"No YOLO weights found. Checked:\n"
        f"1. Moderate weights: {YOLO_MODERATE_WEIGHTS}\n"
        f"2. Improved weights: {YOLO_IMPROVED_WEIGHTS}\n"
        f"3. Trial weights: {YOLO_TRIAL_WEIGHTS}\n"
        f"4. Pretrained weights: {ARCHIVE_PRETRAINED_PT}"
    )


def load_yolo_model(weights_path: str | None = None) -> Any:
    """
    Load and initialize a YOLO detector model with safe fallback.
    If the primary model fails to load, attempts fallback to the previous model.
    """
    from ultralytics import YOLO

    path = resolve_model_weights(weights_path)
    try:
        model = YOLO(path)
        return model
    except Exception as exc:
        # Fallback candidates if loading primary weights encountered an error
        fallback_candidates = [
            YOLO_IMPROVED_WEIGHTS,
            YOLO_BACKUP_WEIGHTS,
            ARCHIVE_PRETRAINED_PT,
        ]
        for fb_path in fallback_candidates:
            if fb_path != path and os.path.isfile(fb_path):
                try:
                    return YOLO(fb_path)
                except Exception:
                    continue
        raise RuntimeError(
            f"Failed to load YOLO model from '{path}' and all fallbacks failed: {exc}"
        ) from exc


def get_class_color(class_name: str) -> str:
    """Return a consistent color for the given fruit category."""
    clean_name = class_name.strip().lower()
    for key, color in CLASS_COLORS.items():
        if key in clean_name:
            return color
    return CLASS_COLORS["default"]


def draw_bounding_boxes(
    image: Image.Image,
    items: list[DetectionItem],
    line_width: int = 3,
) -> Image.Image:
    """Draw clear, professional bounding boxes and labels onto an image copy."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Responsive font sizing based on image dimensions
    font_size = max(13, int(min(width, height) * 0.025))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for item in items:
        x1, y1, x2, y2 = item.bbox
        color = get_class_color(item.class_name)

        # Draw outer rectangle
        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Draw label badge
        label = f"{item.class_name.capitalize()} {item.confidence:.0%}"
        try:
            bbox_text = font.getbbox(label)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_w, text_h = len(label) * 8, 14

        padding = 4
        badge_y1 = max(0, y1 - text_h - padding * 2)
        badge_y2 = badge_y1 + text_h + padding * 2
        badge_x1 = x1
        badge_x2 = min(width, x1 + text_w + padding * 2)

        draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw.text(
            (badge_x1 + padding, badge_y1 + padding),
            label,
            fill="#FFFFFF",
            font=font,
        )

    return annotated


def detect_fruits_in_image(
    image: Image.Image,
    model: Any,
    image_name: str = "uploaded_image",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: int = DEFAULT_INFERENCE_IMGSZ,
) -> DetectionResult:
    """
    Run YOLO fruit detection on a single PIL image.
    Extracts individual fruit items, counts, and returns the annotated image.
    """
    if image is None:
        return DetectionResult(
            image_name=image_name,
            original_image=Image.new("RGB", (100, 100)),
            annotated_image=Image.new("RGB", (100, 100)),
            error="Empty or invalid image provided.",
        )

    try:
        # Run inference
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device="cpu",
        )

        detected_items: list[DetectionItem] = []
        counts: dict[str, int] = {}

        if results and len(results) > 0:
            result = results[0]
            names = model.names
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    coords = [float(coord.item()) for coord in box.xyxy[0]]
                    raw_name = names.get(cls_id, f"Fruit_{cls_id}")
                    clean_name = str(raw_name).strip().capitalize()

                    item = DetectionItem(
                        class_name=clean_name,
                        confidence=conf,
                        bbox=coords,
                    )
                    detected_items.append(item)
                    counts[clean_name] = counts.get(clean_name, 0) + 1

        annotated = draw_bounding_boxes(image, detected_items)

        return DetectionResult(
            image_name=image_name,
            original_image=image,
            annotated_image=annotated,
            items=detected_items,
            counts=counts,
            total_count=len(detected_items),
        )

    except Exception as exc:
        return DetectionResult(
            image_name=image_name,
            original_image=image,
            annotated_image=image.copy(),
            error=f"Detection failed: {str(exc)}",
        )


def detect_fruits_batch(
    images_with_names: list[tuple[str, Image.Image]],
    model: Any,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: int = DEFAULT_INFERENCE_IMGSZ,
) -> list[DetectionResult]:
    """Run detection across a batch of uploaded images."""
    results: list[DetectionResult] = []
    for name, img in images_with_names:
        res = detect_fruits_in_image(
            image=img,
            model=model,
            image_name=name,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        results.append(res)
    return results
