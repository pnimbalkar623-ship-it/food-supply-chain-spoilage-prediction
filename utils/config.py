"""
config.py
---------
Central configuration for the project: file paths, model hyperparameters,
image dimensions, random seeds, etc.

TODO: Populate as the project develops.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────────
IMAGE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
CONDITION_CSV  = os.path.join(PROJECT_ROOT, "data", "condition_data", "Dataset.csv")

# ── Model save paths ─────────────────────────────────────────────────────────
IMAGE_MODEL_DIR     = os.path.join(PROJECT_ROOT, "models", "image_model")
CONDITION_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "condition_model")
YOLO_MODEL_DIR      = os.path.join(PROJECT_ROOT, "models", "object_detector")

# ── YOLO & Archive Dataset paths ─────────────────────────────────────────────
ARCHIVE_DATASET_DIR   = os.path.join(os.path.dirname(PROJECT_ROOT), "archive")
ARCHIVE_DATA_YAML     = os.path.join(ARCHIVE_DATASET_DIR, "data.yaml")
ARCHIVE_PRETRAINED_PT = os.path.join(ARCHIVE_DATASET_DIR, "yolov8s.pt")
YOLO_MODERATE_WEIGHTS = os.path.join(YOLO_MODEL_DIR, "yolov8s_fruit_moderate.pt")
YOLO_IMPROVED_WEIGHTS = os.path.join(YOLO_MODEL_DIR, "yolov8s_fruit_improved.pt")
YOLO_BACKUP_WEIGHTS   = os.path.join(YOLO_MODEL_DIR, "yolov8s_fruit_improved_backup.pt")
YOLO_TRIAL_WEIGHTS    = os.path.join(PROJECT_ROOT, "runs", "detect", "models", "object_detector", "cpu_trial", "weights", "best.pt")

# ── Inference Configuration ──────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MIN_CONFIDENCE_THRESHOLD = 0.05
MAX_CONFIDENCE_THRESHOLD = 0.95

DEFAULT_IOU_THRESHOLD = 0.55
MIN_IOU_THRESHOLD = 0.10
MAX_IOU_THRESHOLD = 0.95

DEFAULT_INFERENCE_IMGSZ = 320
SUPPORTED_INFERENCE_RESOLUTIONS = [320, 480, 640]

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
