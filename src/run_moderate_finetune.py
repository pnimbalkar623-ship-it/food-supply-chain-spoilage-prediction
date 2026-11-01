"""
run_moderate_finetune.py
-------------------------
Executes a 1,500-image, 5-epoch CPU fine-tuning run on YOLOv8s.
Evaluates on the held-out test dataset and compares metrics against the baseline.
"""

import json
import os
import time
from ultralytics import YOLO

ARCHIVE_YAML = r"C:\Users\pnimb\Downloads\archive\data.yaml"
STARTING_WEIGHTS = r"C:\Users\pnimb\Downloads\archive\yolov8s.pt"
PROJECT_DIR = "models/object_detector"
RUN_NAME = "moderate_finetune"

def main():
    print("=" * 70)
    print("Starting 1,500-image, 5-epoch CPU fine-tuning run...")
    print(f"Data configuration: {ARCHIVE_YAML}")
    print(f"Starting weights: {STARTING_WEIGHTS}")
    print("Configuration: epochs=5, imgsz=320, batch=16, fraction=0.211 (~1,500 images), freeze=10")
    print("=" * 70)

    start_train_time = time.time()
    model = YOLO(STARTING_WEIGHTS)

    # Fine-tune model
    model.train(
        data=ARCHIVE_YAML,
        epochs=5,
        imgsz=320,
        batch=16,
        fraction=0.211,
        freeze=10,
        device="cpu",
        workers=0,
        val=True,
        plots=False,
        verbose=True,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    train_duration = time.time() - start_train_time
    print(f"\nTraining completed in {train_duration:.1f} seconds ({train_duration / 60:.2f} minutes).")

    # Locate saved best weights
    best_weights_path = os.path.join(
        "runs", "detect", PROJECT_DIR, RUN_NAME, "weights", "best.pt"
    )
    if not os.path.isfile(best_weights_path):
        # Alternative path depending on ultralytics run directory resolution
        alt_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
        if os.path.isfile(alt_path):
            best_weights_path = alt_path
        else:
            raise FileNotFoundError(f"Could not locate saved best weights at {best_weights_path} or {alt_path}")

    print(f"Saved best weights: {best_weights_path}")

    # Evaluate on held-out test split (457 images)
    print("\nEvaluating new model on held-out TEST set (457 images)...")
    start_eval_time = time.time()
    eval_model = YOLO(best_weights_path)
    metrics = eval_model.val(
        data=ARCHIVE_YAML,
        split="test",
        device="cpu",
        workers=0,
        imgsz=320,
        batch=16,
        verbose=True,
    )
    eval_duration = time.time() - start_eval_time
    print(f"Evaluation completed in {eval_duration:.1f} seconds ({eval_duration / 60:.2f} minutes).")

    # Format and save evaluation results
    names = eval_model.names
    results = {
        "model_name": "YOLOv8s Fruit Detector (Moderate Fine-tune 1500 images, 5 epochs)",
        "weights_path": best_weights_path,
        "train_duration_seconds": train_duration,
        "eval_duration_seconds": eval_duration,
        "test_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision_mp": float(metrics.box.mp),
            "recall_mr": float(metrics.box.mr),
        },
        "per_class": {},
    }

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS (MODERATE FINE-TUNE):")
    print(f"Overall mAP50   : {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")
    print(f"Mean Precision  : {metrics.box.mp:.4f}")
    print(f"Mean Recall     : {metrics.box.mr:.4f}")
    print("-" * 70)

    for i, name in enumerate(names.values()):
        if i < len(metrics.box.ap50):
            p = float(metrics.box.p[i])
            r = float(metrics.box.r[i])
            ap50 = float(metrics.box.ap50[i])
            ap = float(metrics.box.ap[i])
            results["per_class"][name] = {
                "precision": p,
                "recall": r,
                "mAP50": ap50,
                "mAP50_95": ap,
            }
            print(f"{name:12}: P={p:.3f}, R={r:.3f}, mAP50={ap50:.3f}, mAP50-95={ap:.3f}")

    print("=" * 70)

    # Save summary json
    summary_path = os.path.join("runs", "detect", PROJECT_DIR, RUN_NAME, "test_evaluation.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation summary to: {summary_path}")

if __name__ == "__main__":
    main()
