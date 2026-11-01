"""
spoilage_assessment.py
-----------------------
Bridges fruit object detection with verified supply-chain condition modeling.
Provides transparent condition evaluation while honestly disclosing that a
validated spoilage-risk percentage is unavailable because the condition dataset's
class meanings ('BAD', 'Bad', 'Good') are not established.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.prediction import load_condition_model, predict_condition


@dataclass
class SpoilageAssessment:
    """Summary of condition evaluation across detected fruits."""
    total_items_detected: int
    at_risk_items_count: int
    safe_items_count: int
    spoilage_percentage: float | None
    condition_class: str
    assessment_notes: str
    formula_explanation: str
    is_available: bool = True
    unsupported_items_count: int = 0
    unsupported_categories: list[str] = field(default_factory=list)
    failed_items_count: int = 0
    failed_categories: list[str] = field(default_factory=list)
    category_predictions: dict[str, str] = field(default_factory=dict)


def evaluate_batch_spoilage_risk(
    detected_counts: dict[str, int],
    temperature_c: float,
    humidity_pct: float,
    light_lux: float = 10.0,
    co2_ppm: float = 350.0,
    condition_artifact: dict[str, Any] | None = None,
) -> SpoilageAssessment:
    """
    Evaluate condition model predictions across detected fruit categories under
    specified environmental storage conditions.

    A validated spoilage-risk percentage is UNAVAILABLE because the source dataset
    does not establish the empirical or clinical spoilage meaning of its target
    labels ('BAD', 'Bad', 'Good').
    """
    total_detected = sum(detected_counts.values())

    # Safe zero-detection handling
    if total_detected == 0:
        return SpoilageAssessment(
            total_items_detected=0,
            at_risk_items_count=0,
            safe_items_count=0,
            spoilage_percentage=None,
            condition_class="None Detected",
            assessment_notes="No fruits were detected in the provided images.",
            formula_explanation=(
                "A validated spoilage-risk percentage is unavailable because the condition dataset's "
                "class meanings are not established. With 0 items detected, no condition evaluation was performed."
            ),
            is_available=False,
        )

    try:
        artifact = condition_artifact or load_condition_model()
    except Exception as exc:
        return SpoilageAssessment(
            total_items_detected=total_detected,
            at_risk_items_count=0,
            safe_items_count=0,
            spoilage_percentage=None,
            condition_class="Model Unavailable",
            assessment_notes=f"Condition model could not be loaded: {exc}",
            formula_explanation="Condition evaluation unavailable: model loading failed.",
            is_available=False,
            failed_items_count=total_detected,
        )

    # Valid fruit classes in Dataset.csv / condition model schema
    valid_schema_fruits = {"Banana", "Orange", "Pineapple", "Tomato"}

    at_risk_count = 0
    safe_count = 0
    unsupported_items_count = 0
    unsupported_categories: list[str] = []
    failed_items_count = 0
    failed_categories: list[str] = []
    category_predictions: dict[str, str] = {}

    for raw_fruit, count in detected_counts.items():
        fruit_name = str(raw_fruit).strip().capitalize()
        if fruit_name not in valid_schema_fruits:
            # Handle unsupported categories transparently: do NOT map to Orange!
            unsupported_categories.append(fruit_name)
            unsupported_items_count += count
            continue

        inputs = {
            "Fruit": fruit_name,
            "Temp": float(min(max(temperature_c, 15.0), 35.0)),
            "Humid (%)": float(min(max(humidity_pct, 60.0), 99.0)),
            "Light (Fux)": float(min(max(light_lux, 0.0), 25.0)),
            "CO2 (pmm)": float(min(max(co2_ppm, 200.0), 500.0)),
        }

        try:
            pred_class = predict_condition(inputs, artifact)
            category_predictions[fruit_name] = pred_class
            is_bad = str(pred_class).upper() == "BAD"
            if is_bad:
                at_risk_count += count
            else:
                safe_count += count
        except Exception:
            # Do NOT silently count prediction errors as safe!
            failed_categories.append(fruit_name)
            failed_items_count += count

    # Determine status and honesty notices
    if failed_items_count > 0:
        is_available = False
        condition_class = "Assessment Incomplete"
        notes = (
            f"Condition model prediction failed for {', '.join(failed_categories)}. "
            f"Prediction errors are reported directly and are not counted as safe."
        )
    elif unsupported_items_count == total_detected:
        is_available = False
        condition_class = "Unsupported Categories"
        notes = (
            f"None of the detected fruit categories ({', '.join(unsupported_categories)}) "
            f"are present in the condition dataset (supported: {', '.join(sorted(valid_schema_fruits))}). "
            "Unsupported categories are handled transparently and are not evaluated."
        )
    elif unsupported_items_count > 0:
        is_available = False
        dominant_eval = "Bad" if at_risk_count > 0 else "Good"
        condition_class = f"{dominant_eval} (Partial)"
        notes = (
            f"Partial assessment under {temperature_c}°C, {humidity_pct}% RH: "
            f"evaluated supported produce ({', '.join(category_predictions.keys())}) using condition model "
            f"(95.9% classification accuracy on dataset labels 'BAD'/'Bad'/'Good'). "
            f"Unsupported categories ({', '.join(unsupported_categories)}) were not evaluated."
        )
    else:
        is_available = True
        condition_class = "Bad" if at_risk_count > 0 else "Good"
        notes = (
            f"Assessed under environmental conditions: {temperature_c}°C, {humidity_pct}% RH, {co2_ppm} ppm CO2 "
            "using the condition classification model (95.9% classification accuracy on source dataset labels 'BAD'/'Bad'/'Good')."
        )

    formula_exp = (
        "A validated spoilage-risk percentage is unavailable because the condition dataset's "
        "class meanings ('BAD', 'Bad', 'Good') are not established as empirical or clinical spoilage definitions."
    )

    return SpoilageAssessment(
        total_items_detected=total_detected,
        at_risk_items_count=at_risk_count,
        safe_items_count=safe_count,
        spoilage_percentage=None,
        condition_class=condition_class,
        assessment_notes=notes,
        formula_explanation=formula_exp,
        is_available=is_available,
        unsupported_items_count=unsupported_items_count,
        unsupported_categories=unsupported_categories,
        failed_items_count=failed_items_count,
        failed_categories=failed_categories,
        category_predictions=category_predictions,
    )
