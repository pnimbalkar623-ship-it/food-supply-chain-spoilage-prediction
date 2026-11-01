"""
test_predictions.py
-------------------
Unit tests for fruit detection, bounding box generation, count aggregation,
spoilage percentage calculations, and safety disclaimers.
"""

import unittest
from unittest.mock import MagicMock
from PIL import Image

from src.fruit_detector import (
    DetectionItem,
    DetectionResult,
    detect_fruits_batch,
    detect_fruits_in_image,
    draw_bounding_boxes,
    get_class_color,
    resolve_model_weights,
)
from src.fruit_recommendations import (
    get_recommendation_for_fruit,
    get_scientific_disclaimer,
)
from src.spoilage_assessment import evaluate_batch_spoilage_risk
from utils.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFERENCE_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    MAX_CONFIDENCE_THRESHOLD,
    MAX_IOU_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_IOU_THRESHOLD,
    SUPPORTED_INFERENCE_RESOLUTIONS,
)


class TestFruitDetector(unittest.TestCase):
    """Test suite for YOLO fruit detection pipeline and utilities."""

    def test_model_weights_resolution(self):
        """Ensure model weights resolution finds a valid weights file."""
        weights_path = resolve_model_weights()
        self.assertIsNotNone(weights_path)
        self.assertTrue(len(weights_path) > 0)

    def test_class_colors_consistency(self):
        """Verify defined colors exist for main fruit categories."""
        for fruit in ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]:
            color = get_class_color(fruit)
            self.assertTrue(color.startswith("#"))
            self.assertEqual(len(color), 7)

    def test_draw_bounding_boxes(self):
        """Ensure bounding boxes are drawn on an image without altering dimensions."""
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        items = [
            DetectionItem(class_name="Apple", confidence=0.88, bbox=[20.0, 20.0, 80.0, 80.0]),
            DetectionItem(class_name="Banana", confidence=0.92, bbox=[100.0, 50.0, 180.0, 150.0]),
        ]
        annotated = draw_bounding_boxes(img, items)
        self.assertEqual(annotated.size, (200, 200))
        self.assertEqual(annotated.mode, "RGB")

    def test_spoilage_percentage_unavailable_zero_items(self):
        """Verify handling of zero detections reports percentage as unavailable."""
        assessment = evaluate_batch_spoilage_risk(
            detected_counts={},
            temperature_c=25.0,
            humidity_pct=85.0,
        )
        self.assertEqual(assessment.total_items_detected, 0)
        self.assertIsNone(assessment.spoilage_percentage)
        self.assertFalse(assessment.is_available)
        self.assertIn("unavailable", assessment.formula_explanation.lower())

    def test_unsupported_categories_handled_transparently_not_mapped_to_orange(self):
        """Verify unsupported fruits are never mapped to substitute categories."""
        detected = {"Apple": 3, "Orange": 2}
        assessment = evaluate_batch_spoilage_risk(
            detected_counts=detected,
            temperature_c=22.0,
            humidity_pct=95.0,
        )
        self.assertEqual(assessment.total_items_detected, 5)
        self.assertEqual(assessment.unsupported_items_count, 3)
        self.assertIn("Apple", assessment.unsupported_categories)
        self.assertNotIn("Apple", assessment.category_predictions)
        self.assertIn("Orange", assessment.category_predictions)
        self.assertIsNone(assessment.spoilage_percentage)
        self.assertFalse(assessment.is_available)

    def test_prediction_failure_reported_as_incomplete_not_counted_safe(self):
        """Ensure prediction exceptions are never silently counted as safe."""
        faulty_artifact = {
            "model": None,
            "feature_columns": ["Fruit", "Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"],
            "classes": ["BAD", "Bad", "Good"],
            "input_schema": [],
        }
        assessment = evaluate_batch_spoilage_risk(
            detected_counts={"Banana": 4},
            temperature_c=22.0,
            humidity_pct=90.0,
            condition_artifact=faulty_artifact,
        )
        self.assertEqual(assessment.safe_items_count, 0)
        self.assertEqual(assessment.failed_items_count, 4)
        self.assertFalse(assessment.is_available)
        self.assertEqual(assessment.condition_class, "Assessment Incomplete")
        self.assertIn("Banana", assessment.failed_categories)

    def test_all_supported_categories_condition_assessment(self):
        """Verify evaluation of supported categories without claiming spoilage accuracy."""
        detected = {"Banana": 2, "Orange": 3}
        assessment = evaluate_batch_spoilage_risk(
            detected_counts=detected,
            temperature_c=22.0,
            humidity_pct=90.0,
        )
        self.assertEqual(assessment.total_items_detected, 5)
        self.assertEqual(assessment.unsupported_items_count, 0)
        self.assertTrue(assessment.is_available)
        self.assertIsNone(assessment.spoilage_percentage)
        self.assertIn("95.9% classification accuracy", assessment.assessment_notes)
        self.assertNotIn("95.9% spoilage-prediction accuracy", assessment.assessment_notes)

    def test_fruit_recommendations(self):
        """Verify storage recommendations exist for standard fruits."""
        for fruit in ["Apple", "Banana", "Orange", "Grape", "Pineapple", "Watermelon"]:
            rec = get_recommendation_for_fruit(fruit)
            self.assertIsNotNone(rec.optimal_temp)
            self.assertIsNotNone(rec.optimal_humidity)
            self.assertTrue(len(rec.storage_tips) > 0)

    def test_scientific_disclaimer(self):
        """Confirm presence of mandatory optical image limitation disclaimer."""
        disclaimer = get_scientific_disclaimer()
        self.assertIn("formalin", disclaimer.lower())
        self.assertIn("optical", disclaimer.lower())


    def test_default_confidence_threshold_configured(self):
        """Ensure default confidence threshold is 0.25 with sensible bounds and NMS/resolution defaults."""
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.25)
        self.assertEqual(MIN_CONFIDENCE_THRESHOLD, 0.05)
        self.assertEqual(MAX_CONFIDENCE_THRESHOLD, 0.95)
        self.assertEqual(DEFAULT_IOU_THRESHOLD, 0.55)
        self.assertEqual(MIN_IOU_THRESHOLD, 0.10)
        self.assertEqual(MAX_IOU_THRESHOLD, 0.95)
        self.assertEqual(DEFAULT_INFERENCE_IMGSZ, 320)
        self.assertEqual(SUPPORTED_INFERENCE_RESOLUTIONS, [320, 480, 640])

    def test_confidence_threshold_passed_to_yolo_predict(self):
        """Verify confidence threshold, IoU threshold, and imgsz are passed to model.predict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))

        detect_fruits_in_image(
            image=img,
            model=mock_model,
            image_name="test_img.jpg",
            confidence_threshold=0.25,
        )

        mock_model.predict.assert_called_once()
        _, kwargs = mock_model.predict.call_args
        self.assertEqual(kwargs.get("conf"), 0.25)
        self.assertEqual(kwargs.get("iou"), 0.55)
        self.assertEqual(kwargs.get("imgsz"), 320)

    def test_custom_iou_and_imgsz_passed_to_predict(self):
        """Verify custom IoU threshold and inference image size are correctly forwarded."""
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))

        detect_fruits_in_image(
            image=img,
            model=mock_model,
            image_name="test_img.jpg",
            confidence_threshold=0.20,
            iou_threshold=0.65,
            imgsz=480,
        )

        mock_model.predict.assert_called_once()
        _, kwargs = mock_model.predict.call_args
        self.assertEqual(kwargs.get("conf"), 0.20)
        self.assertEqual(kwargs.get("iou"), 0.65)
        self.assertEqual(kwargs.get("imgsz"), 480)

    def test_detect_fruits_backward_compatible_signatures(self):
        """Ensure calling detect_fruits without optional parameters uses configured defaults."""
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))

        # Single image call with minimum positional args
        res_single = detect_fruits_in_image(img, mock_model)
        self.assertIsNone(res_single.error)
        mock_model.predict.assert_called_once()
        _, kwargs = mock_model.predict.call_args
        self.assertEqual(kwargs.get("conf"), 0.25)
        self.assertEqual(kwargs.get("iou"), 0.55)
        self.assertEqual(kwargs.get("imgsz"), 320)

        # Batch call with default parameters
        mock_model.reset_mock()
        res_batch = detect_fruits_batch([("batch1.jpg", img)], mock_model)
        self.assertEqual(len(res_batch), 1)
        mock_model.predict.assert_called_once()
        _, batch_kwargs = mock_model.predict.call_args
        self.assertEqual(batch_kwargs.get("conf"), 0.25)
        self.assertEqual(batch_kwargs.get("iou"), 0.55)
        self.assertEqual(batch_kwargs.get("imgsz"), 320)

    def test_detect_fruits_in_image_handles_none_or_empty_image(self):
        """Verify graceful error reporting when image input is None."""
        mock_model = MagicMock()
        result = detect_fruits_in_image(None, mock_model, image_name="invalid.jpg")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.total_count, 0)
        self.assertEqual(len(result.items), 0)

    def test_detect_fruits_in_image_processes_camera_captured_image(self):
        """Verify processing a simulated phone-camera captured image produces valid result."""
        mock_model = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 0)]  # Apple
        mock_box.conf = [MagicMock(item=lambda: 0.88)]
        mock_box.xyxy = [[
            MagicMock(item=lambda: 10.0),
            MagicMock(item=lambda: 10.0),
            MagicMock(item=lambda: 50.0),
            MagicMock(item=lambda: 50.0),
        ]]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.names = {0: "Apple"}
        mock_model.predict.return_value = [mock_result]

        cam_img = Image.new("RGB", (128, 128), color=(255, 200, 200))
        res = detect_fruits_in_image(
            image=cam_img,
            model=mock_model,
            image_name="Camera Photo (camera_capture.jpg)",
            confidence_threshold=0.35,
        )

        self.assertIsNone(res.error)
        self.assertEqual(res.total_count, 1)
        self.assertEqual(res.counts.get("Apple"), 1)
        self.assertEqual(res.items[0].class_name, "Apple")
        self.assertEqual(res.annotated_image.size, (128, 128))


if __name__ == "__main__":
    unittest.main()
