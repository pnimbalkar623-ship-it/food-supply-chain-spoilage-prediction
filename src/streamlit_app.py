"""Shared data loading and page rendering for the Streamlit application."""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.prediction import (
    load_condition_model,
    load_image_model,
    predict_condition,
    predict_image,
)
from src.fruit_detector import (
    detect_fruits_in_image,
    load_yolo_model,
    resolve_model_weights,
)
from src.fruit_recommendations import (
    get_recommendation_for_fruit,
    get_scientific_disclaimer,
)
from src.spoilage_assessment import evaluate_batch_spoilage_risk
from utils.config import (
    ARCHIVE_PRETRAINED_PT,
    CONDITION_CSV,
    CONDITION_MODEL_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFERENCE_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    IMAGE_DATA_DIR,
    IMAGE_MODEL_DIR,
    MAX_CONFIDENCE_THRESHOLD,
    MAX_IOU_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_IOU_THRESHOLD,
    PROJECT_ROOT,
    SUPPORTED_INFERENCE_RESOLUTIONS,
    YOLO_BACKUP_WEIGHTS,
    YOLO_IMPROVED_WEIGHTS,
    YOLO_MODERATE_WEIGHTS,
    YOLO_TRIAL_WEIGHTS,
)

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MODEL_EXTENSIONS = {".h5", ".joblib", ".keras", ".model", ".onnx", ".pkl", ".pickle", ".pt", ".pth", ".sav"}
EVALUATION_FILENAME = "evaluation_results.json"
DISPLAY_METRICS = {
    "accuracy",
    "precision",
    "precision_macro",
    "precision_weighted",
    "recall",
    "recall_macro",
    "recall_weighted",
    "f1",
    "f1_score",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "auc",
}


def apply_page_style() -> None:
    """Apply a light-green accent theme while keeping the page background white."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #1f2933;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #174c2b !important;
        }
        .stApp p, .stApp li, .stApp label,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"] {
            color: #26352c;
        }
        [data-testid="stSidebar"] {
            background-color: #f1f8f2;
            color: #1f2933;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #174c2b !important;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.35rem;
        }
        [data-testid="stSidebarNav"] a {
            background: #ffffff;
            border: 1px solid #dcebdd;
            border-radius: 0.6rem;
            margin: 0.2rem 0;
            color: #20352a !important;
        }
        [data-testid="stSidebarNav"] a p,
        [data-testid="stSidebarNav"] a span {
            color: #20352a !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e1f1e3;
            border-color: #88bd91;
            color: #174c2b !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] p,
        [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: #174c2b !important;
        }
        div[data-testid="stMetric"] {
            background: #f3f8f3;
            border: 1px solid #d7e7d9;
            border-radius: 0.75rem;
            padding: 0.8rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #34483a !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #173d25 !important;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #e4f3e6;
            border: 1px solid #9bcaa1;
            color: #245b32;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_condition_data(csv_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load the condition CSV, returning a readable error when it is unavailable."""
    try:
        return pd.read_csv(csv_path), None
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        return None, str(error)


@st.cache_data(show_spinner=False)
def scan_image_folders(image_root: str) -> tuple[pd.DataFrame, list[str]]:
    """Count supported image files in each folder that directly contains images."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    if not os.path.isdir(image_root):
        return pd.DataFrame(columns=["Folder", "Food category", "Food class", "Label", "Images"]), errors

    def record_walk_error(error: OSError) -> None:
        errors.append(str(error))

    for current_root, _, filenames in os.walk(image_root, onerror=record_walk_error):
        image_count = sum(
            Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in filenames
        )
        if not image_count:
            continue

        relative_parts = Path(current_root).relative_to(image_root).parts
        rows.append(
            {
                "Folder": " / ".join(relative_parts) or ".",
                "Food category": relative_parts[0] if relative_parts else "(root)",
                "Food class": relative_parts[1] if len(relative_parts) > 1 else "(not specified)",
                "Label": relative_parts[2] if len(relative_parts) > 2 else "(not specified)",
                "Images": image_count,
            }
        )

    folder_counts = pd.DataFrame(
        rows,
        columns=["Folder", "Food category", "Food class", "Label", "Images"],
    ).sort_values("Folder", ignore_index=True)
    return folder_counts, errors


def find_model_artifacts(model_directory: str) -> list[str]:
    """List likely model files without loading or executing them."""
    if not os.path.isdir(model_directory):
        return []

    matches: list[str] = []
    for current_root, _, filenames in os.walk(model_directory):
        matches.extend(
            os.path.join(current_root, filename)
            for filename in filenames
            if Path(filename).suffix.lower() in MODEL_EXTENSIONS
        )
    return sorted(matches)


def load_evaluation_document(
    model_directory: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load a model's saved evaluation report, if it exists."""
    results_path = os.path.join(model_directory, EVALUATION_FILENAME)
    if not os.path.isfile(results_path):
        return None, None
    try:
        modified_time = os.path.getmtime(results_path)
    except OSError as error:
        return None, f"Could not inspect {results_path}: {error}"
    return _read_evaluation_document(results_path, modified_time)


@st.cache_data(show_spinner=False)
def _read_evaluation_document(
    results_path: str,
    modified_time: float,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if os.path.getmtime(results_path) != modified_time:
            return None, f"The evaluation report changed while it was being read: {results_path}"
        with open(results_path, encoding="utf-8") as results_file:
            results = json.load(results_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"Could not read {results_path}: {error}"

    if not isinstance(results, dict):
        return None, f"{results_path} must contain a JSON object of evaluation results."
    return results, None


def load_evaluation_results(model_directory: str) -> tuple[dict[str, float] | None, str | None]:
    """Read explicitly named evaluation output and keep recognized numeric metrics only."""
    results, error = load_evaluation_document(model_directory)
    if error or results is None:
        return None, error

    source = results.get("metrics", results)
    if not isinstance(source, dict):
        results_path = os.path.join(model_directory, EVALUATION_FILENAME)
        return None, f"The 'metrics' value in {results_path} must be a JSON object."

    metrics = {
        str(name): float(value)
        for name, value in source.items()
        if str(name).lower() in DISPLAY_METRICS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    if not metrics:
        return None, f"No recognized numeric evaluation metrics were found in {results_path}."
    return metrics, None


def _show_data_error(error: str | None) -> None:
    if error:
        st.error(f"The condition dataset could not be read: {error}")
    else:
        st.warning(
            "Condition dataset not found. Add Dataset.csv to data/condition_data/ "
            "to enable dataset exploration."
        )


def _get_data() -> pd.DataFrame | None:
    data, error = load_condition_data(CONDITION_CSV)
    if error:
        _show_data_error(error)
    elif data is None:
        _show_data_error(None)
    return data


def _get_image_counts() -> tuple[pd.DataFrame, list[str]]:
    return scan_image_folders(IMAGE_DATA_DIR)


def _model_status(model_directory: str) -> tuple[str, list[str]]:
    artifacts = find_model_artifacts(model_directory)
    if artifacts:
        return "Trained model artifact found.", artifacts
    return "Not trained yet — no model artifact found.", artifacts


def _show_image_scan_warnings(errors: list[str]) -> None:
    for error in errors:
        st.warning(f"Some image folders could not be scanned: {error}")


def render_home() -> None:
    st.title("Food Supply Chain Spoilage Prediction")
    st.write(
        "Explore the shipment condition dataset and analyze food produce. Users can upload "
        "food images or capture photos with their device camera for spoilage-related visual "
        "analysis, fruit detection, and postharvest storage recommendations using the saved model."
    )

    data, data_error = load_condition_data(CONDITION_CSV)
    if data_error:
        _show_data_error(data_error)
    elif data is None:
        _show_data_error(None)

    condition_rows = len(data) if data is not None else 0
    condition_columns = len(data.columns) if data is not None else 0

    col1, col2 = st.columns(2)
    col1.metric("Condition records", f"{condition_rows:,}")
    col2.metric("Condition fields", condition_columns)

    st.info(
        "📷 **Food Image Analysis:** You can upload food images or use your device camera "
        "for spoilage-related visual analysis, fruit detection, and item counting using the saved model. "
        "Navigate to **Food Image Analysis** in the sidebar to get started."
    )

    st.caption(
        "Prediction outputs use the dataset's original labels. They are model estimates, "
        "not a substitute for food-safety inspection or shipment-quality controls."
    )


@st.cache_resource(show_spinner=False)
def _get_cached_yolo_model(weights_path: str | None = None) -> Any:
    """Load and cache the YOLO model once in memory."""
    return load_yolo_model(weights_path)


def _render_legacy_classifier_expander() -> None:
    """Preserve existing whole-image baseline classifier in a transparent expander."""
    with st.expander("🔬 Legacy Whole-Image Color Baseline (Experimental SGD)", expanded=False):
        st.markdown(
            "This project includes an experimental whole-image color-feature baseline "
            "(`SGDClassifier` trained on 12×12 downscaled thumbnails + 16-bin color histograms). "
            "**Evaluated Accuracy:** ~34.3% on held-out food types. As documented, "
            "optical color heuristics cannot reliably localize individual items or confirm chemical safety."
        )
        _, artifacts = _model_status(IMAGE_MODEL_DIR)
        if artifacts:
            image_report, _ = load_evaluation_document(IMAGE_MODEL_DIR)
            if image_report:
                st.json(image_report.get("metrics", {}))
        else:
            st.info("Legacy image baseline artifact not present.")


def render_food_image_analysis() -> None:
    st.title("🍎 Multi-Fruit Detection & Spoilage Prediction")
    st.markdown(
        "Upload single or multiple food photographs to detect individual fruits with **YOLOv8 bounding boxes**, "
        "count fruit categories, view shipment condition model classifications under simulated environmental parameters, "
        "and generate postharvest storage recommendations."
    )

    with st.expander("ℹ️ Scientific Notice & Visual Inspection Limitations", expanded=False):
        st.warning(get_scientific_disclaimer())
        st.caption(
            "Visual inspection detects surface boundaries, produce varieties, and visible decay signs. "
            "Environmental assessments use the trained condition model (which achieved 95.9% classification "
            "accuracy on source dataset labels 'BAD', 'Bad', and 'Good'). A validated spoilage-prediction "
            "accuracy or risk percentage is not available because the dataset source does not define the empirical "
            "or clinical spoilage meaning of those class labels."
        )

    # ── Sidebar / Setup Controls ─────────────────────────────────────────────
    st.subheader("1. Detection & Storage Parameters")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1])

    with ctrl_col1:
        confidence_thresh = st.slider(
            "YOLO Detection Confidence Threshold",
            min_value=MIN_CONFIDENCE_THRESHOLD,
            max_value=MAX_CONFIDENCE_THRESHOLD,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help=(
                "Adjustable confidence threshold (0.05 to 0.95, default 0.25). "
                "Lowering the threshold can increase detections of individual fruits (higher recall) "
                "but may also increase false positives. Raising it reduces false alarms but may miss more fruits."
            ),
        )
        st.caption(
            f"**Active Threshold: `{confidence_thresh:.2f}`** — "
            "Lower threshold increases fruit recall; higher threshold improves precision."
        )

        iou_thresh = st.slider(
            "YOLO NMS IoU Overlap Threshold",
            min_value=MIN_IOU_THRESHOLD,
            max_value=MAX_IOU_THRESHOLD,
            value=DEFAULT_IOU_THRESHOLD,
            step=0.05,
            help=(
                "Non-Maximum Suppression (NMS) IoU threshold (0.10 to 0.95, default 0.55). "
                "Higher values allow overlapping, touching, or clustered fruits to be detected together."
            ),
        )
        st.caption(
            f"**Active IoU Threshold: `{iou_thresh:.2f}`** — "
            "Higher threshold prevents suppression of clustered fruits."
        )

    with ctrl_col2:
        inference_resolution = st.selectbox(
            "Inference Resolution (Image Size)",
            options=SUPPORTED_INFERENCE_RESOLUTIONS,
            index=SUPPORTED_INFERENCE_RESOLUTIONS.index(DEFAULT_INFERENCE_IMGSZ) if DEFAULT_INFERENCE_IMGSZ in SUPPORTED_INFERENCE_RESOLUTIONS else 0,
            format_func=lambda x: f"{x}×{x} px" + (" (Native Speed)" if x == 320 else (" (Balanced)" if x == 480 else " (High Detail)")),
            help=(
                "Select image resolution fed into the YOLO detector. "
                "Higher resolutions (480px, 640px) improve small and clustered fruit detection in bulk images."
            ),
        )
        st.caption(f"**Inference Resolution: `{inference_resolution}×{inference_resolution} px`**")

        weights_options = [
            "Promoted Moderate Model (1,500 samples, mAP50 33.4%)",
            "Previous Improved Model (426 samples, mAP50 23.7%)",
            "Baseline Trial Weights (7 samples)",
            "Pretrained YOLOv8s (COCO starting weights)",
        ]
        default_idx = (
            0 if os.path.isfile(YOLO_MODERATE_WEIGHTS)
            else (1 if os.path.isfile(YOLO_IMPROVED_WEIGHTS)
            else (2 if os.path.isfile(YOLO_TRIAL_WEIGHTS) else 3))
        )
        weights_choice = st.radio(
            "Detection Weights",
            options=weights_options,
            index=default_idx,
            help="Choose between fine-tuned 6-class fruit detector models or baseline starting weights.",
        )
        if weights_choice.startswith("Promoted") and os.path.isfile(YOLO_MODERATE_WEIGHTS):
            selected_weights = YOLO_MODERATE_WEIGHTS
        elif weights_choice.startswith("Previous") and os.path.isfile(YOLO_IMPROVED_WEIGHTS):
            selected_weights = YOLO_IMPROVED_WEIGHTS
        elif weights_choice.startswith("Baseline") and os.path.isfile(YOLO_TRIAL_WEIGHTS):
            selected_weights = YOLO_TRIAL_WEIGHTS
        else:
            selected_weights = ARCHIVE_PRETRAINED_PT

    with ctrl_col3:
        st.caption("Active Model:")
        if os.path.isfile(selected_weights):
            st.success(f"Loaded: `{os.path.basename(selected_weights)}`")
        else:
            st.error("Weights file missing.")

    # ── Shipment / Storage Environmental Simulation ──────────────────────────
    with st.expander("🚚 Shipment / Warehouse Environmental Conditions", expanded=True):
        st.caption("Adjust simulated storage sensor parameters to evaluate batch spoilage probability.")
        env_col1, env_col2, env_col3, env_col4 = st.columns(4)
        with env_col1:
            temp_input = st.number_input("Temperature (°C)", min_value=15.0, max_value=35.0, value=24.0, step=0.5)
        with env_col2:
            humid_input = st.number_input("Humidity (% RH)", min_value=60.0, max_value=99.0, value=92.0, step=1.0)
        with env_col3:
            co2_input = st.number_input("CO2 (ppm)", min_value=200.0, max_value=500.0, value=370.0, step=10.0)
        with env_col4:
            light_input = st.number_input("Light (Lux)", min_value=0.0, max_value=25.0, value=12.0, step=1.0)

    # ── Image Acquisition: Upload & Mobile Camera Capture ────────────────────
    st.subheader("2. Image Acquisition & Analysis")
    st.caption("Upload produce photographs from your device or capture a live still photograph using your mobile phone camera.")

    upload_col, camera_col = st.columns([1.2, 1.0])

    with upload_col:
        st.markdown("**📁 Batch / Single Image Upload**")
        uploaded_files = st.file_uploader(
            "Choose one or multiple food images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, PNG. Upload multiple images to analyze in batch.",
        )

    with camera_col:
        st.markdown("**📱 Mobile Phone Camera Capture**")
        camera_file = st.camera_input(
            "Take a photo with your device camera",
            help=(
                "Capture a live still photograph directly with your mobile or desktop camera. "
                "Requires browser camera permission (HTTPS connection required for remote mobile browsers)."
            ),
        )

    with st.expander("🧪 Or test with built-in demo produce samples", expanded=False):
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            load_demo1 = st.button("🍏 Load Sample 1 (Apples)")
        with d_col2:
            load_demo2 = st.button("🥦 Load Sample 2 (Produce)")

    # Collect images to process: list of (filename, PIL.Image)
    images_to_process: list[tuple[str, Image.Image]] = []
    seen_hashes: set[str] = set()

    def _process_and_add_image(name: str, raw_bytes: bytes) -> bool:
        """Validates and adds an image if unique by SHA-256 hash, avoiding duplicate counting."""
        if not raw_bytes:
            return False
        import hashlib
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest in seen_hashes:
            return False
        seen_hashes.add(digest)
        try:
            with Image.open(BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(BytesIO(raw_bytes)) as valid_img:
                images_to_process.append((name, valid_img.copy()))
            return True
        except (UnidentifiedImageError, OSError, ValueError) as err:
            st.error(f"Could not open '{name}' as a valid image: {err}")
            return False

    # 1. Process uploaded files
    if uploaded_files:
        file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        for u_file in file_list:
            try:
                _process_and_add_image(u_file.name, u_file.getvalue())
            except Exception as err:
                st.error(f"Error reading '{u_file.name}': {err}")

    # 2. Process camera-captured file
    if camera_file is not None:
        try:
            cam_bytes = camera_file.getvalue()
            cam_name = getattr(camera_file, "name", None) or "camera_capture.jpg"
            display_name = f"Camera Photo ({cam_name})"
            added = _process_and_add_image(display_name, cam_bytes)
            if not added and cam_bytes:
                st.info("Camera image was already uploaded or processed; duplicate was skipped.")
        except Exception as err:
            st.error(f"Error reading camera capture: {err}")

    # 3. Process demo samples if neither upload nor camera was provided
    if not images_to_process:
        if load_demo1:
            sample_path = os.path.join(PROJECT_ROOT, "images.jpg")
            if os.path.isfile(sample_path):
                with open(sample_path, "rb") as sf:
                    _process_and_add_image("images.jpg (Demo Apples)", sf.read())

        elif load_demo2:
            sample_path = os.path.join(PROJECT_ROOT, "images (1).jpg")
            if os.path.isfile(sample_path):
                with open(sample_path, "rb") as sf:
                    _process_and_add_image("images (1).jpg (Demo Produce)", sf.read())

    if not images_to_process:
        st.info("Upload one or more food images above, capture a photo using your mobile camera, or click a demo sample to begin analysis.")
        _render_legacy_classifier_expander()
        return

    # ── Run Detection ────────────────────────────────────────────────────────
    try:
        detector_model = _get_cached_yolo_model(selected_weights)
    except Exception as exc:
        st.error(f"Failed to load YOLO detection model: {exc}")
        return

    if detector_model is None:
        st.error("Failed to load YOLO detection model: Model could not be initialized.")
        return

    st.markdown("---")
    st.subheader("3. Detection Results by Image")

    batch_detection_results = []
    overall_category_counts: dict[str, int] = {}

    for idx, (img_name, pil_img) in enumerate(images_to_process, start=1):
        det_result = detect_fruits_in_image(
            image=pil_img,
            model=detector_model,
            image_name=img_name,
            confidence_threshold=confidence_thresh,
            iou_threshold=iou_thresh,
            imgsz=inference_resolution,
        )
        batch_detection_results.append(det_result)

        for cat, cnt in det_result.counts.items():
            overall_category_counts[cat] = overall_category_counts.get(cat, 0) + cnt

        with st.expander(f"📷 Image {idx}: {img_name} ({det_result.total_count} fruits detected)", expanded=True):
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.markdown("**Original Image**")
                st.image(det_result.original_image, width="stretch")
                st.caption(f"Dimensions: {pil_img.width} x {pil_img.height} px")

            with img_c2:
                st.markdown("**Annotated Bounding Boxes**")
                st.image(det_result.annotated_image, width="stretch")

            if det_result.error:
                st.error(det_result.error)
            elif det_result.items:
                st.markdown("**Detected Items Breakdown:**")
                chips = " &nbsp; | &nbsp; ".join(
                    [f"**{cat}**: {cnt}" for cat, cnt in det_result.counts.items()]
                )
                st.markdown(f"🏷️ {chips}")

                detail_rows = [
                    {
                        "Item #": item_idx + 1,
                        "Category": item.class_name,
                        "Confidence": f"{item.confidence:.1%}",
                        "Bounding Box [x1, y1, x2, y2]": f"[{int(item.bbox[0])}, {int(item.bbox[1])}, {int(item.bbox[2])}, {int(item.bbox[3])}]",
                    }
                    for item_idx, item in enumerate(det_result.items)
                ]
                st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
            else:
                st.info("No fruits were recognized above the current confidence threshold in this image.")

    # ── Batch Summary Dashboard & Spoilage Assessment ────────────────────────
    st.markdown("---")
    st.subheader("4. Batch Summary & Spoilage Risk Assessment")

    total_all_fruits = sum(overall_category_counts.values())

    assessment = evaluate_batch_spoilage_risk(
        detected_counts=overall_category_counts,
        temperature_c=temp_input,
        humidity_pct=humid_input,
        light_lux=light_input,
        co2_ppm=co2_input,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Images", len(images_to_process))
    with m2:
        st.metric("Total Fruits Detected", total_all_fruits)
    with m3:
        st.metric("Condition Model Status", assessment.condition_class)
    with m4:
        st.metric(
            "Validated Spoilage Risk",
            "Not Available*",
            help="A validated numerical spoilage-risk percentage is unavailable because the condition dataset's class meanings are not established.",
        )

    # Honest disclosure notice regarding unavailable spoilage risk score
    st.info(
        "ℹ️ **Notice: Validated Spoilage Risk Percentage is Unavailable**\n\n"
        "A validated spoilage-risk percentage is unavailable because the source condition dataset's "
        "class meanings ('BAD', 'Bad', and 'Good') are not established as verified spoilage definitions. "
        "The condition model provides dataset label classification (95.9% test accuracy on dataset labels), "
        "not calibrated spoilage probabilities or validated risk scores.\n\n"
        f"*{assessment.assessment_notes}*"
    )

    if assessment.unsupported_categories:
        st.warning(
            f"⚠️ **Unsupported Produce Categories ({', '.join(sorted(assessment.unsupported_categories))}):** "
            "These fruits are not present in the condition dataset schema (which supports Banana, Orange, Pineapple, Tomato). "
            "They are reported transparently and are not mapped to substitute categories."
        )

    if assessment.failed_categories:
        st.error(
            f"❌ **Prediction Failures ({', '.join(sorted(assessment.failed_categories))}):** "
            "Condition predictions encountered errors and are reported as incomplete rather than assumed safe."
        )

    if assessment.category_predictions:
        with st.expander("📊 Condition Model Classifications by Supported Category", expanded=False):
            for fruit_k, pred_v in sorted(assessment.category_predictions.items()):
                st.write(f"- **{fruit_k}**: Dataset label `{pred_v}` under current environmental inputs")

    # ── Produce Storage Recommendations ──────────────────────────────────────
    st.subheader("5. Produce-Specific Storage Recommendations")
    if overall_category_counts:
        for fruit_cat in sorted(overall_category_counts.keys()):
            rec = get_recommendation_for_fruit(fruit_cat)
            with st.container():
                st.markdown(f"### 📦 Storage Protocol: **{rec.fruit_name}** ({overall_category_counts[fruit_cat]} detected)")
                r_c1, r_c2, r_c3 = st.columns([1, 1, 1.2])
                with r_c1:
                    st.markdown(f"**Optimal Temperature:**\n`{rec.optimal_temp}`")
                with r_c2:
                    st.markdown(f"**Optimal Humidity:**\n`{rec.optimal_humidity}`")
                with r_c3:
                    st.markdown(f"**Ethylene Characteristics:**\n{rec.ethylene_behavior}")

                st.markdown("**Key Handling & Spoilage Prevention Tips:**")
                for tip in rec.storage_tips:
                    st.markdown(f"- {tip}")
                st.caption(f"Estimated Safe Shelf Life: {rec.max_shelf_life}")
                st.divider()
    else:
        st.write("No specific fruits detected to display custom storage protocols.")

    # ── Legacy Whole-Image Classifier Baseline ───────────────────────────────
    st.markdown("---")
    st.subheader("6. Whole-Image Color Baseline (Experimental)")
    st.caption(
        "Single whole-image baseline classifier (SGDClassifier). Uncalibrated probabilities "
        "are displayed below as experimental visual indicators."
    )
    first_image = images_to_process[0][1]
    _, artifacts = _model_status(IMAGE_MODEL_DIR)
    if artifacts:
        try:
            model_artifact = _load_image_model(
                IMAGE_MODEL_DIR,
                _get_model_modified_time(IMAGE_MODEL_DIR, "image_model.joblib"),
            )
            prediction, probabilities = predict_image(first_image, model_artifact)
            st.success(f"Predicted folder label: **{prediction}**")
            st.caption(
                "The prediction uses the label text from the image folders; "
                "the label meanings are not reinterpreted. The model's class "
                "probabilities are uncalibrated and are not validated confidence "
                "or food-safety risk."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Folder label": label,
                            "Uncalibrated model probability": f"{probability:.1%}",
                        }
                        for label, probability in sorted(
                            probabilities.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        except Exception as error:
            st.error(f"Legacy image model could not be used: {error}")
    else:
        _render_legacy_classifier_expander()


@st.cache_resource(show_spinner=False)
def _load_condition_model(model_directory: str, modified_time: float) -> dict[str, Any]:
    model_path = os.path.join(model_directory, "condition_model.joblib")
    if os.path.getmtime(model_path) != modified_time:
        raise OSError(f"The condition model changed while it was being loaded: {model_path}")
    return load_condition_model(model_directory)


@st.cache_resource(show_spinner=False)
def _load_image_model(model_directory: str, modified_time: float) -> dict[str, Any]:
    model_path = os.path.join(model_directory, "image_model.joblib")
    if os.path.getmtime(model_path) != modified_time:
        raise OSError(f"The image model changed while it was being loaded: {model_path}")
    return load_image_model(model_directory)


def _get_model_modified_time(model_directory: str, model_filename: str) -> float:
    return os.path.getmtime(os.path.join(model_directory, model_filename))


def render_shipment_risk_analysis() -> None:
    st.title("Shipment Risk Analysis")
    st.write("The fields below come directly from the condition dataset.")
    data = _get_data()
    if data is not None:
        target_column = "Class"
        input_columns = [column for column in data.columns if column != target_column]
        if input_columns:
            summary = pd.DataFrame(
                {
                    "Field": input_columns,
                    "Data type": [str(data[column].dtype) for column in input_columns],
                    "Missing values": [int(data[column].isna().sum()) for column in input_columns],
                    "Example values": [
                        ", ".join(data[column].dropna().astype(str).unique()[:3])
                        for column in input_columns
                    ],
                }
            )
            st.dataframe(summary, width="stretch", hide_index=True)
        else:
            st.warning("No input fields are available in the current CSV.")

        if target_column in data.columns:
            st.caption(f"'{target_column}' is the dataset label, not an input field.")
        else:
            st.caption("No 'Class' label column was found; all CSV columns are listed as inputs.")

    _, artifacts = _model_status(CONDITION_MODEL_DIR)
    if artifacts:
        try:
            model_artifact = _load_condition_model(
                CONDITION_MODEL_DIR,
                _get_model_modified_time(
                    CONDITION_MODEL_DIR,
                    "condition_model.joblib",
                ),
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"The saved condition model could not be loaded: {error}")
            return

        if data is not None:
            dataset_features = [column for column in data.columns if column != "Class"]
            if dataset_features != model_artifact["feature_columns"]:
                st.error(
                    "The current CSV inputs do not match the trained model's feature "
                    "columns. Retrain the condition model before making predictions."
                )
                return

        class_labels = ", ".join(model_artifact.get("classes", []))
        st.subheader("Predict the dataset class")
        st.caption(
            "Inputs and permitted ranges come from the trained model's dataset. "
            "The source CSV does not document measurement units; values are shown "
            "exactly in its recorded scale."
        )
        st.info(
            "**Validated spoilage risk percentage: not available.** "
            f"The model predicts the dataset labels ({class_labels}), but the source "
            "does not define their spoilage meaning. The model probabilities have not "
            "been calibrated, so they cannot be presented as spoilage risk. "
            "Validated Low/Medium/High thresholds are also unavailable."
        )
        with st.form("shipment_condition_prediction"):
            inputs: dict[str, Any] = {}
            for field in model_artifact["input_schema"]:
                name = field["name"]
                if field["kind"] == "categorical":
                    options = field["options"]
                    if not options:
                        st.error(f"No trained options are available for {name}.")
                        continue
                    inputs[name] = st.selectbox(
                        name,
                        options,
                        index=options.index(field["default"])
                        if field["default"] in options
                        else 0,
                        help=f"Values observed in the training data: {', '.join(options)}.",
                    )
                elif field["integer"]:
                    range_help = (
                        f"Observed training-data range: {field['minimum']:g} to "
                        f"{field['maximum']:g}. Unit not documented by the source CSV."
                    )
                    inputs[name] = st.number_input(
                        name,
                        min_value=int(field["minimum"]),
                        max_value=int(field["maximum"]),
                        value=int(field["default"]),
                        step=1,
                        help=range_help,
                    )
                else:
                    range_help = (
                        f"Observed training-data range: {field['minimum']:g} to "
                        f"{field['maximum']:g}. Unit not documented by the source CSV."
                    )
                    numeric_range = field["maximum"] - field["minimum"]
                    inputs[name] = st.number_input(
                        name,
                        min_value=float(field["minimum"]),
                        max_value=float(field["maximum"]),
                        value=float(field["default"]),
                        step=max(numeric_range / 100.0, 0.01),
                        format="%.4f",
                        help=range_help,
                    )
            submitted = st.form_submit_button("Predict dataset class")

        if submitted:
            try:
                prediction = predict_condition(inputs, model_artifact)
            except (ValueError, KeyError, TypeError) as error:
                st.error(f"Could not generate a condition prediction: {error}")
            else:
                st.success(f"Predicted dataset class: **{prediction}**")
                st.caption(
                    "Low/Medium/High risk bands, per-factor contributions, and "
                    "handling recommendations are not provided because this model "
                    "and dataset do not establish causal contributions or validated "
                    "interventions."
                )
    else:
        st.info(
            "Predictions are not ready because no trained condition model exists yet. "
            "This page only inspects the available inputs and will not estimate spoilage risk."
        )


def render_dataset_explorer() -> None:
    st.title("Dataset Explorer")
    data = _get_data()

    if data is not None:
        st.subheader("Condition dataset")
        first, second, third = st.columns(3)
        first.metric("Rows", f"{len(data):,}")
        second.metric("Columns", len(data.columns))
        third.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

        st.markdown("**Columns**")
        st.write(", ".join(str(column) for column in data.columns))
        st.markdown("**Sample rows**")
        st.dataframe(data.head(10), width="stretch", hide_index=True)

        st.markdown("**Missing values by column**")
        missing_values = data.isna().sum().rename("Missing values").rename_axis("Column")
        st.dataframe(
            missing_values.reset_index(),
            width="stretch",
            hide_index=True,
        )

        if "Class" in data.columns:
            st.markdown("**Class distribution**")
            class_distribution = (
                data["Class"].fillna("(missing)").astype(str).value_counts().rename("Records")
            )
            st.bar_chart(class_distribution)
            st.dataframe(
                class_distribution.rename_axis("Class").reset_index(),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No 'Class' column is available to show a class distribution.")

    st.subheader("Food image folders")
    image_counts, scan_errors = _get_image_counts()
    _show_image_scan_warnings(scan_errors)
    if image_counts.empty:
        if not os.path.isdir(IMAGE_DATA_DIR):
            st.info(
                "The image-training folders are not included in this deployment. "
                "The saved image model can still classify uploaded JPG, JPEG, and PNG files."
            )
        else:
            st.info("No JPG, JPEG, or PNG images were found in the image folders.")
    else:
        category_counts = (
            image_counts.groupby("Food category", as_index=False)["Images"]
            .sum()
            .sort_values("Food category")
        )
        st.markdown("**Image totals by top-level category**")
        st.dataframe(category_counts, width="stretch", hide_index=True)
        st.markdown("**Counts by folder containing images**")
        st.dataframe(image_counts, width="stretch", hide_index=True)


def render_model_performance() -> None:
    st.title("Model Performance")
    st.write("Metrics are read only from saved test-set evaluation output; this page does not train models.")

    models = [
        ("Food image model", IMAGE_MODEL_DIR),
        ("Shipment condition model", CONDITION_MODEL_DIR),
    ]
    for model_name, model_directory in models:
        st.subheader(model_name)
        report, error = load_evaluation_document(model_directory)
        if error:
            st.error(error)
        elif report is None:
            st.info("Not evaluated yet.")
        else:
            metrics = report.get("metrics", {})
            if not isinstance(metrics, dict):
                st.error("The saved evaluation report has an invalid metrics section.")
                continue
            st.caption(
                f"{report.get('model_name', 'Model')} · "
                f"Target: {report.get('target_column', 'not specified')}"
            )
            st.dataframe(
                pd.DataFrame(
                    [{"Metric": name, "Value": f"{value:.4f}"} for name, value in metrics.items()]
                ),
                width="stretch",
                hide_index=True,
            )
            split = report.get("split", {})
            if isinstance(split, dict):
                split_details = [str(split.get("method", "Evaluation split"))]
                if "train_rows" in split and "test_rows" in split:
                    split_details.append(
                        f"train: {split['train_rows']:,}, test: {split['test_rows']:,}"
                    )
                elif "train_images" in split and "test_images" in split:
                    split_details.append(
                        f"train images: {split['train_images']:,}, "
                        f"test images: {split['test_images']:,}"
                    )
                if "actual_test_fraction" in split:
                    split_details.append(
                        f"test fraction: {split['actual_test_fraction']:.1%}"
                    )
                if "exact_duplicate_rows_removed_before_split" in split:
                    split_details.append(
                        "exact duplicates removed before split: "
                        f"{split['exact_duplicate_rows_removed_before_split']:,}"
                    )
                if "test_food_groups" in split:
                    split_details.append(
                        "held-out food groups: "
                        + ", ".join(str(group) for group in split["test_food_groups"])
                    )
                st.caption(" · ".join(split_details))
            confusion = report.get("confusion_matrix")
            if isinstance(confusion, dict):
                labels = confusion.get("labels")
                values = confusion.get("values")
                if (
                    isinstance(labels, list)
                    and isinstance(values, list)
                    and len(values) == len(labels)
                    and all(
                        isinstance(row, list) and len(row) == len(labels)
                        for row in values
                    )
                ):
                    st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
                    matrix = pd.DataFrame(
                        values,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    )
                    st.dataframe(matrix, width="stretch")
                else:
                    st.warning("The saved confusion matrix has an invalid shape.")
            st.caption(f"Source: {os.path.join(model_directory, EVALUATION_FILENAME)}")
