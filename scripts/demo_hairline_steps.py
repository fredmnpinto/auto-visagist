"""Demo script visualising each step of the hairline detection algorithm."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visagism.errors import ModelNotFoundError, VisagismError  # noqa: E402
from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.hairline_detector import HairlineDetector  # noqa: E402
from visagism.image_loader import ImageLoader  # noqa: E402
from visagism.landmark_detector import LandmarkDetector  # noqa: E402
from visagism.model_finder import ModelFinder  # noqa: E402
from visagism.types import FacialLandmarks, ImageArray  # noqa: E402


def _get_default_image_path() -> Path:
    """Return the first image found in ``test_images/``."""
    test_images_dir = _PROJECT_ROOT / "test_images"
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        candidates = sorted(test_images_dir.glob(ext))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No test images found in {test_images_dir}")


def _show_step(window_name: str, image: np.ndarray) -> None:
    """Display a single step window and wait for a key press."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _draw_graph(
    values: np.ndarray, title: str, y_max: float | None = None,
    highlight_idx: int | None = None,
) -> np.ndarray:
    """Draw a static line graph on a 600x300 dark gray canvas."""
    canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
    cv2.line(canvas, (50, 250), (580, 250), (180, 180, 180), 1)
    cv2.line(canvas, (50, 20), (50, 250), (180, 180, 180), 1)
    n = len(values)
    if n == 0:
        return canvas
    vmax = y_max if y_max is not None else float(np.max(values))
    vmax = vmax or 1.0
    points = [(50 + int((i / (n - 1)) * 530) if n > 1 else 315,
               250 - int((val / vmax) * 220)) for i, val in enumerate(values)]
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], (0, 200, 255), 2)
    if highlight_idx is not None and 0 <= highlight_idx < n:
        hx = points[highlight_idx][0]
        cv2.line(canvas, (hx, 20), (hx, 250), (0, 0, 255), 2)
    cv2.putText(canvas, title, (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    return canvas


def _create_step1_image(
    img_bgr: ImageArray, landmarks: FacialLandmarks,
    xs: int, xe: int, ys: int, ye: int,
) -> np.ndarray:
    """Create Step 1 image: original with face box, eyebrow line, and ROI."""
    img = img_bgr.copy()
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    brows = (landmarks.landmarks_by_region["left_eyebrow"]
             + landmarks.landmarks_by_region["right_eyebrow"])
    avg_y = int(np.mean([pt[1] for pt in brows]))
    cv2.line(img, (xs, avg_y), (xe, avg_y), (0, 0, 255), 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (xs, ys), (xe, ye), (255, 200, 100), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.rectangle(img, (xs, ys), (xe, ye), (255, 255, 0), 2)

    cv2.putText(img, "Face box", (fx + fw + 10, fy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Forehead ROI", (xs + 10, ys + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return img


def _create_step6_image(
    img_bgr: ImageArray, landmarks: FacialLandmarks,
    hairline_y: int, method: str, ratio: float,
) -> np.ndarray:
    """Create Step 6 image: final result with dashed hairline and text."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    color = (0, 255, 255) if method == "edge" else (0, 165, 255)
    for x in range(0, w, 40):
        cv2.line(img, (x, hairline_y), (min(x + 20, w), hairline_y), color, 3)
    text = f"Hairline: Y={hairline_y} ({method}) | Ratio={ratio:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (5, h - th - 15), (tw + 15, h - 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return img


def _ensure_output_dir(image_path: Path) -> Path:
    """Create and return the output directory for a given image.

    Parameters
    ----------
    image_path : Path
        Path to the input image.

    Returns
    -------
    Path
        Path to ``output/<image_stem>/``.
    """
    output_dir = Path("output") / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_step_images(
    img_bgr: ImageArray,
    landmarks: FacialLandmarks,
    steps: dict,
) -> list[np.ndarray]:
    """Build all 6 step images for the hairline detection pipeline.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    landmarks : FacialLandmarks
        Detected facial landmarks.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.

    Returns
    -------
    list[np.ndarray]
        List of 6 BGR images, one per step.
    """
    x_start, x_end, y_start, y_end = steps["roi_coords"]
    roi = steps["roi_raw"]
    roi_enhanced = steps["roi_enhanced"]
    row_intensities = steps["row_intensities"]
    abs_gradient = steps["abs_gradient"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    gradient_ratio = steps["gradient_ratio"]

    step_images: list[np.ndarray] = []

    # Step 1: Face and ROI
    step_images.append(
        _create_step1_image(img_bgr, landmarks, x_start, x_end, y_start, y_end)
    )

    # Step 2: Forehead ROI
    if roi.size > 0 and roi.ndim == 2:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    elif roi.size > 0:
        roi_bgr = roi.copy()
    else:
        roi_bgr = np.full((100, 100, 3), 40, dtype=np.uint8)
        cv2.putText(
            roi_bgr, "Empty ROI", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
    h_roi, w_roi = roi_bgr.shape[:2]
    cv2.putText(roi_bgr, f"{w_roi}x{h_roi} px", (10, h_roi - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    step_images.append(roi_bgr)

    # Step 3: CLAHE enhanced
    if roi_enhanced.size > 0 and roi_enhanced.ndim == 2:
        clahe_bgr = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)
    elif roi_enhanced.size > 0:
        clahe_bgr = roi_enhanced.copy()
    else:
        clahe_bgr = np.full((100, 100, 3), 40, dtype=np.uint8)
        cv2.putText(
            clahe_bgr, "Empty ROI", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
    step_images.append(clahe_bgr)

    # Step 4: Intensity graph
    searchable_rows = steps.get("searchable_rows", 0)
    if len(row_intensities) > 0 and searchable_rows > 0:
        min_idx = int(np.argmin(row_intensities[:searchable_rows]))
    else:
        min_idx = None
    step_images.append(
        _draw_graph(row_intensities, "Average intensity per row", 255.0, min_idx)
    )

    # Step 5: Gradient graph
    step_images.append(
        _draw_graph(
            abs_gradient,
            "Vertical gradient magnitude",
            steps["max_gradient_value_full"] if steps["max_gradient_value_full"] > 0 else None,  # noqa: E501
            steps["max_gradient_idx"] if steps["max_gradient_idx"] >= 0 else None,
        )
    )

    # Step 6: Final result
    step_images.append(
        _create_step6_image(img_bgr, landmarks, hairline_y, method, gradient_ratio)
    )

    return step_images


def _save_step_images(output_dir: Path, step_images: list[np.ndarray]) -> None:
    """Save the 6 step images as PNG files.

    Parameters
    ----------
    output_dir : Path
        Directory where images will be saved.
    step_images : list[np.ndarray]
        List of 6 BGR images.
    """
    names = [
        "step01_face_and_roi.png",
        "step02_forehead_roi.png",
        "step03_clahe_enhanced.png",
        "step04_intensity_graph.png",
        "step05_gradient_graph.png",
        "step06_final_result.png",
    ]
    for name, img in zip(names, step_images):
        path = output_dir / name
        try:
            cv2.imwrite(str(path), img)
        except OSError as exc:
            warnings.warn(f"Could not save {path}: {exc}")


def _serialize_for_json(value: object) -> object:
    """Convert NumPy types to JSON-serializable Python types.

    Parameters
    ----------
    value : object
        Value to serialize.

    Returns
    -------
    object
        JSON-serializable equivalent.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_data_json(output_dir: Path, steps: dict) -> None:
    """Write raw numerical data to ``data.json``.

    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.
    """
    keys_to_save = [
        "roi_coords",
        "hairline_y",
        "gradient",
        "row_intensities",
        "abs_gradient",
        "max_gradient_idx",
        "max_gradient_value",
        "max_gradient_value_full",
        "median_gradient",
        "gradient_ratio",
        "method",
        "avg_eyebrow_y",
        "searchable_rows",
        "face_rect",
    ]
    data = {}
    for key in keys_to_save:
        if key in steps:
            data[key] = _serialize_for_json(steps[key])
        else:
            data[key] = None

    path = output_dir / "data.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        warnings.warn(f"Could not save {path}: {exc}")


def _save_profiles_csv(output_dir: Path, steps: dict) -> None:
    """Write per-row intensity and gradient data to ``profiles.csv``.

    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.
    """
    row_intensities = steps.get("row_intensities", np.array([]))
    gradient = steps.get("gradient", np.array([]))
    abs_gradient = steps.get("abs_gradient", np.array([]))

    path = output_dir / "profiles.csv"
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_index", "intensity", "gradient", "abs_gradient"])
            n = len(row_intensities)
            for i in range(n):
                writer.writerow([
                    i,
                    float(row_intensities[i]),
                    float(gradient[i]) if i < len(gradient) else "",
                    float(abs_gradient[i]) if i < len(abs_gradient) else "",
                ])
    except OSError as exc:
        warnings.warn(f"Could not save {path}: {exc}")


def _build_summary_text(image_path: Path, face_rect: tuple, steps: dict) -> str:
    """Build the summary text for the analysis.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    face_rect : tuple
        Face bounding box (x, y, w, h).
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.

    Returns
    -------
    str
        Multi-line summary text.
    """
    x_start, x_end, y_start, y_end = steps["roi_coords"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    gradient_ratio = steps["gradient_ratio"]

    lines = [
        f"Image: {image_path.name}",
        f"Face size: {face_rect[2]}x{face_rect[3]} px",
        f"Forehead ROI: {x_end - x_start}x{y_end - y_start} px",
        f"Searchable rows: {steps['searchable_rows']}/{y_end - y_start} (top 75%)",
        f"Max gradient: {steps['max_gradient_value']:.2f} at row {steps['max_gradient_idx']}",  # noqa: E501
        f"Median gradient: {steps['median_gradient']:.2f}",
        f"Ratio: {gradient_ratio:.2f}",
        f"Result: {method} detection at y={hairline_y}",
    ]
    return "\n".join(lines)


def _save_summary_txt(output_dir: Path, summary_text: str) -> None:
    """Write summary text to ``summary.txt``.

    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved.
    summary_text : str
        Text content to write.
    """
    path = output_dir / "summary.txt"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_text)
            f.write("\n")
    except OSError as exc:
        warnings.warn(f"Could not save {path}: {exc}")


def process_image(image_path: Path, visualize: bool) -> None:
    """Process a single image, save all steps to disk, and optionally display them."""
    print(f"\n{'=' * 60}\nProcessing: {image_path.name}\n{'=' * 60}")
    img_bgr, img_gray = ImageLoader.load(image_path)
    img_h, img_w = img_gray.shape[:2]
    face_rect = FaceDetector().detect(img_gray, (img_h, img_w))
    model_path = ModelFinder.find()
    landmarks = LandmarkDetector(model_path).detect(img_gray, face_rect, image_path)
    steps = HairlineDetector().detect(img_gray, landmarks)

    # Build step images (always, even in headless mode)
    step_images = _build_step_images(img_bgr, landmarks, steps)

    # Create output directory and save all files
    output_dir = _ensure_output_dir(image_path)
    _save_step_images(output_dir, step_images)
    _save_data_json(output_dir, steps)
    _save_profiles_csv(output_dir, steps)

    summary_text = _build_summary_text(image_path, face_rect, steps)
    _save_summary_txt(output_dir, summary_text)

    # Print summary to console
    print(f"\n  {summary_text.replace(chr(10), chr(10) + '  ')}")
    print(f"\n  Output saved to: {output_dir.resolve()}")

    # Display windows only if requested
    if visualize:
        titles = [
            "Step 1: Detect face & eyebrows",
            "Step 2: Extract forehead ROI",
            "Step 3: CLAHE enhancement",
            "Step 4: Average intensity per row",
            "Step 5: Vertical gradient magnitude",
            "Step 6: Detected hairline",
        ]
        for title, img in zip(titles, step_images):
            _show_step(title, img)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Visualise each step of the hairline detection algorithm.")
    parser.add_argument(
        "--input", type=Path,
        help="Path to input image (default: first image in test_images/)",
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Display OpenCV windows for each step")
    args = parser.parse_args()
    try:
        ModelFinder.find()
    except ModelNotFoundError as exc:
        print(f"Model not found: {exc}")
        sys.exit(1)
    if args.input:
        image_path = args.input
    else:
        try:
            image_path = _get_default_image_path()
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    try:
        process_image(image_path, args.visualize)
    except VisagismError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)
    print(f"\n{'=' * 60}\nDemo complete.\n{'=' * 60}")


if __name__ == "__main__":
    main()
