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


def _draw_bar_chart(
    values: np.ndarray, title: str, highlight_idx: int | None = None,
) -> np.ndarray:
    """Draw a bar chart on a 600x300 dark gray canvas."""
    canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
    n = len(values)
    if n == 0:
        cv2.putText(canvas, title, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        return canvas

    bar_width = max(1, 530 // n)
    max_val = float(np.max(values)) if np.max(values) > 0 else 1.0
    x_offset = 50

    for i, val in enumerate(values):
        h = int((val / max_val) * 220)
        x = x_offset + i * bar_width
        color = (0, 200, 255)
        if highlight_idx is not None and i == highlight_idx:
            color = (0, 0, 255)
        cv2.rectangle(canvas, (x, 250 - h), (x + bar_width - 1, 250), color, -1)

    cv2.line(canvas, (50, 250), (580, 250), (180, 180, 180), 1)
    cv2.line(canvas, (50, 20), (50, 250), (180, 180, 180), 1)
    cv2.putText(canvas, title, (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    return canvas


def _create_step1_image(
    img_bgr: ImageArray, landmarks: FacialLandmarks,
    xs: int, xe: int, ys: int, ye: int,
    ctx_xs: int, ctx_xe: int,
) -> np.ndarray:
    """Create Step 1 image: original with face box, eyebrow line, ROI, and context."""
    img = img_bgr.copy()
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    brows = (landmarks.landmarks_by_region["left_eyebrow"]
             + landmarks.landmarks_by_region["right_eyebrow"])
    avg_y = int(np.mean([pt[1] for pt in brows]))
    cv2.line(img, (xs, avg_y), (xe, avg_y), (0, 0, 255), 2)

    # Context rectangle (cyan, semi-transparent)
    overlay = img.copy()
    cv2.rectangle(overlay, (ctx_xs, ys), (ctx_xe, ye), (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    cv2.rectangle(img, (ctx_xs, ys), (ctx_xe, ye), (255, 255, 0), 2)

    # 1-pixel ROI strip (magenta, thick for visibility)
    cv2.line(img, (xs, ys), (xs, ye), (255, 0, 255), 3)

    cv2.putText(img, "Face box", (fx + fw + 10, fy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Canny context", (ctx_xe + 10, ys + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, "1px ROI", (xs + 10, ys + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return img


def _create_step6_image(
    img_bgr: ImageArray, landmarks: FacialLandmarks,
    hairline_y: int, method: str, edge_count: int,
) -> np.ndarray:
    """Create Step 6 image: final result with dashed hairline and text."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    color = (0, 255, 255) if method == "canny" else (0, 165, 255)
    for x in range(0, w, 40):
        cv2.line(img, (x, hairline_y), (min(x + 20, w), hairline_y), color, 3)
    text = f"Hairline: Y={hairline_y} ({method}) | Edges={edge_count}"
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
    ctx_xs, ctx_xe, ctx_ys, ctx_ye = steps["canny_context_coords"]
    roi = steps["roi_raw"]
    context = steps["canny_context_raw"]
    edges = steps["canny_edge_map"]
    center_column = steps["center_column"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    edge_count = steps["edge_pixels_count"]
    first_edge_idx = steps["first_edge_idx"]

    step_images: list[np.ndarray] = []

    # Step 1: Face and ROI
    step_images.append(
        _create_step1_image(
            img_bgr, landmarks, x_start, x_end, y_start, y_end,
            ctx_xs, ctx_xe,
        )
    )

    # Step 2: Forehead ROI (1-pixel strip, scaled for visibility)
    if roi.size > 0 and roi.ndim == 2:
        # Scale the 1-pixel strip horizontally for visibility
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        roi_bgr = cv2.resize(
            roi_bgr, (200, roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    elif roi.size > 0:
        roi_bgr = roi.copy()
    else:
        roi_bgr = np.full((100, 200, 3), 40, dtype=np.uint8)
        cv2.putText(
            roi_bgr, "Empty ROI", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
    h_roi, w_roi = roi_bgr.shape[:2]
    cv2.putText(roi_bgr, f"{w_roi}x{h_roi} px (scaled)", (10, h_roi - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    step_images.append(roi_bgr)

    # Step 3: Canny context
    if context.size > 0 and context.ndim == 2:
        context_bgr = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)
    elif context.size > 0:
        context_bgr = context.copy()
    else:
        context_bgr = np.full((100, 100, 3), 40, dtype=np.uint8)
        cv2.putText(
            context_bgr, "Empty context", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
    step_images.append(context_bgr)

    # Step 4: Canny edge map
    if edges.size > 0 and edges.ndim == 2:
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Highlight center column in red
        face_center_x = landmarks.face_rect[0] + landmarks.face_rect[2] // 2
        center_col_idx = face_center_x - ctx_xs
        if 0 <= center_col_idx < edges_bgr.shape[1]:
            edges_bgr[:, center_col_idx] = (0, 0, 255)
    else:
        edges_bgr = np.full((100, 100, 3), 40, dtype=np.uint8)
        cv2.putText(
            edges_bgr, "Empty edge map", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
    step_images.append(edges_bgr)

    # Step 5: Center column scan
    if len(center_column) > 0:
        step_images.append(
                _draw_bar_chart(
                center_column.astype(np.float32),
                "Center column scan (bottom-to-up, 0=dark, 255=edge)",
                highlight_idx=first_edge_idx if first_edge_idx >= 0 else None,
            )
        )
    else:
        empty_chart = np.full((300, 600, 3), 40, dtype=np.uint8)
        cv2.putText(
            empty_chart, "No center column data", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
        )
        step_images.append(empty_chart)

    # Step 6: Final result
    step_images.append(
        _create_step6_image(img_bgr, landmarks, hairline_y, method, edge_count)
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
        "step03_canny_context.png",
        "step04_canny_edge_map.png",
        "step05_center_column_scan.png",
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
        "method",
        "roi_raw",
        "canny_context_coords",
        "canny_context_raw",
        "canny_edge_map",
        "center_column",
        "first_edge_idx",
        "edge_pixels_count",
        "gaussian_ksize",
        "canny_low",
        "canny_high",
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
    """Write per-row center column data to ``profiles.csv``.

    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.
    """
    center_column = steps.get("center_column", np.array([]))

    path = output_dir / "profiles.csv"
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_index", "center_column_value"])
            n = len(center_column)
            for i in range(n):
                writer.writerow([i, int(center_column[i])])
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
    edge_count = steps["edge_pixels_count"]
    first_edge_idx = steps["first_edge_idx"]

    lines = [
        f"Image: {image_path.name}",
        f"Face size: {face_rect[2]}x{face_rect[3]} px",
        f"Forehead ROI: {x_end - x_start}x{y_end - y_start} px (1px wide strip)",
        f"Searchable rows: {steps['searchable_rows']}",
        f"First edge index: {first_edge_idx}",
        f"Edge pixels in center column: {edge_count}",
        f"Canny thresholds: low={steps['canny_low']}, high={steps['canny_high']}",
        f"Gaussian ksize: {steps['gaussian_ksize']}",
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
            "Step 1: Detect face & define ROI",
            "Step 2: Extract 1px forehead ROI",
            "Step 3: Canny context",
            "Step 4: Canny edge map",
            "Step 5: Center column scan",
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
