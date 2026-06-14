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


def _create_step2_pure_canny(
    img_gray: ImageArray,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    """Create Step 2 image: pure Canny on raw grayscale without preprocessing.

    Parameters
    ----------
    img_gray : ImageArray
        Full grayscale input image.
    canny_low : int
        Lower Canny threshold.
    canny_high : int
        Upper Canny threshold.

    Returns
    -------
    np.ndarray
        BGR image of Canny edges on the full image.
    """
    edges = cv2.Canny(img_gray, canny_low, canny_high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _create_step3_preprocessed(
    img_bgr: ImageArray,
    steps: dict,
) -> np.ndarray:
    """Create Step 3: full image with closed_context composited and ROI overlay.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.

    Returns
    -------
    np.ndarray
        Full-size BGR image with preprocessed context and ROI overlay.
    """
    img = img_bgr.copy()
    ctx_xs, ctx_xe, ctx_ys, ctx_ye = steps["canny_context_coords"]
    closed = steps["closed_context"]

    has_content = closed.size > 0 and closed.ndim == 2

    if has_content:
        closed_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        h, w = closed_bgr.shape[:2]
        dst_y1 = max(0, ctx_ys)
        dst_y2 = min(img.shape[0], ctx_ys + h)
        dst_x1 = max(0, ctx_xs)
        dst_x2 = min(img.shape[1], ctx_xs + w)
        src_y1 = max(0, -ctx_ys)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x1 = max(0, -ctx_xs)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            img[dst_y1:dst_y2, dst_x1:dst_x2] = (
                closed_bgr[src_y1:src_y2, src_x1:src_x2]
            )
        label = "Forehead Context"
        label_color = (0, 255, 0)
    else:
        label = "Empty ROI — fallback used"
        label_color = (0, 0, 255)

    if ctx_ys < ctx_ye and ctx_xs < ctx_xe:
        cv2.rectangle(
            img, (ctx_xs, ctx_ys), (ctx_xe, ctx_ye), (0, 255, 0), 2,
        )
        cv2.putText(
            img, label, (ctx_xs, ctx_ys - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2,
        )
    else:
        h, w = img.shape[:2]
        cv2.putText(
            img, "Invalid ROI", (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

    return img


def _create_step4_final_canny(
    img_bgr: ImageArray,
    landmarks: FacialLandmarks,
    steps: dict,
) -> np.ndarray:
    """Create Step 4: full image with edge map composited, ROI, and hairline.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    landmarks : FacialLandmarks
        Detected facial landmarks (unused but kept for API consistency).
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.

    Returns
    -------
    np.ndarray
        Full-size BGR image with edge overlay, ROI, and hairline line.
    """
    img = img_bgr.copy()
    ctx_xs, ctx_xe, ctx_ys, ctx_ye = steps["canny_context_coords"]
    edges = steps["canny_edge_map"]
    hairline_y = steps["hairline_y"]

    has_content = edges.size > 0 and edges.ndim == 2

    if has_content:
        h, w = edges.shape[:2]
        dst_y1 = max(0, ctx_ys)
        dst_y2 = min(img.shape[0], ctx_ys + h)
        dst_x1 = max(0, ctx_xs)
        dst_x2 = min(img.shape[1], ctx_xs + w)
        src_y1 = max(0, -ctx_ys)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x1 = max(0, -ctx_xs)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            edge_region = edges[src_y1:src_y2, src_x1:src_x2]
            mask = edge_region == 255
            roi_img = img[dst_y1:dst_y2, dst_x1:dst_x2]
            roi_img[mask] = (255, 255, 0)  # cyan
            img[dst_y1:dst_y2, dst_x1:dst_x2] = roi_img
        label = "Forehead Context"
        label_color = (0, 255, 0)
    else:
        label = "Empty ROI — fallback used"
        label_color = (0, 0, 255)

    if ctx_ys < ctx_ye and ctx_xs < ctx_xe:
        cv2.rectangle(
            img, (ctx_xs, ctx_ys), (ctx_xe, ctx_ye), (0, 255, 0), 2,
        )
        cv2.putText(
            img, label, (ctx_xs, ctx_ys - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2,
        )
    else:
        h, w = img.shape[:2]
        cv2.putText(
            img, "Invalid ROI", (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

    if 0 <= hairline_y < img.shape[0]:
        for x in range(0, img.shape[1], 40):
            cv2.line(
                img,
                (x, hairline_y),
                (min(x + 20, img.shape[1]), hairline_y),
                (0, 0, 255),
                3,
            )
        cv2.putText(
            img, f"Hairline y={hairline_y}", (10, hairline_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

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
        Path to ``data/output/<image_stem>/``.
    """
    output_dir = Path("data/output") / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_step_images(
    img_bgr: ImageArray,
    img_gray: ImageArray,
    steps: dict,
    landmarks: FacialLandmarks,
) -> list[np.ndarray]:
    """Build all 4 step images for the hairline detection pipeline.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    img_gray : ImageArray
        Original grayscale image.
    steps : dict
        Result dictionary from ``HairlineDetector.detect()``.
    landmarks : FacialLandmarks
        Detected facial landmarks.

    Returns
    -------
    list[np.ndarray]
        List of 4 BGR images, one per step, all matching ``img_bgr`` dimensions.
    """
    step_images: list[np.ndarray] = []

    # Step 1: Original Image
    step_images.append(img_bgr.copy())

    # Step 2: Pure Canny on raw grayscale
    step_images.append(
        _create_step2_pure_canny(
            img_gray, steps["canny_low"], steps["canny_high"]
        )
    )

    # Step 3: Preprocessed image with ROI overlay
    step_images.append(_create_step3_preprocessed(img_bgr, steps))

    # Step 4: Final Canny with ROI and hairline overlay
    step_images.append(_create_step4_final_canny(img_bgr, landmarks, steps))

    return step_images


def _save_step_images(output_dir: Path, step_images: list[np.ndarray]) -> None:
    """Save the 4 step images as PNG files.

    Parameters
    ----------
    output_dir : Path
        Directory where images will be saved.
    step_images : list[np.ndarray]
        List of 4 BGR images.
    """
    names = [
        "step01_original_image.png",
        "step02_pure_canny.png",
        "step03_preprocessed.png",
        "step04_final_canny.png",
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
        "closed_context",
        "close_ksize",
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
        f"Closing ksize: {steps['close_ksize']}",
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
    step_images = _build_step_images(img_bgr, img_gray, steps, landmarks)

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
            "Step 1: Original Image",
            "Step 2: Pure Canny (no preprocessing)",
            "Step 3: Preprocessed (blur + closing)",
            "Step 4: Final Canny Edge Map",
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
