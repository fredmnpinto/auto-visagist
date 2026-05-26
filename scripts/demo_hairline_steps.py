"""Demo script visualising each step of the hairline detection algorithm."""

from __future__ import annotations

import argparse
import sys
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


def process_image(image_path: Path, visualize: bool) -> None:
    """Process a single image and display each step sequentially."""
    print(f"\n{'=' * 60}\nProcessing: {image_path.name}\n{'=' * 60}")
    img_bgr, img_gray = ImageLoader.load(image_path)
    img_h, img_w = img_gray.shape[:2]
    face_rect = FaceDetector().detect(img_gray, (img_h, img_w))
    model_path = ModelFinder.find()
    landmarks = LandmarkDetector(model_path).detect(img_gray, face_rect, image_path)
    steps = HairlineDetector().detect(img_gray, landmarks)
    roi = steps["roi_raw"]
    roi_enhanced = steps["roi_enhanced"]
    row_intensities = steps["row_intensities"]
    abs_gradient = steps["abs_gradient"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    gradient_ratio = steps["gradient_ratio"]
    x_start, x_end, y_start, y_end = steps["roi_coords"]
    print(f"\n  Image: {image_path.name}\n"
          f"  Face size: {face_rect[2]}x{face_rect[3]} px\n"
          f"  Forehead ROI: {x_end - x_start}x{y_end - y_start} px\n"
          f"  Searchable rows: {steps['searchable_rows']}/{y_end - y_start} (top 75%)\n"
          f"  Max gradient: {steps['max_gradient_value']:.2f} at row {steps['max_gradient_idx']}\n"  # noqa: E501
          f"  Median gradient: {steps['median_gradient']:.2f}\n"
          f"  Ratio: {gradient_ratio:.2f}\n"
          f"  Result: {method} detection at y={hairline_y}")
    if not visualize:
        print("Running headless...")
        return
    _show_step("Step 1: Detect face & eyebrows",
               _create_step1_image(img_bgr, landmarks, x_start, x_end, y_start, y_end))
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi
    h_roi, w_roi = roi.shape[:2]
    cv2.putText(roi_bgr, f"{w_roi}x{h_roi} px", (10, h_roi - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    _show_step("Step 2: Extract forehead ROI", roi_bgr)
    _show_step("Step 3: CLAHE enhancement",
               cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR))
    min_idx = int(np.argmin(row_intensities[:steps["searchable_rows"]]))
    _show_step(
        "Step 4: Average intensity per row",
        _draw_graph(row_intensities, "Average intensity per row", 255.0, min_idx),
    )
    _show_step(
        "Step 5: Vertical gradient magnitude",
        _draw_graph(
            abs_gradient, "Vertical gradient magnitude",
            steps["max_gradient_value_full"], steps["max_gradient_idx"],
        ),
    )
    _show_step(
        "Step 6: Detected hairline",
        _create_step6_image(
            img_bgr, landmarks, hairline_y, method, gradient_ratio,
        ),
    )


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
