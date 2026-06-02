"""Demo script visualising each step of the non-ML landmark detection."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visagism.errors import VisagismError  # noqa: E402
from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.image_loader import ImageLoader  # noqa: E402
from visagism.nonml_landmark_detector import NonMLLandmarkDetector  # noqa: E402
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
    values: np.ndarray,
    title: str,
    highlight_idx: int | None = None,
    highlight_idx2: int | None = None,
) -> np.ndarray:
    """Draw a bar chart on a 600x300 dark gray canvas."""
    canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
    n = len(values)
    if n == 0:
        cv2.putText(
            canvas, title, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1,
        )
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
        if highlight_idx2 is not None and i == highlight_idx2:
            color = (0, 0, 255)
        cv2.rectangle(canvas, (x, 250 - h), (x + bar_width - 1, 250), color, -1)

    cv2.line(canvas, (50, 250), (580, 250), (180, 180, 180), 1)
    cv2.line(canvas, (50, 20), (50, 250), (180, 180, 180), 1)
    cv2.putText(
        canvas, title, (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1,
    )
    return canvas


def _create_step1_image(
    img_bgr: ImageArray, face_rect: tuple[int, int, int, int]
) -> np.ndarray:
    """Create Step 1 image: original with face bounding box."""
    img = img_bgr.copy()
    x, y, w, h = face_rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img, "Face bounding box", (x + w + 10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return img


def _create_step2_image(
    img_bgr: ImageArray,
    face_rect: tuple[int, int, int, int],
    eye_steps: dict,
    eye_points: list[tuple[int, int]],
) -> np.ndarray:
    """Create Step 2 image: tight eye ROIs with pupils and corners."""
    if eye_steps.get("fallback"):
        canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
        cv2.putText(
            canvas, "Eye detection fallback", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
        )
        return canvas

    img = img_bgr.copy()
    x, y, w, h = face_rect

    # Tight eye ROIs using face proportions
    left_roi_x = x + int(w * 0.15)
    left_roi_y = y + int(h * 0.35)
    left_roi_w = int(w * 0.30)
    left_roi_h = int(h * 0.20)

    right_roi_x = x + int(w * 0.55)
    right_roi_y = y + int(h * 0.35)
    right_roi_w = int(w * 0.30)
    right_roi_h = int(h * 0.20)

    # Draw tight eye ROIs (blue rectangles)
    cv2.rectangle(
        img, (left_roi_x, left_roi_y),
        (left_roi_x + left_roi_w, left_roi_y + left_roi_h), (255, 0, 0), 2,
    )
    cv2.rectangle(
        img, (right_roi_x, right_roi_y),
        (right_roi_x + right_roi_w, right_roi_y + right_roi_h), (255, 0, 0), 2,
    )

    # Draw pupil centers (red dots)
    left_pupil = eye_steps.get("left_pupil")
    right_pupil = eye_steps.get("right_pupil")
    if left_pupil:
        cv2.circle(img, left_pupil, 5, (0, 0, 255), -1)
    if right_pupil:
        cv2.circle(img, right_pupil, 5, (0, 0, 255), -1)

    # Draw detected corners (green dots)
    left_corners = eye_steps.get("left_corners")
    right_corners = eye_steps.get("right_corners")
    if left_corners:
        for pt in left_corners:
            cv2.circle(img, pt, 5, (0, 255, 0), -1)
    if right_corners:
        for pt in right_corners:
            cv2.circle(img, pt, 5, (0, 255, 0), -1)

    cv2.putText(
        img, "Tight ROIs (blue), pupils (red), corners (green)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return img


def _create_step3_image(
    nose_steps: dict, face_rect: tuple[int, int, int, int]
) -> np.ndarray:
    """Create Step 3 image: vertical brightness profile with peak marked."""
    if nose_steps.get("fallback"):
        canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
        cv2.putText(
            canvas, "Nose detection fallback", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
        )
        return canvas

    profile = nose_steps["profile"]
    nose_base_y = nose_steps["nose_base_y"]
    y_start = face_rect[1] + int(face_rect[3] * 0.35)
    highlight_idx = nose_base_y - y_start
    highlight_idx = max(0, min(highlight_idx, len(profile) - 1))

    return _draw_bar_chart(
        profile,
        "Vertical brightness profile (nose base peak)",
        highlight_idx=highlight_idx,
    )


def _create_step4_image(mouth_steps: dict) -> np.ndarray:
    """Create Step 4 image: horizontal Sobel gradient with peaks marked."""
    if mouth_steps.get("fallback"):
        canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
        cv2.putText(
            canvas, "Mouth detection fallback", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
        )
        return canvas

    gradient_profile = mouth_steps["gradient_profile"]
    left_idx = mouth_steps["left_idx"]
    right_idx = mouth_steps["right_idx"]

    return _draw_bar_chart(
        gradient_profile.astype(np.float32),
        "Horizontal Sobel gradient (mouth corners)",
        highlight_idx=left_idx,
        highlight_idx2=right_idx,
    )


def _create_step5_image(
    eyebrow_steps: dict,
    face_rect: tuple[int, int, int, int],
    brow_points: list[tuple[int, int]],
) -> np.ndarray:
    """Create Step 5 image: horizontal edges with peaks marked."""
    if eyebrow_steps.get("fallback"):
        canvas = np.full((300, 600, 3), 40, dtype=np.uint8)
        cv2.putText(
            canvas, "Eyebrow detection fallback", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
        )
        return canvas

    sobel_y = eyebrow_steps["sobel_y"]
    abs_sobel = np.abs(sobel_y)

    # Normalize for visualization
    max_val = float(np.max(abs_sobel)) if abs_sobel.size > 0 else 1.0
    if max_val > 0:
        vis = (abs_sobel / max_val * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(abs_sobel, dtype=np.uint8)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw circles at detected peaks (convert full-image coords to ROI-relative)
    x, y, w, h = face_rect
    y_start = y + int(h * 0.10)
    for pt in brow_points:
        rel_x = pt[0] - x
        rel_y = pt[1] - y_start
        cv2.circle(vis_bgr, (rel_x, rel_y), 5, (0, 0, 255), -1)

    # Resize for visibility if too small
    h, w = vis_bgr.shape[:2]
    if w < 400:
        scale = max(1, 400 // w)
        vis_bgr = cv2.resize(
            vis_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
        )

    cv2.putText(
        vis_bgr, "Horizontal edges (eyebrow peaks)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return vis_bgr


def _create_step6_image(
    img_bgr: ImageArray,
    landmarks: FacialLandmarks,
) -> np.ndarray:
    """Create Step 6 image: all 14 anchors overlaid on original image."""
    img = img_bgr.copy()
    lm = landmarks.landmarks_68

    # Color map for regions
    colors = {
        "jaw": (0, 255, 0),
        "left_eye": (255, 0, 0),
        "right_eye": (255, 0, 0),
        "nose_bridge": (0, 255, 255),
        "nose_tip": (0, 255, 255),
        "outer_mouth": (0, 0, 255),
        "left_eyebrow": (255, 255, 0),
        "right_eyebrow": (255, 255, 0),
    }

    # Draw all 14 anchor points
    anchor_indices = [0, 8, 16, 31, 33, 35, 36, 39, 42, 45, 48, 54, 19, 24]
    region_map = {
        0: "jaw", 8: "jaw", 16: "jaw",
        31: "nose_tip", 33: "nose_bridge", 35: "nose_tip",
        36: "left_eye", 39: "left_eye",
        42: "right_eye", 45: "right_eye",
        48: "outer_mouth", 54: "outer_mouth",
        19: "left_eyebrow", 24: "right_eyebrow",
    }

    for idx in anchor_indices:
        pt = lm[idx]
        if pt == (-1, -1):
            continue
        color = colors.get(region_map.get(idx, ""), (255, 255, 255))
        cv2.circle(img, pt, 6, color, -1)
        cv2.circle(img, pt, 6, (0, 0, 0), 1)

    # Draw connecting lines for jawline
    jaw_pts = [lm[i] for i in [0, 8, 16] if lm[i] != (-1, -1)]
    if len(jaw_pts) == 3:
        cv2.line(img, jaw_pts[0], jaw_pts[1], colors["jaw"], 2)
        cv2.line(img, jaw_pts[1], jaw_pts[2], colors["jaw"], 2)

    # Draw eye lines
    if lm[36] != (-1, -1) and lm[39] != (-1, -1):
        cv2.line(img, lm[36], lm[39], colors["left_eye"], 2)
    if lm[42] != (-1, -1) and lm[45] != (-1, -1):
        cv2.line(img, lm[42], lm[45], colors["right_eye"], 2)

    # Draw mouth line
    if lm[48] != (-1, -1) and lm[54] != (-1, -1):
        cv2.line(img, lm[48], lm[54], colors["outer_mouth"], 2)

    # Draw nose line
    if lm[31] != (-1, -1) and lm[33] != (-1, -1) and lm[35] != (-1, -1):
        cv2.line(img, lm[31], lm[33], colors["nose_tip"], 2)
        cv2.line(img, lm[33], lm[35], colors["nose_tip"], 2)

    # Legend
    h, w = img.shape[:2]
    legend_y = h - 120
    items = [
        ("Jaw", colors["jaw"]),
        ("Eyes", colors["left_eye"]),
        ("Nose", colors["nose_bridge"]),
        ("Mouth", colors["outer_mouth"]),
        ("Eyebrows", colors["left_eyebrow"]),
    ]
    for i, (label, color) in enumerate(items):
        y_pos = legend_y + i * 20
        cv2.circle(img, (w - 150, y_pos), 6, color, -1)
        cv2.putText(
            img, label, (w - 130, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return img


def _ensure_output_dir(image_path: Path) -> Path:
    """Create and return the output directory for a given image."""
    output_dir = Path("output") / f"{image_path.stem}_nonml"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_step_images(output_dir: Path, step_images: list[np.ndarray]) -> None:
    """Save the 6 step images as PNG files."""
    names = [
        "step01_face_box.png",
        "step02_eye_detection.png",
        "step03_nose_detection.png",
        "step04_mouth_detection.png",
        "step05_eyebrow_detection.png",
        "step06_final_anchors.png",
    ]
    for name, img in zip(names, step_images):
        path = output_dir / name
        try:
            cv2.imwrite(str(path), img)
        except OSError as exc:
            warnings.warn(f"Could not save {path}: {exc}")


def process_image(image_path: Path, visualize: bool) -> None:
    """Process a single image, save all steps to disk, and optionally display them."""
    print(f"\n{'=' * 60}\nProcessing: {image_path.name}\n{'=' * 60}")
    img_bgr, img_gray = ImageLoader.load(image_path)
    img_h, img_w = img_gray.shape[:2]
    face_rect = FaceDetector().detect(img_gray, (img_h, img_w))

    detector = NonMLLandmarkDetector(Path("/dev/null"))
    landmarks, steps = detector.detect_with_steps(img_gray, face_rect, image_path)

    eye_points = [
        landmarks.landmarks_68[i]
        for i in [36, 39, 42, 45]
        if landmarks.landmarks_68[i] != (-1, -1)
    ]
    brow_points = [
        landmarks.landmarks_68[i]
        for i in [19, 24]
        if landmarks.landmarks_68[i] != (-1, -1)
    ]

    step_images: list[np.ndarray] = []

    # Step 1: Original image with face bounding box
    step_images.append(_create_step1_image(img_bgr, face_rect))

    # Step 2: Eye detection
    step_images.append(
        _create_step2_image(img_bgr, face_rect, steps["eye"], eye_points)
    )

    # Step 3: Nose detection
    step_images.append(_create_step3_image(steps["nose"], face_rect))

    # Step 4: Mouth detection
    step_images.append(_create_step4_image(steps["mouth"]))

    # Step 5: Eyebrow detection
    step_images.append(_create_step5_image(steps["eyebrow"], face_rect, brow_points))

    # Step 6: Final result with all anchors
    step_images.append(_create_step6_image(img_bgr, landmarks))

    # Create output directory and save all files
    output_dir = _ensure_output_dir(image_path)
    _save_step_images(output_dir, step_images)

    print(f"\n  Output saved to: {output_dir.resolve()}")

    # Display windows only if requested
    if visualize:
        titles = [
            "Step 1: Original image with face bounding box",
            "Step 2: Eye detection — tight ROIs with pupils and corners",
            "Step 3: Nose detection — vertical brightness profile",
            "Step 4: Mouth detection — horizontal Sobel gradient",
            "Step 5: Eyebrow detection — horizontal edges",
            "Step 6: Final result — all 14 anchors",
        ]
        for title, img in zip(titles, step_images):
            _show_step(title, img)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Visualise each step of the non-ML landmark detection.")
    parser.add_argument(
        "--input", type=Path,
        help="Path to input image (default: first image in test_images/)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Display OpenCV windows for each step")
    args = parser.parse_args()

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
