"""Diagnostic script for hairline detection edge-detection pipeline.

For each image in ``test_images/``, this script:

1. Loads the image, detects the face and landmarks.
2. Manually extracts the forehead ROI and runs the intermediate
   preprocessing steps (Gaussian blur, Canny edges, row-wise run scan).
3. Creates a 4-subplot diagnostic figure.
4. Saves the figure to ``output/diagnose_<image_name>.png``.
5. Prints a text summary of the results.

This helps understand why Canny edge detection may be falling back to the
geometric estimate on test images.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Allow imports from the project root when running this script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from visagism.constants import (
    HAIRLINE_CANNY_HIGH,
    HAIRLINE_CANNY_LOW,
    HAIRLINE_GAUSSIAN_KSIZE,
    HAIRLINE_MIN_EDGE_SPAN,
)
from visagism.errors import ModelNotFoundError, VisagismError
from visagism.face_detector import FaceDetector
from visagism.hairline_detector import HairlineDetector
from visagism.image_loader import ImageLoader
from visagism.landmark_detector import LandmarkDetector
from visagism.model_finder import ModelFinder


def _longest_consecutive_run(row: np.ndarray, target: int) -> int:
    """Find the longest consecutive run of *target* in a 1-D array.

    Parameters
    ----------
    row : np.ndarray
        1-D array of pixel values.
    target : int
        The pixel value to search for (typically 255 for edges).

    Returns
    -------
    int
        Length of the longest consecutive segment equal to *target*.
    """
    mask = (row == target).astype(np.int8)
    changes = np.diff(mask, prepend=0, append=0)
    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0]
    if len(run_starts) == 0:
        return 0
    run_lengths = run_ends - run_starts
    return int(np.max(run_lengths))


def compute_row_runs(edges: np.ndarray) -> List[int]:
    """Compute the longest edge run for each row in an edge map.

    Parameters
    ----------
    edges : np.ndarray
        Binary edge map (typically output of Canny).

    Returns
    -------
    list of int
        Longest consecutive run of 255 pixels for each row.
    """
    roi_h = edges.shape[0]
    runs: List[int] = []
    for r in range(roi_h):
        row = edges[r, :]
        longest_run = _longest_consecutive_run(row, 255)
        runs.append(longest_run)
    return runs


def diagnose_image(
    image_path: Path,
    face_detector: FaceDetector,
    landmark_detector: LandmarkDetector,
    output_dir: Path,
) -> None:
    """Run hairline detection diagnostics on a single image.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    face_detector : FaceDetector
        Initialised face detector.
    landmark_detector : LandmarkDetector
        Initialised landmark detector.
    output_dir : Path
        Directory where diagnostic figures are saved.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 60}")

    # 1. Load image
    img_bgr, img_gray = ImageLoader.load(image_path)
    img_h, img_w = img_gray.shape[:2]
    print(f"  Image size: {img_w}x{img_h}")

    # 2. Detect face
    face_rect = face_detector.detect(img_gray, (img_h, img_w))
    fx, fy, fw, fh = face_rect
    print(f"  Face rect: ({fx}, {fy}, {fw}, {fh})")

    # 3. Detect landmarks
    landmarks = landmark_detector.detect(img_gray, face_rect, image_path)

    # 4. Get all intermediate data from HairlineDetector
    hairline_detector = HairlineDetector()
    steps = hairline_detector.debug_steps(img_gray, landmarks)

    roi = steps["roi_raw"]
    roi_enhanced = steps["roi_enhanced"]
    x_start, x_end, y_start, y_end = steps["roi_coords"]
    avg_eyebrow_y = steps["avg_eyebrow_y"]
    new_hairline_y = steps["hairline_y"]
    new_method = steps["method"]

    roi_width = x_end - x_start
    roi_height = y_end - y_start
    print(f"  Avg eyebrow y: {avg_eyebrow_y}")
    print(f"  ROI: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
    print(f"  ROI dimensions: {roi_width}x{roi_height}")

    # 5. Run Canny edge detection on the enhanced ROI for diagnostic comparison
    if roi_enhanced.size == 0 or roi_height < 2 or roi_width < 2:
        print("  ERROR: Invalid ROI — cannot extract forehead region.")
        return

    ksize = HAIRLINE_GAUSSIAN_KSIZE
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(roi_enhanced, (ksize, ksize), 0)
    edges = cv2.Canny(blurred, HAIRLINE_CANNY_LOW, HAIRLINE_CANNY_HIGH)

    # 6. Scan rows for longest edge runs
    runs = compute_row_runs(edges)
    max_run = max(runs) if runs else 0
    max_run_row = runs.index(max_run) if runs else 0
    threshold = HAIRLINE_MIN_EDGE_SPAN * fw
    edge_detected = max_run >= threshold

    # Detected hairline y (either from edge or fallback)
    if edge_detected:
        detected_hairline_y = y_start + max_run_row
        method = "edge detection"
    else:
        # Fallback: superior third = medium third
        nose_base_y = landmarks.landmarks_68[33][1]
        medium_third_height = nose_base_y - avg_eyebrow_y
        detected_hairline_y = avg_eyebrow_y - medium_third_height
        detected_hairline_y = max(detected_hairline_y, fy)
        method = "fallback (geometric estimate)"

    # 7. Print summary
    print("\n  --- Summary ---")
    print(f"  Face width: {fw} px")
    print(f"  ROI dimensions: {roi_width}x{roi_height} px")
    print(f"  Maximum edge run: {max_run} px ({max_run / fw * 100:.1f}% of face width)")
    print(f"  Threshold ({HAIRLINE_MIN_EDGE_SPAN} * face_width): {threshold:.1f} px")
    print(f"  Edge detection succeeded: {'YES' if edge_detected else 'NO'}")
    print(f"  Method used: {method}")
    print(f"  Detected hairline y: {detected_hairline_y}")
    print("\n  --- New Gradient-Based Detection ---")
    print(f"  New detected hairline y: {new_hairline_y}")
    print(f"  New method: {new_method}")
    print(f"  Difference (new - old): {new_hairline_y - detected_hairline_y} px")

    # 9. Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Hairline Diagnostic: {image_path.name}",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Original image with overlays
    ax1 = axes[0]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title("Original with Overlays")

    # Face bounding box (green)
    rect = plt.Rectangle(
        (fx, fy), fw, fh, linewidth=2, edgecolor="lime", facecolor="none"
    )
    ax1.add_patch(rect)

    # Original face top (dashed cyan line to show expansion)
    ax1.axhline(
        y=fy,
        color="cyan",
        linestyle="--",
        linewidth=1.5,
        label="Original face top",
    )

    # Eyebrow line (red)
    ax1.axhline(
        y=avg_eyebrow_y,
        color="red",
        linestyle="-",
        linewidth=1.5,
        label="Eyebrow line",
    )

    # Detected hairline (yellow dashed)
    ax1.axhline(
        y=detected_hairline_y,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label="Old hairline (Canny)",
    )

    # New gradient-based hairline (magenta)
    ax1.axhline(
        y=new_hairline_y,
        color="magenta",
        linestyle="-",
        linewidth=2,
        label="New hairline (gradient)",
    )

    # ROI rectangle (cyan, semi-transparent)
    roi_rect = plt.Rectangle(
        (x_start, y_start),
        roi_width,
        roi_height,
        linewidth=1.5,
        edgecolor="cyan",
        facecolor="cyan",
        alpha=0.15,
    )
    ax1.add_patch(roi_rect)

    ax1.legend(loc="upper right", fontsize=8)
    ax1.axis("off")

    # Subplot 2: Forehead ROI (grayscale)
    ax2 = axes[1]
    ax2.imshow(roi, cmap="gray")
    ax2.set_title("Forehead ROI (Grayscale)")
    ax2.axis("off")

    # Subplot 3: Canny edge map
    ax3 = axes[2]
    ax3.imshow(edges, cmap="gray")
    ax3.set_title("Canny Edge Map")
    ax3.axis("off")

    # Subplot 4: Bar chart of longest run per row
    ax4 = axes[3]
    row_indices = list(range(len(runs)))
    bars = ax4.bar(row_indices, runs, color="steelblue", width=1.0)

    # Highlight max run row
    if runs:
        bars[max_run_row].set_color("orange")

    # Threshold line
    ax4.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.1f})",
    )

    ax4.set_title("Longest Edge Run per Row")
    ax4.set_xlabel("Row index (within ROI)")
    ax4.set_ylabel("Run length (px)")
    ax4.legend(loc="upper right", fontsize=8)

    # Annotate max run
    if runs:
        ax4.annotate(
            f"Max: {max_run}",
            xy=(max_run_row, max_run),
            xytext=(max_run_row + 5, max_run + 5),
            fontsize=9,
            color="orange",
            arrowprops=dict(arrowstyle="->", color="orange"),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 10. Save figure
    output_path = output_dir / f"diagnose_{image_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved diagnostic figure: {output_path}")


def main() -> None:
    """Run hairline diagnostics on all images in ``test_images/``."""
    project_root = Path(__file__).resolve().parent.parent
    test_images_dir = project_root / "test_images"
    output_dir = project_root / "output"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find dlib model
    try:
        model_path = ModelFinder.find()
    except ModelNotFoundError as exc:
        print(f"Model not found: {exc}")
        print("Skipping diagnosis — download the model and try again.")
        return

    # Initialise detectors
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector(model_path)

    # Gather test images
    image_paths = (
        sorted(test_images_dir.glob("*.jpg"))
        + sorted(test_images_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No test images found in {test_images_dir}")
        return

    print(f"Found {len(image_paths)} test image(s) in {test_images_dir}")

    for image_path in image_paths:
        try:
            diagnose_image(image_path, face_detector, landmark_detector, output_dir)
        except VisagismError as exc:
            print(f"\n  ERROR processing {image_path.name}: {exc}")
        except Exception as exc:
            print(f"\n  UNEXPECTED ERROR processing {image_path.name}: {exc}")

    print(f"\n{'=' * 60}")
    print("Diagnosis complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
