"""Diagnostic script for hairline detection edge-detection pipeline.

For each image in ``test_images/``, this script:

1. Loads the image, detects the face and landmarks.
2. Runs ``HairlineDetector.detect()`` and extracts all intermediate data.
3. Creates a 4-subplot diagnostic figure.
4. Saves the figure to ``output/diagnose_<image_name>.png``.
5. Prints a text summary of the results.

This helps understand why Canny edge detection may be falling back to the
geometric estimate on test images.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the project root when running this script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from visagism.errors import ModelNotFoundError, VisagismError  # noqa: E402
from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.hairline_detector import HairlineDetector  # noqa: E402
from visagism.image_loader import ImageLoader  # noqa: E402
from visagism.landmark_detector import LandmarkDetector  # noqa: E402
from visagism.model_finder import ModelFinder  # noqa: E402


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
    steps = hairline_detector.detect(img_gray, landmarks)

    context = steps["canny_context_raw"]
    edges = steps["canny_edge_map"]
    center_column = steps["center_column"]
    x_start, x_end, y_start, y_end = steps["roi_coords"]
    ctx_xs, ctx_xe, ctx_ys, ctx_ye = steps["canny_context_coords"]
    avg_eyebrow_y = steps["avg_eyebrow_y"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    first_edge_idx = steps["first_edge_idx"]
    edge_pixels_count = steps["edge_pixels_count"]

    roi_width = x_end - x_start
    roi_height = y_end - y_start
    print(f"  Avg eyebrow y: {avg_eyebrow_y}")
    print(f"  ROI: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
    print(f"  ROI dimensions: {roi_width}x{roi_height}")
    print(f"  Context: x=[{ctx_xs}, {ctx_xe}], y=[{ctx_ys}, {ctx_ye}]")

    # 5. Print summary
    print("\n  --- Summary ---")
    print(f"  Face width: {fw} px")
    print(f"  ROI dimensions: {roi_width}x{roi_height} px")
    print(f"  Searchable rows: {steps['searchable_rows']}")
    print(f"  First edge index: {first_edge_idx}")
    print(f"  Edge pixels in center column: {edge_pixels_count}")
    print(f"  Canny thresholds: low={steps['canny_low']}, high={steps['canny_high']}")
    print(f"  Gaussian ksize: {steps['gaussian_ksize']}")
    print(f"  Method used: {method}")
    print(f"  Detected hairline y: {hairline_y}")

    # 6. Create visualization
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
        y=hairline_y,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label=f"Hairline ({method})",
    )

    # Context rectangle (cyan, semi-transparent)
    ctx_rect = plt.Rectangle(
        (ctx_xs, ctx_ys),
        ctx_xe - ctx_xs,
        ctx_ye - ctx_ys,
        linewidth=1.5,
        edgecolor="cyan",
        facecolor="cyan",
        alpha=0.15,
    )
    ax1.add_patch(ctx_rect)

    # 1-pixel ROI strip (magenta)
    roi_rect = plt.Rectangle(
        (x_start, y_start),
        roi_width,
        roi_height,
        linewidth=2,
        edgecolor="magenta",
        facecolor="magenta",
        alpha=0.3,
    )
    ax1.add_patch(roi_rect)

    ax1.legend(loc="upper right", fontsize=8)
    ax1.axis("off")

    # Subplot 2: Forehead context (grayscale)
    ax2 = axes[1]
    if context.size > 0:
        ax2.imshow(context, cmap="gray")
    ax2.set_title("Canny Context (Grayscale)")
    ax2.axis("off")

    # Subplot 3: Canny edge map
    ax3 = axes[2]
    if edges.size > 0:
        ax3.imshow(edges, cmap="gray")
        # Highlight center column
        face_center_x = fx + fw // 2
        center_col_idx = face_center_x - ctx_xs
        if 0 <= center_col_idx < edges.shape[1]:
            ax3.axvline(
                x=center_col_idx,
                color="red",
                linestyle="-",
                linewidth=1.5,
                label="Center column",
            )
    ax3.set_title("Canny Edge Map")
    ax3.axis("off")

    # Subplot 4: Center column values
    ax4 = axes[3]
    if len(center_column) > 0:
        row_indices = list(range(len(center_column)))
        bars = ax4.bar(row_indices, center_column, color="steelblue", width=1.0)

        # Highlight first edge row
        if first_edge_idx >= 0:
            bars[first_edge_idx].set_color("orange")
            ax4.annotate(
                f"First edge (bottom-to-up): row {first_edge_idx}",
                xy=(first_edge_idx, center_column[first_edge_idx]),
                xytext=(first_edge_idx + 5, 200),
                fontsize=9,
                color="orange",
                arrowprops=dict(arrowstyle="->", color="orange"),
            )
    ax4.set_title("Center Column Values")
    ax4.set_xlabel("Row index (within ROI)")
    ax4.set_ylabel("Edge value (0 or 255)")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # 7. Save figure
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
