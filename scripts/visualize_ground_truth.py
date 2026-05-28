"""Visualize ground-truth landmarks on test images.

Loads each image from ``test_images/`` and its corresponding ground-truth
JSON from ``ground_truth/``, draws the 68 landmarks (coloured by facial
region), the hairline as a dashed horizontal line, and a legend.  The
result is saved to ``output/ground_truth_visualization/``.

Usage
-----
    python scripts/visualize_ground_truth.py

"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without package install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2

from visagism.constants import (
    HAIRLINE_COLOR,
    HAIRLINE_DASH_LENGTH,
    HAIRLINE_GAP_LENGTH,
    HAIRLINE_LINE_THICKNESS,
    LANDMARK_LINE_THICKNESS,
    LANDMARK_POINT_RADIUS,
    LEGEND_ALPHA,
    LEGEND_BG_COLOR,
    LEGEND_FONT_SCALE,
    LEGEND_FONT_THICKNESS,
    LEGEND_LINE_HEIGHT,
    LEGEND_MARGIN,
    REGION_COLORS,
    REGION_CONNECTIONS,
    REGION_INDICES,
    REGION_LABELS,
)
from visagism.errors import VisagismError
from visagism.landmark_ground_truth import LandmarkGroundTruth


def _build_landmarks_by_region(
    landmarks_68: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    """Group a flat 68-point list into regions.

    Parameters
    ----------
    landmarks_68 : list of tuple
        Full 68 landmark coordinates.

    Returns
    -------
    dict
        Mapping from region name to list of points in that region.
    """
    return {
        region: [landmarks_68[i] for i in indices]
        for region, indices in REGION_INDICES.items()
    }


def _draw_landmarks(
    img: cv2.Mat,
    gt: LandmarkGroundTruth,
) -> cv2.Mat:
    """Draw ground-truth landmarks, connections, hairline, and legend.

    Parameters
    ----------
    img : cv2.Mat
        Original BGR image.
    gt : LandmarkGroundTruth
        Loaded ground-truth data.

    Returns
    -------
    cv2.Mat
        Annotated image.
    """
    annotated = img.copy()
    landmarks_by_region = _build_landmarks_by_region(gt.landmarks_68)

    # Draw points and connections for each region
    for region_name, color in REGION_COLORS.items():
        pts = landmarks_by_region.get(region_name, [])

        # Draw points (skip unplaced landmarks)
        for pt in pts:
            if pt == (-1, -1):
                continue
            cv2.circle(
                annotated,
                pt,
                LANDMARK_POINT_RADIUS,
                color,
                -1,  # filled
            )

        # Draw connections
        connections = REGION_CONNECTIONS.get(region_name, [])
        for i, j in connections:
            if i < len(pts) and j < len(pts):
                p1, p2 = pts[i], pts[j]
                if p1 == (-1, -1) or p2 == (-1, -1):
                    continue
                cv2.line(
                    annotated,
                    p1,
                    p2,
                    color,
                    LANDMARK_LINE_THICKNESS,
                )

    # Draw hairline if available
    if gt.hairline_y is not None:
        y = gt.hairline_y
        x_start = 0
        x_end = annotated.shape[1]
        x = x_start
        while x < x_end:
            seg_end = min(x + HAIRLINE_DASH_LENGTH, x_end)
            cv2.line(
                annotated,
                (x, y),
                (seg_end, y),
                HAIRLINE_COLOR,
                HAIRLINE_LINE_THICKNESS,
            )
            x = seg_end + HAIRLINE_GAP_LENGTH

    # Add legend
    _add_legend(annotated)

    return annotated


def _add_legend(img: cv2.Mat) -> None:
    """Add a semi-transparent legend box in the top-left corner.

    Parameters
    ----------
    img : cv2.Mat
        Image to draw the legend on (modified in-place).
    """
    num_items = len(REGION_LABELS)
    box_h = LEGEND_MARGIN * 2 + num_items * LEGEND_LINE_HEIGHT
    box_w = 160

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (LEGEND_MARGIN, LEGEND_MARGIN),
        (LEGEND_MARGIN + box_w, LEGEND_MARGIN + box_h),
        LEGEND_BG_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, LEGEND_ALPHA, img, 1.0 - LEGEND_ALPHA, 0, img)

    # Draw items
    for i, (region_name, label) in enumerate(REGION_LABELS.items()):
        y = LEGEND_MARGIN + LEGEND_LINE_HEIGHT + i * LEGEND_LINE_HEIGHT
        color = REGION_COLORS[region_name]

        # Colour swatch
        cv2.circle(
            img,
            (LEGEND_MARGIN + 8, y - 4),
            4,
            color,
            -1,
        )

        # Label text
        cv2.putText(
            img,
            label,
            (LEGEND_MARGIN + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            LEGEND_FONT_SCALE,
            (255, 255, 255),
            LEGEND_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def main() -> int:
    """Run the ground-truth visualization script.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    project_root = Path(__file__).resolve().parent.parent
    test_images_dir = project_root / "test_images"
    ground_truth_dir = project_root / "ground_truth"
    output_dir = project_root / "output" / "ground_truth_visualization"

    if not test_images_dir.exists():
        print(f"Error: test_images directory not found: {test_images_dir}")
        return 1

    if not ground_truth_dir.exists():
        print(f"Error: ground_truth directory not found: {ground_truth_dir}")
        return 1

    # Collect image / ground-truth pairs
    image_paths = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    processed = 0
    skipped = 0

    for img_path in image_paths:
        gt_path = ground_truth_dir / f"{img_path.stem}_gt.json"
        if not gt_path.exists():
            print(f"  Skipping {img_path.name}: no ground truth found ({gt_path.name})")
            skipped += 1
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name}: could not read image")
            skipped += 1
            continue

        # Load ground truth
        try:
            gt = LandmarkGroundTruth.load(gt_path)
        except VisagismError as exc:
            print(f"  Skipping {img_path.name}: failed to load ground truth: {exc}")
            skipped += 1
            continue

        # Draw
        annotated = _draw_landmarks(img, gt)

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{img_path.stem}_gt.png"
        cv2.imwrite(str(save_path), annotated)
        print(f"  Saved: {save_path}")
        processed += 1

    print(f"\nDone. Processed {processed} image(s), skipped {skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
