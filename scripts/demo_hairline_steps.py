"""Demo script visualising each step of the hairline detection algorithm.

Displays 6 sequential windows showing the full pipeline, one step at a time,
useful for live MSc assignment presentations, debugging, and documentation.

Steps 4 and 5 provide an interactive combined visualization that correlates
the forehead ROI image with its intensity / gradient profile graphs.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

# Allow imports from the project root when running this script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visagism.constants import HAIRLINE_MIN_GRADIENT_RATIO  # noqa: E402
from visagism.errors import ModelNotFoundError, VisagismError  # noqa: E402
from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.hairline_detector import HairlineDetector  # noqa: E402
from visagism.image_loader import ImageLoader  # noqa: E402
from visagism.landmark_detector import LandmarkDetector  # noqa: E402
from visagism.model_finder import ModelFinder  # noqa: E402
from visagism.types import FacialLandmarks, ImageArray  # noqa: E402

# ---------------------------------------------------------------------------
# Layout constants for the combined interactive visualization
# ---------------------------------------------------------------------------
_ROI_MIN_SCALE: int = 2
_ROI_TARGET_WIDTH: int = 600
_GRAPH_HEIGHT: int = 220
_TEXT_HEIGHT: int = 50
_PADDING: int = 20
_AUTO_DELAY_MS: int = 100


def _get_default_image_path() -> Path:
    """Return the first image found in ``test_images/``.

    Returns
    -------
    Path
        Path to the first ``.jpg`` or ``.png`` in the test images directory.

    Raises
    ------
    FileNotFoundError
        If no test images are found.
    """
    test_images_dir = _PROJECT_ROOT / "test_images"
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        candidates = sorted(test_images_dir.glob(ext))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No test images found in {test_images_dir}")


def _show_step(
    window_name: str,
    image: np.ndarray,
    description: str,
    step_number: int,
    is_last: bool = False,
    save_path: Path | None = None,
    visualize: bool = True,
) -> None:
    """Display a single step window and wait for a key press.

    Parameters
    ----------
    window_name : str
        Name for the OpenCV window.
    image : np.ndarray
        Image to display (BGR format).
    description : str
        Human-readable description of the step.
    step_number : int
        Step number for console output.
    is_last : bool, optional
        If True, prompt says "Press any key to exit..." instead of
        "Press any key to continue...".
    save_path : Path | None, optional
        If provided, save the image to this path before displaying.
    visualize : bool, optional
        If True, display the OpenCV window. Default is True.
    """
    if save_path is not None:
        cv2.imwrite(str(save_path), image)
        print(f"    Saved: {save_path}")

    prompt = (
        f"Step {step_number}: {description} — Press any key to exit..."
        if is_last
        else f"Step {step_number}: {description} — Press any key to continue..."
    )
    print(f"\n{prompt}")

    if visualize:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def _create_step1_image(
    img_bgr: ImageArray,
    landmarks: FacialLandmarks,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> np.ndarray:
    """Create Step 1 image: original with face box, eyebrow line, and ROI shaded.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    landmarks : FacialLandmarks
        Detected facial landmarks.
    x_start, x_end, y_start, y_end : int
        ROI coordinates in original image space.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    img_display = img_bgr.copy()

    # Face bounding box (green)
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img_display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    # Average eyebrow line (red horizontal)
    left_brow = landmarks.landmarks_by_region["left_eyebrow"]
    right_brow = landmarks.landmarks_by_region["right_eyebrow"]
    all_brow_pts = left_brow + right_brow
    avg_eyebrow_y = int(np.mean([pt[1] for pt in all_brow_pts]))
    cv2.line(
        img_display,
        (x_start, avg_eyebrow_y),
        (x_end, avg_eyebrow_y),
        (0, 0, 255),
        2,
    )

    # Forehead ROI shading (light blue, semi-transparent)
    overlay = img_display.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (255, 200, 100), -1)
    cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
    cv2.rectangle(img_display, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

    # Text labels
    cv2.putText(
        img_display,
        "Face bounding box",
        (fx + fw + 10, fy + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img_display,
        "Eyebrow line",
        (x_end + 10, avg_eyebrow_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        img_display,
        "Forehead ROI",
        (x_start + 10, y_start + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    return img_display


def _create_step2_image(roi: np.ndarray) -> np.ndarray:
    """Create Step 2 image: raw forehead ROI with dimensions text.

    Parameters
    ----------
    roi : np.ndarray
        Raw grayscale forehead ROI.

    Returns
    -------
    np.ndarray
        BGR image with text overlay.
    """
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    text = f"{w}x{h} px"

    # Draw text with background for readability
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    tx, ty = 10, h - 15
    cv2.rectangle(
        roi_bgr,
        (tx - 5, ty - th - 5),
        (tx + tw + 5, ty + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        roi_bgr,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return roi_bgr


def _create_step3_image(roi_enhanced: np.ndarray) -> np.ndarray:
    """Create Step 3 image: CLAHE-enhanced ROI.

    Parameters
    ----------
    roi_enhanced : np.ndarray
        CLAHE-enhanced grayscale ROI.

    Returns
    -------
    np.ndarray
        BGR image.
    """
    return cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Interactive combined-visualization helpers (Steps 4 & 5)
# ---------------------------------------------------------------------------

def _make_canvas(roi_enhanced: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
    """Create a blank canvas sized for the combined ROI + graph layout.

    Parameters
    ----------
    roi_enhanced : np.ndarray
        CLAHE-enhanced grayscale ROI.

    Returns
    -------
    tuple
        ``(canvas, roi_scale, roi_disp_w, roi_disp_h, graph_w)``
    """
    roi_h, roi_w = roi_enhanced.shape[:2]
    roi_scale = max(_ROI_MIN_SCALE, _ROI_TARGET_WIDTH // roi_w)
    roi_disp_w = roi_w * roi_scale
    roi_disp_h = roi_h * roi_scale
    graph_w = roi_disp_w

    canvas_w = graph_w + 120  # left margin for y-axis labels
    canvas_h = (
        roi_disp_h + _GRAPH_HEIGHT + _TEXT_HEIGHT + _PADDING * 4
    )
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    return canvas, roi_scale, roi_disp_w, roi_disp_h, graph_w


def _draw_roi_on_canvas(
    canvas: np.ndarray,
    roi_enhanced: np.ndarray,
    roi_scale: int,
    roi_disp_w: int,
    roi_disp_h: int,
    highlight_row: int | None = None,
    overlay_color_map: bool = False,
    row_intensities: np.ndarray | None = None,
) -> None:
    """Draw the scaled ROI on the top portion of the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        Target canvas.
    roi_enhanced : np.ndarray
        CLAHE-enhanced grayscale ROI.
    roi_scale : int
        Integer scale factor.
    roi_disp_w, roi_disp_h : int
        Display dimensions.
    highlight_row : int | None, optional
        Row to highlight with a horizontal red line.
    overlay_color_map : bool, optional
        If True, overlay a semi-transparent color gradient based on
        *row_intensities* (blue = bright, red = dark).
    row_intensities : np.ndarray | None, optional
        Required when *overlay_color_map* is True.
    """
    roi_bgr = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)
    roi_resized = cv2.resize(
        roi_bgr, (roi_disp_w, roi_disp_h), interpolation=cv2.INTER_NEAREST
    )

    y_off = _PADDING
    x_off = _PADDING + 60  # leave left margin for graph y-axis
    canvas[y_off:y_off + roi_disp_h, x_off:x_off + roi_disp_w] = roi_resized

    if overlay_color_map and row_intensities is not None:
        overlay = np.zeros_like(roi_resized)
        roi_h = roi_enhanced.shape[0]
        for row in range(roi_h):
            intensity = float(row_intensities[row])
            norm = intensity / 255.0
            # Blue for bright, red for dark
            b = int(255 * norm)
            g = int(128 * (1.0 - abs(norm - 0.5) * 2))  # slight green in mid
            r = int(255 * (1.0 - norm))
            y1 = row * roi_scale
            y2 = (row + 1) * roi_scale
            cv2.rectangle(overlay, (0, y1), (roi_disp_w, y2), (b, g, r), -1)
        roi_region = canvas[y_off:y_off + roi_disp_h, x_off:x_off + roi_disp_w]
        cv2.addWeighted(overlay, 0.35, roi_region, 0.65, 0, roi_region)

    if highlight_row is not None:
        hy = _PADDING + highlight_row * roi_scale + roi_scale // 2
        cv2.line(
            canvas,
            (x_off, hy),
            (x_off + roi_disp_w, hy),
            (0, 0, 255),
            2,
        )


def _draw_graph_on_canvas(
    canvas: np.ndarray,
    values: np.ndarray,
    graph_w: int,
    y_offset: int,
    highlight_idx: int | None = None,
    max_idx: int | None = None,
    median_value: float | None = None,
    y_max: float | None = None,
    color: Tuple[int, int, int] = (180, 130, 70),
    title: str = "",
    y_label: str = "",
) -> None:
    """Draw a line graph on the canvas using OpenCV primitives.

    Parameters
    ----------
    canvas : np.ndarray
        Target canvas.
    values : np.ndarray
        1-D array of values to plot.
    graph_w : int
        Width of the graph in pixels.
    y_offset : int
        Top y-coordinate of the graph area.
    highlight_idx : int | None, optional
        Index to highlight with a vertical red line.
    max_idx : int | None, optional
        Index of the maximum value to mark with a red circle.
    median_value : float | None, optional
        Value at which to draw an orange dashed horizontal line.
    y_max : float | None, optional
        Maximum value for y-axis scaling. If None, uses ``np.max(values)``.
    color : tuple[int, int, int], optional
        BGR color for the data line.
    title : str, optional
        Graph title.
    y_label : str, optional
        Label for the y-axis.
    """
    x_off = _PADDING + 60
    graph_h = _GRAPH_HEIGHT

    # Background
    cv2.rectangle(
        canvas,
        (x_off, y_offset),
        (x_off + graph_w, y_offset + graph_h),
        (35, 35, 35),
        -1,
    )

    # Axes
    cv2.line(
        canvas,
        (x_off, y_offset + graph_h),
        (x_off + graph_w, y_offset + graph_h),
        (180, 180, 180),
        1,
    )
    cv2.line(
        canvas,
        (x_off, y_offset),
        (x_off, y_offset + graph_h),
        (180, 180, 180),
        1,
    )

    # Grid lines
    for i in range(1, 5):
        gy = y_offset + int(graph_h * i / 5)
        cv2.line(
            canvas, (x_off, gy), (x_off + graph_w, gy), (60, 60, 60), 1
        )

    # Data line
    n = len(values)
    vmax = y_max if y_max is not None else float(np.max(values))
    if vmax == 0:
        vmax = 1.0

    points: list[Tuple[int, int]] = []
    for i, val in enumerate(values):
        px = (
            x_off + int((i / (n - 1)) * graph_w)
            if n > 1
            else x_off + graph_w // 2
        )
        py = y_offset + graph_h - int((val / vmax) * (graph_h - 20)) - 10
        points.append((px, py))

    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], color, 2)

    # Highlight vertical line
    if highlight_idx is not None and 0 <= highlight_idx < n:
        hx = (
            x_off + int((highlight_idx / (n - 1)) * graph_w)
            if n > 1
            else x_off + graph_w // 2
        )
        cv2.line(
            canvas, (hx, y_offset), (hx, y_offset + graph_h), (0, 0, 255), 2
        )

    # Max point marker
    if max_idx is not None and 0 <= max_idx < n:
        mx = (
            x_off + int((max_idx / (n - 1)) * graph_w)
            if n > 1
            else x_off + graph_w // 2
        )
        my = y_offset + graph_h - int((values[max_idx] / vmax) * (graph_h - 20)) - 10
        cv2.circle(canvas, (mx, my), 6, (0, 0, 255), -1)

    # Median dashed line
    if median_value is not None:
        my_line = (
            y_offset
            + graph_h
            - int((median_value / vmax) * (graph_h - 20))
            - 10
        )
        for dx in range(0, graph_w, 12):
            x1 = x_off + dx
            x2 = min(x_off + dx + 6, x_off + graph_w)
            cv2.line(canvas, (x1, my_line), (x2, my_line), (0, 140, 255), 2)

    # Title
    if title:
        cv2.putText(
            canvas,
            title,
            (x_off, y_offset - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    # Y label (rotated text simulated as horizontal on the left margin)
    if y_label:
        cv2.putText(
            canvas,
            y_label,
            (5, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    # Y-axis tick labels
    for i in range(0, 3):
        tick_val = vmax * (i / 2.0)
        ty = y_offset + graph_h - int((tick_val / vmax) * (graph_h - 20)) - 10
        label_text = f"{tick_val:.0f}"
        cv2.putText(
            canvas,
            label_text,
            (x_off - 50, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )


def _draw_text_bar(
    canvas: np.ndarray,
    text: str,
    y_offset: int,
) -> None:
    """Draw a text bar at the bottom of the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        Target canvas.
    text : str
        Text to display.
    y_offset : int
        Top y-coordinate of the text area.
    """
    cv2.putText(
        canvas,
        text,
        (_PADDING + 60, y_offset + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def _run_step4_interactive(
    window_name: str,
    roi_enhanced: np.ndarray,
    row_intensities: np.ndarray,
    step_number: int,
    save_path: Path | None,
    auto: bool,
    visualize: bool,
) -> None:
    """Run the interactive Step 4: intensity profile with correlated ROI.

    Parameters
    ----------
    window_name : str
        OpenCV window name.
    roi_enhanced : np.ndarray
        CLAHE-enhanced grayscale ROI.
    row_intensities : np.ndarray
        Average intensity per row.
    step_number : int
        Step number for console output.
    save_path : Path | None
        Path to save the final summary frame.
    auto : bool
        If True, auto-advance without waiting for key press.
    visualize : bool
        If True, display the OpenCV window.
    """
    canvas, roi_scale, roi_disp_w, roi_disp_h, graph_w = _make_canvas(roi_enhanced)
    roi_h = roi_enhanced.shape[0]
    row_increment = max(10, int(roi_h * 0.05))

    print(f"\nStep {step_number}: Average intensity per row — Interactive")
    print("  Press any key to advance rows. Auto-advance is", "ON" if auto else "OFF")

    # Animation loop
    current_row = 0
    while current_row < roi_h:
        canvas[:] = (20, 20, 20)

        _draw_roi_on_canvas(
            canvas, roi_enhanced, roi_scale, roi_disp_w, roi_disp_h,
            highlight_row=current_row,
        )

        graph_y = _PADDING + roi_disp_h + _PADDING
        _draw_graph_on_canvas(
            canvas,
            row_intensities,
            graph_w,
            graph_y,
            highlight_idx=current_row,
            y_max=255.0,
            color=(180, 130, 70),
            title="Average intensity per row",
            y_label="Intensity",
        )

        text_y = graph_y + _GRAPH_HEIGHT + _PADDING
        intensity_val = float(row_intensities[current_row])
        status_text = (
            f"Row {current_row}/{roi_h} | Intensity: {intensity_val:.1f} | "
            f"Press any key to scan..."
        )
        _draw_text_bar(canvas, status_text, text_y)

        if visualize:
            cv2.imshow(window_name, canvas)
            delay = _AUTO_DELAY_MS if auto else 0
            key = cv2.waitKey(delay)
            if not auto and key == 27:  # ESC aborts
                break
        current_row += row_increment

    # Final summary frame
    canvas[:] = (20, 20, 20)
    min_idx = int(np.argmin(row_intensities))

    _draw_roi_on_canvas(
        canvas, roi_enhanced, roi_scale, roi_disp_w, roi_disp_h,
        highlight_row=min_idx,
        overlay_color_map=True,
        row_intensities=row_intensities,
    )

    graph_y = _PADDING + roi_disp_h + _PADDING
    _draw_graph_on_canvas(
        canvas,
        row_intensities,
        graph_w,
        graph_y,
        highlight_idx=min_idx,
        y_max=255.0,
        color=(180, 130, 70),
        title="Average intensity per row  (FINAL)",
        y_label="Intensity",
    )

    text_y = graph_y + _GRAPH_HEIGHT + _PADDING
    min_val = float(row_intensities[min_idx])
    summary_text = (
        f"MIN INTENSITY at row {min_idx} = {min_val:.1f} (darkest = likely hair) | "
        f"Press any key to continue..."
    )
    _draw_text_bar(canvas, summary_text, text_y)

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)
        print(f"    Saved: {save_path}")

    if visualize:
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def _run_step5_interactive(
    window_name: str,
    roi_enhanced: np.ndarray,
    abs_gradient: np.ndarray,
    max_gradient_idx: int,
    max_gradient_value: float,
    median_gradient: float,
    step_number: int,
    save_path: Path | None,
    auto: bool,
    visualize: bool,
) -> None:
    """Run the interactive Step 5: gradient profile with correlated ROI.

    Parameters
    ----------
    window_name : str
        OpenCV window name.
    roi_enhanced : np.ndarray
        CLAHE-enhanced grayscale ROI.
    abs_gradient : np.ndarray
        Absolute gradient per row.
    max_gradient_idx : int
        Index of maximum gradient.
    max_gradient_value : float
        Maximum gradient value.
    median_gradient : float
        Median gradient value.
    step_number : int
        Step number for console output.
    save_path : Path | None
        Path to save the final summary frame.
    auto : bool
        If True, auto-advance without waiting for key press.
    visualize : bool
        If True, display the OpenCV window.
    """
    canvas, roi_scale, roi_disp_w, roi_disp_h, graph_w = _make_canvas(roi_enhanced)
    roi_h = roi_enhanced.shape[0]
    row_increment = max(10, int(roi_h * 0.05))

    print(f"\nStep {step_number}: Vertical gradient — Interactive")
    print("  Press any key to advance rows. Auto-advance is", "ON" if auto else "OFF")

    # Animation loop
    current_row = 0
    while current_row < roi_h:
        canvas[:] = (20, 20, 20)

        _draw_roi_on_canvas(
            canvas, roi_enhanced, roi_scale, roi_disp_w, roi_disp_h,
            highlight_row=current_row,
        )

        graph_y = _PADDING + roi_disp_h + _PADDING
        _draw_graph_on_canvas(
            canvas,
            abs_gradient,
            graph_w,
            graph_y,
            highlight_idx=current_row,
            y_max=max_gradient_value,
            color=(180, 130, 70),
            title="Vertical gradient magnitude",
            y_label="Gradient",
        )

        text_y = graph_y + _GRAPH_HEIGHT + _PADDING
        grad_val = float(abs_gradient[current_row])
        status_text = (
            f"Row {current_row}/{roi_h} | Gradient: {grad_val:.2f} | "
            f"Press any key to scan..."
        )
        _draw_text_bar(canvas, status_text, text_y)

        if visualize:
            cv2.imshow(window_name, canvas)
            delay = _AUTO_DELAY_MS if auto else 0
            key = cv2.waitKey(delay)
            if not auto and key == 27:  # ESC aborts
                break
        current_row += row_increment

    # Final summary frame
    canvas[:] = (20, 20, 20)

    _draw_roi_on_canvas(
        canvas, roi_enhanced, roi_scale, roi_disp_w, roi_disp_h,
        highlight_row=max_gradient_idx,
    )
    # Thick red line for max gradient
    x_off = _PADDING + 60
    hy_max = _PADDING + max_gradient_idx * roi_scale + roi_scale // 2
    cv2.line(
        canvas,
        (x_off, hy_max),
        (x_off + roi_disp_w, hy_max),
        (0, 0, 255),
        4,
    )

    # Thin orange line for median (approximate row where gradient == median)
    median_diff = np.abs(abs_gradient - median_gradient)
    median_idx = int(np.argmin(median_diff))
    hy_med = _PADDING + median_idx * roi_scale + roi_scale // 2
    cv2.line(
        canvas,
        (x_off, hy_med),
        (x_off + roi_disp_w, hy_med),
        (0, 140, 255),
        2,
    )

    graph_y = _PADDING + roi_disp_h + _PADDING
    _draw_graph_on_canvas(
        canvas,
        abs_gradient,
        graph_w,
        graph_y,
        highlight_idx=max_gradient_idx,
        max_idx=max_gradient_idx,
        median_value=median_gradient,
        y_max=max_gradient_value,
        color=(180, 130, 70),
        title="Vertical gradient magnitude  (FINAL)",
        y_label="Gradient",
    )

    text_y = graph_y + _GRAPH_HEIGHT + _PADDING
    ratio = max_gradient_value / median_gradient
    summary_text = (
        f"MAX GRADIENT at row {max_gradient_idx} = {max_gradient_value:.2f} | "
        f"Median = {median_gradient:.2f} | Ratio = {ratio:.2f} | "
        f"Press any key to continue..."
    )
    _draw_text_bar(canvas, summary_text, text_y)

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)
        print(f"    Saved: {save_path}")

    if visualize:
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def _create_step6_image(
    img_bgr: ImageArray,
    landmarks: FacialLandmarks,
    hairline_y: int,
    method: str,
    gradient_ratio: float,
) -> np.ndarray:
    """Create Step 6 image: final result with detected hairline and annotations.

    Parameters
    ----------
    img_bgr : ImageArray
        Original BGR image.
    landmarks : FacialLandmarks
        Detected facial landmarks (used for eyebrow line and face box).
    hairline_y : int
        Detected hairline y-coordinate.
    method : str
        Detection method (``"edge"`` or ``"fallback"``).
    gradient_ratio : float
        Gradient ratio for confidence display.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    img_display = img_bgr.copy()
    h, w = img_display.shape[:2]

    # Face bounding box (green)
    fx, fy, fw, fh = landmarks.face_rect
    cv2.rectangle(img_display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    # Average eyebrow line (red horizontal)
    left_brow = landmarks.landmarks_by_region["left_eyebrow"]
    right_brow = landmarks.landmarks_by_region["right_eyebrow"]
    all_brow_pts = left_brow + right_brow
    avg_eyebrow_y = int(np.mean([pt[1] for pt in all_brow_pts]))
    cv2.line(
        img_display,
        (0, avg_eyebrow_y),
        (w, avg_eyebrow_y),
        (0, 0, 255),
        2,
    )

    # Draw dashed hairline
    color = (0, 255, 255) if method == "edge" else (0, 165, 255)  # yellow or orange
    label = "Detected hairline" if method == "edge" else "Fallback hairline"
    dash_length = 20
    for x in range(0, w, dash_length * 2):
        x1 = min(x + dash_length, w)
        cv2.line(img_display, (x, hairline_y), (x1, hairline_y), color, 3)

    # Result text
    result_text = f"Detected hairline: Y={hairline_y} (method: {method})"
    (tw, th), _ = cv2.getTextSize(
        result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    tx, ty = 10, h - 20
    cv2.rectangle(
        img_display,
        (tx - 5, ty - th - 5),
        (tx + tw + 5, ty + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img_display,
        result_text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Confidence / ratio text
    ratio_text = (
        f"Gradient ratio: {gradient_ratio:.2f} "
        f"(threshold: {HAIRLINE_MIN_GRADIENT_RATIO})"
    )
    (rtw, rth), _ = cv2.getTextSize(
        ratio_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    rty = ty - th - 15
    cv2.rectangle(
        img_display,
        (tx - 5, rty - rth - 5),
        (tx + rtw + 5, rty + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img_display,
        ratio_text,
        (tx, rty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if method == "edge" else (0, 165, 255),
        2,
    )

    if method == "fallback":
        fallback_text = "Fallback used"
        (ftw, fth), _ = cv2.getTextSize(
            fallback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        fty = rty - rth - 15
        cv2.rectangle(
            img_display,
            (tx - 5, fty - fth - 5),
            (tx + ftw + 5, fty + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img_display,
            fallback_text,
            (tx, fty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

    # Legend
    cv2.putText(
        img_display,
        label,
        (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    cv2.putText(
        img_display,
        "Face box",
        (w - 250, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        img_display,
        "Eyebrow line",
        (w - 250, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    return img_display


def process_image(
    image_path: Path,
    output_dir: Path,
    save_steps: bool,
    auto: bool,
    visualize: bool,
) -> None:
    """Process a single image and display each step sequentially.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    output_dir : Path
        Directory where output images are saved (if ``save_steps`` is True).
    save_steps : bool
        If True, save each step image to the output directory.
    auto : bool
        If True, interactive steps auto-advance without key press.
    visualize : bool
        If True, display OpenCV windows for each step.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 60}")

    # 1. Load image
    img_bgr, img_gray = ImageLoader.load(image_path)
    img_h, img_w = img_gray.shape[:2]

    # 2. Detect face
    face_detector = FaceDetector()
    face_rect = face_detector.detect(img_gray, (img_h, img_w))
    fx, fy, fw, fh = face_rect

    # 3. Detect landmarks
    model_path = ModelFinder.find()
    landmark_detector = LandmarkDetector(model_path)
    landmarks = landmark_detector.detect(img_gray, face_rect, image_path)

    # 4. Get all intermediate data from HairlineDetector
    hairline_detector = HairlineDetector()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = hairline_detector.debug_steps(img_gray, landmarks)

    roi = steps["roi_raw"]
    roi_enhanced = steps["roi_enhanced"]
    row_intensities = steps["row_intensities"]
    abs_gradient = steps["abs_gradient"]
    max_gradient_idx = steps["max_gradient_idx"]
    max_gradient_value = steps["max_gradient_value"]
    median_gradient = steps["median_gradient"]
    gradient_ratio = steps["gradient_ratio"]
    hairline_y = steps["hairline_y"]
    method = steps["method"]
    x_start, x_end, y_start, y_end = steps["roi_coords"]

    roi_width = x_end - x_start
    roi_height = y_end - y_start

    # 5. Print text summary
    print(f"\n  Image: {image_path.name}")
    print(f"  Face size: {fw}x{fh} px")
    print(f"  Forehead ROI: {roi_width}x{roi_height} px")
    print(f"  Max gradient: {max_gradient_value:.2f} at row {max_gradient_idx}")
    print(f"  Median gradient: {median_gradient:.2f}")
    print(f"  Ratio: {gradient_ratio:.2f}")
    print(f"  Result: {method} detection at y={hairline_y}")

    # 9. Display steps sequentially
    def _save_path(step_num: int, step_name: str) -> Path | None:
        if not save_steps:
            return None
        return (
            output_dir
            / f"demo_hairline_step{step_num}_{step_name}_{image_path.stem}.png"
        )

    # Step 1: Original image with annotations
    step1_img = _create_step1_image(
        img_bgr, landmarks, x_start, x_end, y_start, y_end
    )
    _show_step(
        "Step 1: Detect face & eyebrows",
        step1_img,
        "Detect face & eyebrows",
        1,
        save_path=_save_path(1, "face_and_eyebrows"),
        visualize=visualize,
    )

    # Step 2: Raw forehead ROI
    step2_img = _create_step2_image(roi)
    _show_step(
        "Step 2: Extract forehead ROI",
        step2_img,
        "Extract forehead ROI",
        2,
        save_path=_save_path(2, "forehead_roi"),
        visualize=visualize,
    )

    # Step 3: CLAHE-enhanced ROI
    step3_img = _create_step3_image(roi_enhanced)
    _show_step(
        "Step 3: CLAHE enhancement",
        step3_img,
        "CLAHE enhancement",
        3,
        save_path=_save_path(3, "clahe"),
        visualize=visualize,
    )

    # Step 4: Interactive intensity profile
    _run_step4_interactive(
        "Step 4: Average intensity per row",
        roi_enhanced,
        row_intensities,
        4,
        save_path=_save_path(4, "intensity_profile"),
        auto=auto,
        visualize=visualize,
    )

    # Step 5: Interactive gradient profile
    _run_step5_interactive(
        "Step 5: Vertical gradient",
        roi_enhanced,
        abs_gradient,
        max_gradient_idx,
        max_gradient_value,
        median_gradient,
        5,
        save_path=_save_path(5, "gradient_profile"),
        auto=auto,
        visualize=visualize,
    )

    # Step 6: Final result
    step6_img = _create_step6_image(
        img_bgr, landmarks, hairline_y, method, gradient_ratio
    )
    _show_step(
        "Step 6: Detected hairline",
        step6_img,
        "Detected hairline",
        6,
        is_last=True,
        save_path=_save_path(6, "final_result"),
        visualize=visualize,
    )

    if save_steps:
        print(f"\n  All step images saved to: {output_dir}")


def main() -> None:
    """Parse arguments and run the demo on one or more images."""
    parser = argparse.ArgumentParser(
        description="Visualise each step of the hairline detection algorithm."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input image (default: first image in test_images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "output",
        help="Output directory for saved step images (default: ./output)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save each step as an individual image to the output directory",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Automatically advance through interactive steps (Steps 4 & 5) "
            "without waiting for key press"
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display OpenCV windows for each step (default: headless mode)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.visualize:
        print("Running in headless mode. Use --visualize to display windows.")

    # Find dlib model first to fail fast
    try:
        ModelFinder.find()
    except ModelNotFoundError as exc:
        print(f"Model not found: {exc}")
        print("Download the model and try again.")
        sys.exit(1)

    if args.input:
        image_paths = [args.input]
    else:
        # Default: first image in test_images/
        try:
            default_path = _get_default_image_path()
            image_paths = [default_path]
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    for image_path in image_paths:
        try:
            process_image(image_path, output_dir, args.save, args.auto, args.visualize)
        except VisagismError as exc:
            print(f"\n  ERROR processing {image_path.name}: {exc}")
        except Exception as exc:
            print(f"\n  UNEXPECTED ERROR processing {image_path.name}: {exc}")
            raise

    print(f"\n{'=' * 60}")
    print("Demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
