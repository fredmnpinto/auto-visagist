"""Visualize facial landmarks overlaid on the original image."""

from __future__ import annotations

from pathlib import Path

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
    REGION_LABELS,
)
from visagism.types import FacialLandmarks, ImageArray


class LandmarkVisualizer:
    """Draws facial landmarks, connections, and a legend on the image.

    Supports both display (``show``) and file output (``save``).
    """

    def draw_landmarks(
        self,
        img_bgr: ImageArray,
        landmarks: FacialLandmarks,
    ) -> ImageArray:
        """Draw landmark points and connections on a copy of the image.

        Each facial region is drawn with a distinct colour.
        A legend is added in the top-left corner.

        If more than 50% of the 68 points are ``(-1, -1)`` (sparse
        landmark set), the point radius and line thickness are increased
        to make the visible anchors more prominent.

        Parameters
        ----------
        img_bgr : ImageArray
            Original BGR image.
        landmarks : FacialLandmarks
            Detected landmark data.

        Returns
        -------
        ImageArray
            Annotated image with landmarks, connections, and legend.
        """
        annotated = img_bgr.copy()

        # Determine if this is a sparse landmark set (>50% missing)
        missing_count = sum(1 for pt in landmarks.landmarks_68 if pt == (-1, -1))
        is_sparse = missing_count > 34

        radius = LANDMARK_POINT_RADIUS * 2 if is_sparse else LANDMARK_POINT_RADIUS
        thickness = (
            LANDMARK_LINE_THICKNESS + 1
            if is_sparse
            else LANDMARK_LINE_THICKNESS
        )

        # Draw points and connections for each region
        for region_name, color in REGION_COLORS.items():
            pts = landmarks.landmarks_by_region[region_name]

            # Draw points
            for pt in pts:
                if pt == (-1, -1):
                    continue
                cv2.circle(
                    annotated,
                    pt,
                    radius,
                    color,
                    -1,  # filled
                )

            # Draw connections
            connections = REGION_CONNECTIONS.get(region_name, [])
            for i, j in connections:
                if i < len(pts) and j < len(pts):
                    if pts[i] == (-1, -1) or pts[j] == (-1, -1):
                        continue
                    cv2.line(
                        annotated,
                        pts[i],
                        pts[j],
                        color,
                        thickness,
                    )

        # Draw hairline if available
        annotated = self.draw_hairline(annotated, landmarks)

        # Add legend
        self._add_legend(annotated)

        return annotated

    def draw_hairline(
        self,
        img_bgr: ImageArray,
        landmarks: FacialLandmarks,
    ) -> ImageArray:
        """Draw a dashed horizontal line at the hairline position.

        Parameters
        ----------
        img_bgr : ImageArray
            Image to draw on.
        landmarks : FacialLandmarks
            Detected landmark data. If ``hairline_y`` is None, the image
            is returned unchanged.

        Returns
        -------
        ImageArray
            Image with hairline drawn (or original if no hairline).
        """
        if landmarks.hairline_y is None:
            return img_bgr

        fx, fy, fw, fh = landmarks.face_rect
        y = landmarks.hairline_y
        x_start = fx
        x_end = fx + fw

        # Draw dashed line: segments of length HAIRLINE_DASH_LENGTH
        # separated by gaps of length HAIRLINE_GAP_LENGTH
        x = x_start
        while x < x_end:
            seg_end = min(x + HAIRLINE_DASH_LENGTH, x_end)
            cv2.line(
                img_bgr,
                (x, y),
                (seg_end, y),
                HAIRLINE_COLOR,
                HAIRLINE_LINE_THICKNESS,
            )
            x = seg_end + HAIRLINE_GAP_LENGTH

        return img_bgr

    def _add_legend(self, img: ImageArray) -> None:
        """Add a semi-transparent legend box in the top-left corner.

        Parameters
        ----------
        img : ImageArray
            Image to draw the legend on (modified in-place).
        """
        h, w = img.shape[:2]
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

    def show(self, img: ImageArray, title: str = "Facial Landmarks") -> None:
        """Display the annotated image in a window.

        Only works if a GUI display is available (not headless).

        Parameters
        ----------
        img : ImageArray
            Image to display.
        title : str
            Window title (default: "Facial Landmarks").
        """
        if self._gui_available():
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(title, img)
            print("Press any key to close the visualization window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(
                "Warning: No GUI display available. "
                "Use --save-viz to save to file."
            )

    def save(self, img: ImageArray, output_dir: Path, stem: str) -> Path:
        """Save the annotated image to a file.

        Parameters
        ----------
        img : ImageArray
            Annotated image to save.
        output_dir : Path
            Directory to save the file in. Created if it does not exist.
        stem : str
            Filename stem (without extension), typically the input file stem.

        Returns
        -------
        Path
            The path to the saved file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"landmarks_{stem}.jpg"
        cv2.imwrite(str(save_path), img)
        return save_path

    @staticmethod
    def _gui_available() -> bool:
        """Check whether a GUI display is available.

        Attempts to create and destroy a named window. If the OpenCV
        error is raised, we are in a headless environment.

        Returns
        -------
        bool
            ``True`` if GUI display is available.
        """
        try:
            cv2.namedWindow("_probe_")
            cv2.destroyWindow("_probe_")
            return True
        except cv2.error:
            return False
