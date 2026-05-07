"""Visualize facial landmarks overlaid on the original image."""

from __future__ import annotations

from pathlib import Path

import cv2

from visagism.constants import (
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

        # Draw points and connections for each region
        for region_name, color in REGION_COLORS.items():
            pts = landmarks.landmarks_by_region[region_name]

            # Draw points
            for pt in pts:
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
                    cv2.line(
                        annotated,
                        pts[i],
                        pts[j],
                        color,
                        LANDMARK_LINE_THICKNESS,
                    )

        # Add legend
        self._add_legend(annotated)

        return annotated

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
