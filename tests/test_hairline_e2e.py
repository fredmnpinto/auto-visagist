"""End-to-end tests for the HairlineDetector using real face images.

These tests validate that the HairlineDetector works correctly on actual
photographs by running the full detection pipeline (image loading → face
detection → landmark detection → hairline detection).

Tests that require the dlib 68-point shape predictor model are skipped
gracefully when the model is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from visagism.face_detector import FaceDetector
from visagism.hairline_detector import HairlineDetector
from visagism.image_loader import ImageLoader
from visagism.landmark_detector import LandmarkDetector
from visagism.types import FacialLandmarks

# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------
MODEL_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "shape_predictor_68_face_landmarks.dat"
)
# Also check ~/.dlib/
MODEL_PATH_ALT = Path.home() / ".dlib" / "shape_predictor_68_face_landmarks.dat"
MODEL_AVAILABLE = (
    (MODEL_PATH.exists() and MODEL_PATH.is_file())
    or (MODEL_PATH_ALT.exists() and MODEL_PATH_ALT.is_file())
)
# Use the one that exists
MODEL_PATH = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH_ALT

# ---------------------------------------------------------------------------
# Test image paths
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
TEST_IMAGE_PATHS: List[Path] = [
    p for p in TEST_IMAGES_DIR.glob("*.jpg")
    if p.is_file()
]


class TestHairlineE2E:
    """End-to-end test suite for HairlineDetector on real photographs."""

    @pytest.mark.skipif(
        not MODEL_AVAILABLE,
        reason="dlib shape predictor model not found at %s" % MODEL_PATH,
    )
    @pytest.mark.parametrize("image_path", TEST_IMAGE_PATHS)
    def test_hairline_detected_on_real_images(self, image_path: Path) -> None:
        """Run full pipeline and verify hairline is detected on real faces.

        For each test image the complete pipeline is executed:
        ``ImageLoader.load()`` → ``FaceDetector.detect()`` →
        ``LandmarkDetector.detect()`` → ``HairlineDetector.detect()``.

        The resulting ``hairline_y`` must be a valid integer within the
        image bounds and positioned above the eyebrows.  The detection
        method must be ``"edge"`` (gradient-based), proving the new
        algorithm works on real photographs.

        Parameters
        ----------
        image_path : Path
            Path to a real face photograph under ``test_images/``.
        """
        # 1. Load image
        img_bgr, img_gray = ImageLoader.load(image_path)
        img_h, img_w = img_gray.shape[:2]

        # 2. Detect face
        face_detector = FaceDetector()
        face_rect = face_detector.detect(img_gray, (img_h, img_w))

        # 3. Detect landmarks
        landmark_detector = LandmarkDetector(MODEL_PATH)
        landmarks = landmark_detector.detect(img_gray, face_rect, image_path)

        # 4. Detect hairline
        hairline_detector = HairlineDetector()
        result = hairline_detector.detect(img_gray, landmarks)
        hairline_y = result["hairline_y"]
        method = result["method"]

        # 5. Assertions
        assert hairline_y is not None
        assert 0 <= hairline_y < img_h, (
            f"hairline_y={hairline_y} out of image bounds [0, {img_h})"
        )

        # Average eyebrow y-coordinate
        left_brow = landmarks.landmarks_by_region["left_eyebrow"]
        right_brow = landmarks.landmarks_by_region["right_eyebrow"]
        all_brow_pts = left_brow + right_brow
        avg_eyebrow_y = int(np.mean([pt[1] for pt in all_brow_pts]))

        assert hairline_y < avg_eyebrow_y, (
            f"hairline_y={hairline_y} should be above avg_eyebrow_y={avg_eyebrow_y}"
        )

        # The gradient method must succeed on real images
        assert method == "edge", (
            f"Expected edge detection but got {method} for {image_path.name}"
        )

        # Compute fallback estimate manually
        left_brow = landmarks.landmarks_by_region["left_eyebrow"]
        right_brow = landmarks.landmarks_by_region["right_eyebrow"]
        all_brow_pts = left_brow + right_brow
        avg_eyebrow_y = int(np.mean([pt[1] for pt in all_brow_pts]))

        nose_tip_y = landmarks.landmarks_68[33][1]
        medium_third = nose_tip_y - avg_eyebrow_y
        fallback_y = max(avg_eyebrow_y - medium_third, face_rect[1])

        # Assert detected hairline is within tolerance of fallback.
        # With the expanded ROI (up to 25% of face height above the face
        # bounding box), the detected hairline can legitimately be higher
        # than the fallback estimate which is clamped to face_rect[1].
        face_height = face_rect[3]
        base_tolerance = max(50, int(0.10 * face_height))
        upward_expansion = int(0.25 * face_height)
        tolerance = base_tolerance + upward_expansion

        assert abs(hairline_y - fallback_y) <= tolerance, (
            f"hairline_y={hairline_y} deviates too far from fallback={fallback_y} "
            f"(tolerance={tolerance}px, face_height={face_height}px)"
        )

    def test_hairline_fallback_when_no_model(self) -> None:
        """Prove the geometric fallback works without real images or dlib model.

        A synthetic ``FacialLandmarks`` object is created with realistic
        landmark positions and a uniform grey image (no edges) is used.
        Because there are no strong intensity gradients, the detector must
        return the fallback estimate.

        This test always runs and does not require the dlib model.
        """
        from visagism.constants import REGION_INDICES
        from visagism.types import LandmarkRegions

        # Build synthetic 68 landmarks with a realistic face layout
        landmarks_68 = [(0, 0)] * 68

        # Eyebrows at y = 150 (points 17-26)
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100 + (idx % 5) * 10, 150)

        # Nose tip (point 33) at y = 200
        landmarks_68[33] = (150, 200)

        # Fill remaining points with plausible coordinates so the structure
        # looks roughly like a face (values are not critical for fallback)
        for i in range(68):
            if landmarks_68[i] == (0, 0):
                landmarks_68[i] = (150, 100 + i * 2)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }

        landmarks = FacialLandmarks(
            image_path=Path("/fake/synthetic.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        # Uniform grey image — no strong gradients
        img_gray = np.full((400, 400), 128, dtype=np.uint8)

        hairline_detector = HairlineDetector()

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = hairline_detector.detect(img_gray, landmarks)

        hairline_y = result["hairline_y"]
        method = result["method"]

        # Expected fallback: max(avg_eyebrow_y - medium_third, face_rect[1])
        # avg_eyebrow_y = 150, nose_tip_y = 200
        # medium_third = 200 - 150 = 50
        # fallback = max(150 - 50, 50) = max(100, 50) = 100
        expected_fallback = 100

        assert hairline_y == expected_fallback, (
            f"Expected fallback {expected_fallback}, got {hairline_y}"
        )
        assert method == "fallback"
