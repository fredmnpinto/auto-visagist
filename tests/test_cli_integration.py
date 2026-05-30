"""Integration tests for the CLI pipeline in visagism.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visagism.errors import AnalysisError
from visagism.types import FacialLandmarks, LandmarkRegions


def _import_visagism_py():
    """Import visagism.py as a module using importlib.util.

    Returns
    -------
    module
        The visagism.py module object.
    """
    spec = importlib.util.spec_from_file_location(
        "visagism_cli", str(Path(__file__).parent.parent / "visagism.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_config(tmp_path: Path) -> MagicMock:
    """Return a mock CliConfig with sensible defaults."""
    config = MagicMock()
    config.input_path = tmp_path / "photo.jpg"
    config.output_dir = tmp_path / "output"
    config.model_path = None
    config.visualize = False
    config.save_viz = False
    config.kernel_size = 7
    config.canny_low = 30
    config.canny_high = 60
    return config


@pytest.fixture
def mock_landmarks() -> FacialLandmarks:
    """Return realistic FacialLandmarks for from_landmarks()."""
    pts = []
    # Jaw: 0-16
    for i in range(17):
        x = 100 + i * 10
        y = 300 + abs(i - 8) * 5
        pts.append((x, y))
    # Left eyebrow: 17-21
    pts.extend([(110, 120), (125, 115), (140, 110), (155, 115), (170, 120)])
    # Right eyebrow: 22-26
    pts.extend([(180, 120), (195, 115), (210, 110), (225, 115), (240, 120)])
    # Nose bridge: 27-30
    pts.extend([(175, 110), (175, 130), (175, 150), (175, 170)])
    # Nose tip: 31-35
    pts.extend([(165, 190), (170, 195), (175, 200), (180, 195), (185, 190)])
    # Left eye: 36-41
    pts.extend([
        (120, 150), (130, 140), (140, 140),
        (150, 150), (140, 160), (130, 160),
    ])
    # Right eye: 42-47
    pts.extend([
        (200, 150), (210, 140), (220, 140),
        (230, 150), (220, 160), (210, 160),
    ])
    # Outer mouth: 48-59
    pts.extend([
        (150, 240), (160, 230), (175, 225), (190, 230),
        (200, 235), (210, 240), (250, 240), (210, 250),
        (200, 255), (190, 250), (175, 255), (160, 250),
    ])
    # Inner mouth: 60-67
    pts.extend([
        (160, 240), (170, 235), (180, 240), (175, 245),
        (165, 245), (160, 240), (160, 240), (160, 240),
    ])

    from visagism.constants import REGION_INDICES

    landmarks_by_region: LandmarkRegions = {
        name: [pts[i] for i in indices]
        for name, indices in REGION_INDICES.items()
    }

    return FacialLandmarks(
        image_path=Path("/fake/test.jpg"),
        face_rect=(100, 100, 150, 200),
        landmarks_68=pts,
        landmarks_by_region=landmarks_by_region,
        hairline_y=80,
    )


@pytest.fixture
def mock_landmarks_no_hairline(mock_landmarks: FacialLandmarks) -> FacialLandmarks:
    """Return landmarks without hairline_y."""
    from dataclasses import replace
    return replace(mock_landmarks, hairline_y=None)


class TestCliIntegration:
    """Test suite for CLI integration of calculator and report formatter."""

    @pytest.fixture(autouse=True)
    def patch_pipeline(
        self,
        mock_config: MagicMock,
        mock_landmarks: FacialLandmarks,
    ) -> Generator[None, None, None]:
        """Patch the entire CV pipeline for all tests in this class."""
        img_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
        img_gray = np.zeros((400, 400), dtype=np.uint8)

        with (
            patch("visagism.cli.CliParser.parse", return_value=mock_config),
            patch(
                "visagism.image_loader.ImageLoader.load",
                return_value=(img_bgr, img_gray),
            ),
            patch("visagism.face_detector.FaceDetector") as mock_face_cls,
            patch(
                "visagism.model_finder.ModelFinder.find",
                return_value=Path("/fake/model.dat"),
            ),
            patch(
                "visagism.landmark_detector.LandmarkDetector"
            ) as mock_landmark_cls,
            patch(
                "visagism.landmark_detector.LandmarkDetector.check_pose",
                return_value=True,
            ),
            patch(
                "visagism.hairline_detector.HairlineDetector"
            ) as mock_hairline_cls,
            patch("visagism.landmark_visualizer.LandmarkVisualizer"),
        ):

            mock_face = MagicMock()
            mock_face.detect.return_value = (100, 100, 150, 200)
            mock_face_cls.return_value = mock_face

            mock_landmark = MagicMock()
            mock_landmark.detect.return_value = mock_landmarks
            mock_landmark_cls.return_value = mock_landmark

            mock_hairline = MagicMock()
            mock_hairline.detect.return_value = {"hairline_y": 80, "method": "edge"}
            mock_hairline_cls.return_value = mock_hairline

            yield

    def test_calculator_called_and_console_output_contains_results(
        self, mock_config: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Calculator runs and console output contains analysis sections."""
        cli_mod = _import_visagism_py()

        cli_mod.main()

        captured = capsys.readouterr()
        assert "=== FACIAL MEASUREMENTS ===" in captured.out
        assert "=== PROPORTION ANALYSIS ===" in captured.out
        assert "Best Reference:" in captured.out
        assert "Deviations from Best Reference:" in captured.out

    def test_report_file_is_created(
        self, mock_config: MagicMock
    ) -> None:
        """A report file is saved to the output directory."""
        cli_mod = _import_visagism_py()

        cli_mod.main()

        report_files = list(mock_config.output_dir.glob("analysis_report_*.txt"))
        assert len(report_files) == 1
        content = report_files[0].read_text(encoding="utf-8")
        assert "FACIAL VISAGISM ANALYSIS REPORT" in content

    def test_fallback_warning_when_hairline_absent(
        self,
        mock_config: MagicMock,
        mock_landmarks_no_hairline: FacialLandmarks,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Warning is printed when hairline is not detected."""
        with (
            patch(
                "visagism.landmark_detector.LandmarkDetector"
            ) as mock_landmark_cls,
            patch(
                "visagism.hairline_detector.HairlineDetector"
            ) as mock_hairline_cls,
        ):

            mock_landmark = MagicMock()
            mock_landmark.detect.return_value = mock_landmarks_no_hairline
            mock_landmark_cls.return_value = mock_landmark

            mock_hairline = MagicMock()
            mock_hairline.detect.return_value = {
                "hairline_y": None, "method": "fallback",
            }
            mock_hairline_cls.return_value = mock_hairline

            cli_mod = _import_visagism_py()
            cli_mod.main()

        captured = capsys.readouterr()
        assert "Hairline not detected" in captured.out
        assert "may reduce accuracy" in captured.out

    def test_analysis_error_caught_gracefully(
        self,
        mock_config: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AnalysisError is caught and a user-friendly message is printed."""
        with patch(
            "visagism.visagism_calculator.VisagismCalculator.from_landmarks"
        ) as mock_from:
            mock_from.side_effect = AnalysisError("Invalid landmarks")

            cli_mod = _import_visagism_py()

            with pytest.raises(SystemExit) as exc_info:
                cli_mod.main()

            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error: Invalid landmarks" in captured.err
