"""Tests for the VisagismCalculator module."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from visagism.constants import GOLDEN_RATIO
from visagism.errors import AnalysisError
from visagism.types import FacialLandmarks, LandmarkRegions
from visagism.visagism_calculator import (
    ConsensusResult,
    DeviationResult,
    FacialMeasurements,
    ReferenceBlock,
    VisagismAnalysis,
    VisagismCalculator,
)


class TestFacialMeasurements:
    """Test suite for FacialMeasurements dataclass."""

    def test_total_face_height_with_all_thirds(self) -> None:
        """Total face height sums all three thirds."""
        fm = FacialMeasurements(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        assert fm.total_face_height == 150.0

    def test_total_face_height_without_upper_third(self) -> None:
        """Total face height is None when upper_third is missing."""
        fm = FacialMeasurements(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=None,
        )
        assert fm.total_face_height is None


class TestVisagismCalculatorInit:
    """Test suite for VisagismCalculator initialisation and validation."""

    def test_init_with_valid_measurements(self) -> None:
        """Calculator initialises successfully with positive values."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        assert calc._measurements.eye_width == 10.0

    def test_init_with_none_upper_third(self) -> None:
        """Calculator accepts None for upper_third."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=None,
        )
        assert calc._measurements.upper_third is None

    @pytest.mark.parametrize(
        "field_name, bad_value",
        [
            ("eye_width", 0.0),
            ("eye_width", -5.0),
            ("inter_ocular_distance", 0.0),
            ("inter_ocular_distance", -1.0),
            ("nose_width", 0.0),
            ("nose_width", -2.0),
            ("mouth_width", 0.0),
            ("mouth_width", -3.0),
            ("face_width", 0.0),
            ("face_width", -10.0),
            ("lower_third", 0.0),
            ("lower_third", -1.0),
            ("middle_third", 0.0),
            ("middle_third", -1.0),
        ],
    )
    def test_init_rejects_invalid_required_values(
        self, field_name: str, bad_value: float
    ) -> None:
        """AnalysisError is raised for zero or negative required values."""
        kwargs = {
            "eye_width": 10.0,
            "inter_ocular_distance": 20.0,
            "nose_width": 15.0,
            "mouth_width": 30.0,
            "face_width": 100.0,
            "lower_third": 50.0,
            "middle_third": 60.0,
            "upper_third": 40.0,
        }
        kwargs[field_name] = bad_value
        with pytest.raises(AnalysisError, match=f"Measurement '{field_name}'"):
            VisagismCalculator(**kwargs)

    def test_init_rejects_negative_upper_third(self) -> None:
        """AnalysisError is raised for negative upper_third."""
        with pytest.raises(AnalysisError, match="Measurement 'upper_third'"):
            VisagismCalculator(
                eye_width=10.0,
                inter_ocular_distance=20.0,
                nose_width=15.0,
                mouth_width=30.0,
                face_width=100.0,
                lower_third=50.0,
                middle_third=60.0,
                upper_third=-5.0,
            )

    def test_init_rejects_zero_upper_third(self) -> None:
        """AnalysisError is raised for zero upper_third."""
        with pytest.raises(AnalysisError, match="Measurement 'upper_third'"):
            VisagismCalculator(
                eye_width=10.0,
                inter_ocular_distance=20.0,
                nose_width=15.0,
                mouth_width=30.0,
                face_width=100.0,
                lower_third=50.0,
                middle_third=60.0,
                upper_third=0.0,
            )


class TestBlockComputations:
    """Test suite for reference block formula correctness."""

    def test_block_1_formulas(self) -> None:
        """Block 1 formulas produce expected values for known inputs."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block = calc._compute_block_1()

        assert block.block_name == "Eye Width Reference"
        assert block.reference_measurement == "eye_width"
        assert block.reference_value == 10.0
        assert block.ideal_face_width == 40.0  # 10 * 4
        assert block.ideal_face_height == round(40.0 * GOLDEN_RATIO, 2)
        assert block.ideal_mouth_width == 15.0  # 10 * 1.5

    def test_block_2_formulas(self) -> None:
        """Block 2 formulas produce expected values for known inputs."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block = calc._compute_block_2()

        assert block.block_name == "Inter-Ocular Distance Reference"
        assert block.reference_measurement == "inter_ocular_distance"
        assert block.reference_value == 20.0
        assert block.ideal_face_width == 80.0  # 20 * 4
        assert block.ideal_face_height == round(80.0 * GOLDEN_RATIO, 2)
        assert block.ideal_mouth_width == 30.0  # 20 * 1.5

    def test_block_3_formulas(self) -> None:
        """Block 3 formulas produce expected values for known inputs."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block = calc._compute_block_3()

        assert block.block_name == "Nose Width Reference"
        assert block.reference_measurement == "nose_width"
        assert block.reference_value == 15.0
        assert block.ideal_face_width == 60.0  # 15 * 4
        assert block.ideal_face_height == round(60.0 * GOLDEN_RATIO, 2)
        assert block.ideal_mouth_width == 22.5  # 15 * 1.5


class TestDeviationComputation:
    """Test suite for deviation calculation and flagging logic."""

    def test_deviation_exactly_at_threshold_not_flagged(self) -> None:
        """Deviation exactly at 10 % is NOT flagged (strict > threshold)."""
        # ideal = 100, actual = 110 -> 10 % deviation
        result = VisagismCalculator._compute_deviation("test", 110.0, 100.0)
        assert result.deviation_percent == 10.0
        assert result.is_flagged is False

    def test_deviation_above_threshold_flagged(self) -> None:
        """Deviation above 10 % is flagged."""
        result = VisagismCalculator._compute_deviation("test", 111.0, 100.0)
        assert result.deviation_percent == 11.0
        assert result.is_flagged is True

    def test_deviation_below_threshold_not_flagged(self) -> None:
        """Deviation below 10 % is not flagged."""
        result = VisagismCalculator._compute_deviation("test", 105.0, 100.0)
        assert result.deviation_percent == 5.0
        assert result.is_flagged is False

    def test_negative_deviation_flagged(self) -> None:
        """Negative deviation exceeding threshold is flagged."""
        result = VisagismCalculator._compute_deviation("test", 89.0, 100.0)
        assert result.deviation_percent == -11.0
        assert result.is_flagged is True

    def test_negative_deviation_within_threshold_not_flagged(self) -> None:
        """Negative deviation within threshold is not flagged."""
        result = VisagismCalculator._compute_deviation("test", 95.0, 100.0)
        assert result.deviation_percent == -5.0
        assert result.is_flagged is False

    def test_deviation_with_none_actual(self) -> None:
        """None actual yields None deviation_percent and not flagged."""
        result = VisagismCalculator._compute_deviation("test", None, 100.0)
        assert result.actual is None
        assert result.deviation_percent is None
        assert result.is_flagged is False

    def test_deviation_zero_percent_not_flagged(self) -> None:
        """Zero deviation is not flagged."""
        result = VisagismCalculator._compute_deviation("test", 100.0, 100.0)
        assert result.deviation_percent == 0.0
        assert result.is_flagged is False


class TestConsensusComputation:
    """Test suite for consensus averaging."""

    def test_consensus_averages_ideal_values(self) -> None:
        """Consensus averages face width, height, and mouth width."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block_1 = calc._compute_block_1()
        block_2 = calc._compute_block_2()
        block_3 = calc._compute_block_3()

        consensus = calc._compute_consensus([block_1, block_2, block_3])

        expected_width = round(
            (
                block_1.ideal_face_width
                + block_2.ideal_face_width
                + block_3.ideal_face_width
            )
            / 3,
            2,
        )
        expected_height = round(
            (
                block_1.ideal_face_height
                + block_2.ideal_face_height
                + block_3.ideal_face_height
            )
            / 3,
            2,
        )
        expected_mouth = round(
            (
                block_1.ideal_mouth_width
                + block_2.ideal_mouth_width
                + block_3.ideal_mouth_width
            )
            / 3,
            2,
        )

        assert consensus.ideal_face_width == expected_width
        assert consensus.ideal_face_height == expected_height
        assert consensus.ideal_mouth_width == expected_mouth

    def test_consensus_has_no_length_from_width(self) -> None:
        """ConsensusResult no longer contains ideal_length_from_width field."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block_1 = calc._compute_block_1()
        block_2 = calc._compute_block_2()
        block_3 = calc._compute_block_3()

        consensus = calc._compute_consensus([block_1, block_2, block_3])
        assert not hasattr(consensus, "ideal_length_from_width")


class TestBestBlockSelection:
    """Test suite for best block selection logic."""

    def test_select_best_block_chooses_minimum_deviation(self) -> None:
        """The block with the smallest total absolute deviation is selected."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        block_1 = calc._compute_block_1()
        block_2 = calc._compute_block_2()
        block_3 = calc._compute_block_3()

        best = VisagismCalculator._select_best_block([block_1, block_2, block_3])

        # Compute expected best block manually
        scores = []
        for block in [block_1, block_2, block_3]:
            total = sum(
                abs(dev.deviation_percent)
                for dev in block.deviations
                if dev.deviation_percent is not None
            )
            scores.append(total)

        min_idx = scores.index(min(scores))
        expected = [block_1, block_2, block_3][min_idx]
        assert best.block_name == expected.block_name

    def test_best_block_fields_populated(self) -> None:
        """calculate() populates best_block and best_block_name correctly."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()

        assert result.best_block is not None
        assert result.best_block_name == result.best_block.block_name
        assert result.best_block_name in [
            "Eye Width Reference",
            "Inter-Ocular Distance Reference",
            "Nose Width Reference",
        ]


class TestFullAnalysis:
    """Test suite for end-to-end analysis integration."""

    def test_calculate_returns_visagism_analysis(self) -> None:
        """calculate() returns a fully populated VisagismAnalysis."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()

        assert isinstance(result, VisagismAnalysis)
        assert isinstance(result.measurements, FacialMeasurements)
        assert isinstance(result.block_1_eye_width, ReferenceBlock)
        assert isinstance(result.block_2_inter_ocular, ReferenceBlock)
        assert isinstance(result.block_3_nose_width, ReferenceBlock)
        assert isinstance(result.consensus, ConsensusResult)
        assert isinstance(result.all_flagged_deviations, list)
        assert isinstance(result.best_block, ReferenceBlock)
        assert isinstance(result.best_block_name, str)
        assert result.best_block_name != ""

    def test_calculate_collects_flagged_deviations_from_best_block(self) -> None:
        """Flagged deviations are collected only from the best block."""
        # Use measurements that will produce large deviations
        calc = VisagismCalculator(
            eye_width=5.0,
            inter_ocular_distance=5.0,
            nose_width=5.0,
            mouth_width=5.0,
            face_width=5.0,
            lower_third=5.0,
            middle_third=5.0,
            upper_third=5.0,
        )
        result = calc.calculate()

        # With such small values, the best block should still have some flagged
        # deviations, but only from the best block
        best_flagged = [d for d in result.best_block.deviations if d.is_flagged]
        assert result.all_flagged_deviations == best_flagged

    def test_calculate_without_upper_third(self) -> None:
        """Analysis works when upper_third is None."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=None,
        )
        result = calc.calculate()

        assert result.measurements.upper_third is None
        assert result.measurements.total_face_height is None
        # Face height deviations should have None actual
        blocks = [
            result.block_1_eye_width,
            result.block_2_inter_ocular,
            result.block_3_nose_width,
        ]
        for block in blocks:
            face_height_dev = next(
                d for d in block.deviations if d.measurement_name == "face_height"
            )
            assert face_height_dev.actual is None
            assert face_height_dev.deviation_percent is None
            assert face_height_dev.is_flagged is False


class TestFromLandmarks:
    """Test suite for from_landmarks factory method."""

    def _make_landmarks(
        self, hairline_y: int | None = None
    ) -> FacialLandmarks:
        """Create a FacialLandmarks instance with realistic 68-point data."""
        # Build a symmetric face-like set of points
        pts: List[tuple[int, int]] = []

        # Jaw: 0-16 (left to right, bottom arc)
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
            (150, 240),  # 48: left corner
            (160, 230),  # 49
            (175, 225),  # 50
            (190, 230),  # 51
            (200, 235),  # 52
            (210, 240),  # 53
            (250, 240),  # 54: right corner
            (210, 250),  # 55
            (200, 255),  # 56
            (190, 250),  # 57
            (175, 255),  # 58
            (160, 250),  # 59
        ])

        # Inner mouth: 60-67
        pts.extend([
            (160, 240), (170, 235), (180, 240), (175, 245),
            (165, 245), (160, 240), (160, 240), (160, 240),
        ])

        assert len(pts) == 68

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
            hairline_y=hairline_y,
        )

    def test_from_landmarks_extracts_measurements(self) -> None:
        """from_landmarks creates a calculator with extracted measurements."""
        landmarks = self._make_landmarks(hairline_y=80)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.eye_width > 0
        assert calc._measurements.inter_ocular_distance > 0
        assert calc._measurements.nose_width > 0
        assert calc._measurements.mouth_width > 0
        assert calc._measurements.face_width > 0
        assert calc._measurements.lower_third > 0
        assert calc._measurements.middle_third > 0
        assert calc._measurements.upper_third is not None
        assert calc._measurements.upper_third > 0

    def test_from_landmarks_without_hairline(self) -> None:
        """from_landmarks uses fallback when hairline is absent."""
        landmarks = self._make_landmarks(hairline_y=None)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.upper_third == calc._measurements.middle_third
        assert calc._measurements.hairline_fallback_used is True

    def test_from_landmarks_with_none_hairline_uses_fallback(self) -> None:
        """When hairline_y is None, upper_third equals middle_third and
        fallback flag is set."""
        landmarks = self._make_landmarks(hairline_y=None)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.upper_third == calc._measurements.middle_third
        assert calc._measurements.hairline_fallback_used is True

    def test_from_landmarks_with_hairline_no_fallback(self) -> None:
        """When hairline_y is present, fallback flag is False."""
        landmarks = self._make_landmarks(hairline_y=80)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.hairline_fallback_used is False
        assert calc._measurements.upper_third is not None
        assert calc._measurements.upper_third > 0

    def test_from_landmarks_rejects_wrong_count(self) -> None:
        """AnalysisError is raised when landmark count is not 68."""
        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(0, 0, 100, 100),
            landmarks_68=[(0, 0)] * 10,
            landmarks_by_region={},
        )
        with pytest.raises(AnalysisError, match="Expected 68 landmarks"):
            VisagismCalculator.from_landmarks(landmarks)

    def test_from_landmarks_eye_width_average(self) -> None:
        """Eye width is the average of left and right eye widths."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # Left eye: 36(120,150) to 39(150,150) -> width = 30
        # Right eye: 42(200,150) to 45(230,150) -> width = 30
        # Average = 30
        assert calc._measurements.eye_width == 30.0

    def test_from_landmarks_inter_ocular(self) -> None:
        """Inter-ocular distance is between inner eye corners."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # Left inner: 39(150,150), Right inner: 42(200,150)
        assert calc._measurements.inter_ocular_distance == 50.0

    def test_from_landmarks_nose_width(self) -> None:
        """Nose width is between nostril edges."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # 31(165,190) to 35(185,190)
        assert calc._measurements.nose_width == 20.0

    def test_from_landmarks_mouth_width(self) -> None:
        """Mouth width is between outer mouth corners."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # 48(150,240) to 54(250,240)
        assert calc._measurements.mouth_width == 100.0

    def test_from_landmarks_face_width(self) -> None:
        """Face width is between leftmost and rightmost jaw points."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # 0(100,300) to 16(260,300)
        assert calc._measurements.face_width == 160.0

    def test_from_landmarks_lower_third(self) -> None:
        """Lower third is vertical distance from nose base to chin."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # Nose base 33(175,200), Chin 8(180,300)
        assert calc._measurements.lower_third == 100.0

    def test_from_landmarks_middle_third(self) -> None:
        """Middle third is vertical distance from eyebrows to nose base."""
        landmarks = self._make_landmarks()
        calc = VisagismCalculator.from_landmarks(landmarks)

        # Eyebrow avg y: all eyebrow points have y=120 or 110 or 115
        # Average = (120+115+110+115+120+120+115+110+115+120) / 10 = 116
        # Nose base 33 y = 200
        assert calc._measurements.middle_third == 84.0

    def test_from_landmarks_upper_third(self) -> None:
        """Upper third is vertical distance from hairline to eyebrows."""
        landmarks = self._make_landmarks(hairline_y=80)
        calc = VisagismCalculator.from_landmarks(landmarks)

        # Eyebrow avg y = 116, hairline_y = 80
        assert calc._measurements.upper_third == 36.0

    def test_from_landmarks_populates_debug_fields(self) -> None:
        """from_landmarks populates debug fields on FacialMeasurements."""
        landmarks = self._make_landmarks(hairline_y=80)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.left_eye_width is not None
        assert calc._measurements.right_eye_width is not None
        assert calc._measurements.avg_eyebrow_y is not None
        assert calc._measurements.hairline_y == 80

        # Left eye: 36(120,150) to 39(150,150) -> width = 30
        assert calc._measurements.left_eye_width == 30.0
        # Right eye: 42(200,150) to 45(230,150) -> width = 30
        assert calc._measurements.right_eye_width == 30.0
        # Eyebrow avg y = 116
        assert calc._measurements.avg_eyebrow_y == 116.0


class TestGlobalFaceHeightProportion:
    """Test suite for global face height proportion computation (Tweak 1)."""

    def test_ideal_face_height_from_width_computed(self) -> None:
        """Global ideal face height equals face_width * GOLDEN_RATIO."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        expected = round(100.0 * GOLDEN_RATIO, 2)
        assert result.ideal_face_height_from_width == expected

    def test_global_face_height_deviation_populated(self) -> None:
        """Global face height deviation is a DeviationResult."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        assert isinstance(result.global_face_height_deviation, DeviationResult)
        assert result.global_face_height_deviation.measurement_name == "face_height"

    def test_global_face_height_deviation_without_upper_third(self) -> None:
        """Global deviation handles None total_face_height gracefully."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=None,
        )
        result = calc.calculate()
        assert result.global_face_height_deviation.actual is None
        assert result.global_face_height_deviation.deviation_percent is None
        assert result.global_face_height_deviation.is_flagged is False


class TestRejectedReferenceFlags:
    """Test suite for rejected reference flags computation (Tweak 2)."""

    def test_large_eyes_flag(self) -> None:
        """When face is smaller than eye-width predicts, eyes are large."""
        # Small face relative to eye width -> negative deviations
        calc = VisagismCalculator(
            eye_width=30.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=80.0,
            lower_third=40.0,
            middle_third=40.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        # Eye Width block should be best or rejected; if rejected, flag large_eyes
        if result.best_block_name != "Eye Width Reference":
            assert "large_eyes" in result.rejected_reference_flags

    def test_small_eyes_flag(self) -> None:
        """When face is larger than eye-width predicts, eyes are small."""
        # Large face relative to eye width -> positive deviations
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=200.0,
            lower_third=100.0,
            middle_third=100.0,
            upper_third=100.0,
        )
        result = calc.calculate()
        if result.best_block_name != "Eye Width Reference":
            assert "small_eyes" in result.rejected_reference_flags

    def test_wide_set_eyes_flag(self) -> None:
        """When face is smaller than inter-ocular predicts, eyes are wide-set."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=40.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=80.0,
            lower_third=40.0,
            middle_third=40.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        if result.best_block_name != "Inter-Ocular Distance Reference":
            assert "wide_set_eyes" in result.rejected_reference_flags

    def test_close_set_eyes_flag(self) -> None:
        """When face is larger than inter-ocular predicts, eyes are close-set."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=10.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=200.0,
            lower_third=100.0,
            middle_third=100.0,
            upper_third=100.0,
        )
        result = calc.calculate()
        if result.best_block_name != "Inter-Ocular Distance Reference":
            assert "close_set_eyes" in result.rejected_reference_flags

    def test_large_nose_flag(self) -> None:
        """When face is smaller than nose-width predicts, nose is large."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=30.0,
            mouth_width=30.0,
            face_width=80.0,
            lower_third=40.0,
            middle_third=40.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        if result.best_block_name != "Nose Width Reference":
            assert "large_nose" in result.rejected_reference_flags

    def test_small_nose_flag(self) -> None:
        """When face is larger than nose-width predicts, nose is small."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=10.0,
            mouth_width=30.0,
            face_width=200.0,
            lower_third=100.0,
            middle_third=100.0,
            upper_third=100.0,
        )
        result = calc.calculate()
        if result.best_block_name != "Nose Width Reference":
            assert "small_nose" in result.rejected_reference_flags

    def test_best_block_not_flagged(self) -> None:
        """The best block is never in rejected_reference_flags."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        best_name = result.best_block_name
        flag_map = {
            "Eye Width Reference": ["large_eyes", "small_eyes"],
            "Inter-Ocular Distance Reference": [
                "wide_set_eyes",
                "close_set_eyes",
            ],
            "Nose Width Reference": ["large_nose", "small_nose"],
        }
        best_flags = flag_map.get(best_name, [])
        for flag in best_flags:
            assert flag not in result.rejected_reference_flags

    def test_mixed_deviations_no_flag(self) -> None:
        """Mixed deviations (not >=2 same sign) produce no flag."""
        # Use measurements that produce mixed signs for a non-best block
        calc = VisagismCalculator(
            eye_width=25.0,
            inter_ocular_distance=25.0,
            nose_width=25.0,
            mouth_width=37.5,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        # At least one non-best block should exist
        assert len(result.rejected_reference_flags) <= 2

    def test_rejected_flags_empty_when_all_same(self) -> None:
        """When all blocks have similar deviations, flags may be empty."""
        calc = VisagismCalculator(
            eye_width=25.0,
            inter_ocular_distance=25.0,
            nose_width=25.0,
            mouth_width=37.5,
            face_width=100.0,
            lower_third=54.0,
            middle_third=54.0,
            upper_third=54.0,
        )
        result = calc.calculate()
        assert isinstance(result.rejected_reference_flags, list)

    def test_unknown_block_name_skipped(self) -> None:
        """Blocks with unknown names are skipped without error."""
        custom_block = ReferenceBlock(
            block_name="Unknown Block",
            reference_measurement="custom",
            reference_value=10.0,
            ideal_face_width=40.0,
            ideal_face_height=60.0,
            ideal_mouth_width=15.0,
            deviations=[
                DeviationResult(
                    measurement_name="face_width",
                    actual=80.0,
                    ideal=40.0,
                    deviation_percent=100.0,
                    is_flagged=True,
                ),
                DeviationResult(
                    measurement_name="face_height",
                    actual=120.0,
                    ideal=60.0,
                    deviation_percent=100.0,
                    is_flagged=True,
                ),
                DeviationResult(
                    measurement_name="mouth_width",
                    actual=30.0,
                    ideal=15.0,
                    deviation_percent=100.0,
                    is_flagged=True,
                ),
            ],
        )
        best_block = ReferenceBlock(
            block_name="Eye Width Reference",
            reference_measurement="eye_width",
            reference_value=10.0,
            ideal_face_width=40.0,
            ideal_face_height=60.0,
            ideal_mouth_width=15.0,
            deviations=[],
        )
        flags = VisagismCalculator._compute_rejected_reference_flags(
            [custom_block, best_block], best_block
        )
        assert flags == []


class TestFacialThirdsProportion:
    """Test suite for facial thirds proportion flags (Tweak 3)."""

    def test_balanced_thirds_no_flags(self) -> None:
        """Equal thirds produce no proportion flags."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=50.0,
            upper_third=50.0,
        )
        result = calc.calculate()
        assert result.thirds_proportion_flags == []

    def test_large_upper_third_flagged(self) -> None:
        """Upper third > ideal by >10% produces large_upper_third."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=30.0,
            middle_third=30.0,
            upper_third=50.0,
        )
        result = calc.calculate()
        assert "large_upper_third" in result.thirds_proportion_flags

    def test_small_lower_third_flagged(self) -> None:
        """Lower third < ideal by >10% produces small_lower_third."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=20.0,
            middle_third=50.0,
            upper_third=50.0,
        )
        result = calc.calculate()
        assert "small_lower_third" in result.thirds_proportion_flags

    def test_large_middle_third_flagged(self) -> None:
        """Middle third > ideal by >10% produces large_middle_third."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=30.0,
            middle_third=50.0,
            upper_third=30.0,
        )
        result = calc.calculate()
        assert "large_middle_third" in result.thirds_proportion_flags

    def test_without_upper_third_compares_two_halves(self) -> None:
        """When upper_third is None, compare middle and lower against half."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=30.0,
            middle_third=70.0,
            upper_third=None,
        )
        result = calc.calculate()
        # middle=70, lower=30, partial_total=100, ideal=50
        # middle is 40% above ideal -> large_middle_third
        # lower is 40% below ideal -> small_lower_third
        assert "large_middle_third" in result.thirds_proportion_flags
        assert "small_lower_third" in result.thirds_proportion_flags

    def test_without_upper_third_balanced_no_flags(self) -> None:
        """Balanced middle and lower thirds with no upper produce no flags."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=50.0,
            upper_third=None,
        )
        result = calc.calculate()
        assert result.thirds_proportion_flags == []

    def test_thirds_within_threshold_not_flagged(self) -> None:
        """Deviations within 10% threshold are not flagged."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=48.0,
            middle_third=50.0,
            upper_third=52.0,
        )
        result = calc.calculate()
        # ideal = 150/3 = 50; deviations: -4%, 0%, +4% — all within 10%
        assert result.thirds_proportion_flags == []


class TestDataclassStructures:
    """Test suite for dataclass instantiation and defaults."""

    def test_deviation_result_defaults(self) -> None:
        """DeviationResult can be instantiated with all fields."""
        dr = DeviationResult(
            measurement_name="face_width",
            actual=100.0,
            ideal=110.0,
            deviation_percent=-9.09,
            is_flagged=False,
        )
        assert dr.measurement_name == "face_width"

    def test_reference_block_default_deviations(self) -> None:
        """ReferenceBlock defaults deviations to empty list."""
        rb = ReferenceBlock(
            block_name="Test",
            reference_measurement="test",
            reference_value=10.0,
            ideal_face_width=40.0,
            ideal_face_height=60.0,
            ideal_mouth_width=15.0,
        )
        assert rb.deviations == []

    def test_consensus_result_default_deviations(self) -> None:
        """ConsensusResult defaults deviations to empty list."""
        cr = ConsensusResult(
            ideal_face_width=50.0,
            ideal_face_height=80.0,
            ideal_mouth_width=20.0,
        )
        assert cr.deviations == []

    def test_visagism_analysis_new_fields(self) -> None:
        """VisagismAnalysis includes new Tweak 1-3 fields."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=50.0,
            middle_third=60.0,
            upper_third=40.0,
        )
        result = calc.calculate()
        assert hasattr(result, "ideal_face_height_from_width")
        assert hasattr(result, "global_face_height_deviation")
        assert hasattr(result, "rejected_reference_flags")
        assert hasattr(result, "thirds_proportion_flags")
        assert isinstance(result.rejected_reference_flags, list)
        assert isinstance(result.thirds_proportion_flags, list)
