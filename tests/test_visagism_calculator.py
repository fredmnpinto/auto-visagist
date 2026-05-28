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
        assert block.ideal_length_from_width == round(100.0 * GOLDEN_RATIO, 2)

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
        assert block.ideal_length_from_width is None

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
        assert block.ideal_length_from_width is None


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
        # Only block_1 has ideal_length_from_width
        assert consensus.ideal_length_from_width == block_1.ideal_length_from_width

    def test_consensus_without_length_from_width(self) -> None:
        """Consensus length_from_width is None when no block provides it."""
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
        block_2 = calc._compute_block_2()
        block_3 = calc._compute_block_3()

        consensus = calc._compute_consensus([block_2, block_3])
        assert consensus.ideal_length_from_width is None


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

    def test_calculate_collects_flagged_deviations(self) -> None:
        """All flagged deviations from blocks and consensus are collected."""
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

        # With such small values, many deviations should be flagged
        assert len(result.all_flagged_deviations) > 0

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
        """from_landmarks sets upper_third to None when hairline is absent."""
        landmarks = self._make_landmarks(hairline_y=None)
        calc = VisagismCalculator.from_landmarks(landmarks)

        assert calc._measurements.upper_third is None

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

    def test_visagism_analysis_default_flagged(self) -> None:
        """VisagismAnalysis defaults all_flagged_deviations to empty list."""
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
        # all_flagged_deviations is populated by calculate(), not defaulted
        assert isinstance(result.all_flagged_deviations, list)
