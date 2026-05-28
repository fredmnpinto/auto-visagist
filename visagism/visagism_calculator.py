"""Visagism calculator module for facial proportion analysis.

This module implements the visagism analysis methodology based on the
golden ratio (1.618). It computes three reference blocks from facial
measurements (eye width, inter-ocular distance, nose width), derives
consensus ideal values, and flags deviations exceeding the threshold.

Formulas adapted from the PlanilhaAnalise.xlsx spreadsheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from visagism.constants import DEVIATION_THRESHOLD, GOLDEN_RATIO
from visagism.errors import AnalysisError
from visagism.types import FacialLandmarks


@dataclass
class FacialMeasurements:
    """Container for raw facial measurements extracted from landmarks.

    Parameters
    ----------
    eye_width : float
        Average width of the eyes in pixels.
    inter_ocular_distance : float
        Distance between the inner eye corners in pixels.
    nose_width : float
        Width of the nose at the nostrils in pixels.
    mouth_width : float
        Width of the mouth (corner to corner) in pixels.
    face_width : float
        Width of the face (jawline left to right) in pixels.
    lower_third : float
        Vertical distance from nose base to chin in pixels.
    middle_third : float
        Vertical distance from eyebrows to nose base in pixels.
    upper_third : float or None
        Vertical distance from hairline to eyebrows in pixels.
        None when hairline is not available.
    """

    eye_width: float
    inter_ocular_distance: float
    nose_width: float
    mouth_width: float
    face_width: float
    lower_third: float
    middle_third: float
    upper_third: Optional[float] = None
    hairline_fallback_used: bool = False

    @property
    def total_face_height(self) -> Optional[float]:
        """Sum of lower, middle, and upper thirds.

        Returns
        -------
        float or None
            Total face height in pixels, or None if ``upper_third`` is None.
        """
        if self.upper_third is None:
            return None
        return round(self.lower_third + self.middle_third + self.upper_third, 2)


@dataclass
class DeviationResult:
    """Result of comparing an actual measurement against an ideal value.

    Parameters
    ----------
    measurement_name : str
        Human-readable name of the measurement being compared.
    actual : float or None
        Actual measured value in pixels, or None if unavailable.
    ideal : float
        Ideal value computed from the reference block or consensus.
    deviation_percent : float or None
        Percentage deviation from ideal, rounded to 2 decimal places.
        None when ``actual`` is None.
    is_flagged : bool
        True when the absolute deviation exceeds ``DEVIATION_THRESHOLD``
        (10 %).
    """

    measurement_name: str
    actual: Optional[float]
    ideal: float
    deviation_percent: Optional[float]
    is_flagged: bool


@dataclass
class ReferenceBlock:
    """A single reference block derived from one facial measurement.

    Parameters
    ----------
    block_name : str
        Name of the reference block (e.g. "Eye Width Reference").
    reference_measurement : str
        Name of the measurement used as the reference.
    reference_value : float
        Value of the reference measurement in pixels.
    ideal_face_width : float
        Ideal face width computed from the reference.
    ideal_face_height : float
        Ideal face height computed from the reference.
    ideal_mouth_width : float
        Ideal mouth width computed from the reference.
    ideal_length_from_width : float or None
        Ideal face length derived from the actual face width using the
        golden ratio. All three blocks populate this field.
    deviations : list of DeviationResult
        Deviations of actual measurements from the ideal values.
    """

    block_name: str
    reference_measurement: str
    reference_value: float
    ideal_face_width: float
    ideal_face_height: float
    ideal_mouth_width: float
    ideal_length_from_width: Optional[float] = None
    deviations: List[DeviationResult] = field(default_factory=list)


@dataclass
class ConsensusResult:
    """Consensus ideal values averaged across all reference blocks.

    Parameters
    ----------
    ideal_face_width : float
        Average ideal face width across blocks.
    ideal_face_height : float
        Average ideal face height across blocks.
    ideal_mouth_width : float
        Average ideal mouth width across blocks.
    ideal_length_from_width : float or None
        Average ideal length from width across blocks (None if all
        blocks lack this value).
    deviations : list of DeviationResult
        Deviations of actual measurements from the consensus ideals.
    """

    ideal_face_width: float
    ideal_face_height: float
    ideal_mouth_width: float
    ideal_length_from_width: Optional[float] = None
    deviations: List[DeviationResult] = field(default_factory=list)


@dataclass
class VisagismAnalysis:
    """Complete visagism analysis result.

    Parameters
    ----------
    measurements : FacialMeasurements
        Raw measurements extracted from the face.
    block_1_eye_width : ReferenceBlock
        Reference block derived from eye width.
    block_2_inter_ocular : ReferenceBlock
        Reference block derived from inter-ocular distance.
    block_3_nose_width : ReferenceBlock
        Reference block derived from nose width.
    consensus : ConsensusResult
        Consensus ideal values and deviations (deprecated, kept for
        backward compatibility).
    all_flagged_deviations : list of DeviationResult
        Flattened list of all deviations flagged across the best
        reference block.
    best_block : ReferenceBlock
        The reference block with the minimum total absolute deviation.
    best_block_name : str
        Human-readable name of the best reference block.
    """

    measurements: FacialMeasurements
    block_1_eye_width: ReferenceBlock
    block_2_inter_ocular: ReferenceBlock
    block_3_nose_width: ReferenceBlock
    consensus: ConsensusResult
    all_flagged_deviations: List[DeviationResult] = field(default_factory=list)
    best_block: ReferenceBlock = field(default=None)  # type: ignore[assignment]
    best_block_name: str = ""


class VisagismCalculator:
    """Calculate visagism analysis from facial measurements.

    The calculator takes eight facial measurements, validates them, and
    computes three reference blocks based on the golden ratio. It then
    derives consensus ideal values and flags any deviations that exceed
    the configured threshold.

    Parameters
    ----------
    eye_width : float
        Average eye width in pixels. Must be positive.
    inter_ocular_distance : float
        Inter-ocular distance in pixels. Must be positive.
    nose_width : float
        Nose width in pixels. Must be positive.
    mouth_width : float
        Mouth width in pixels. Must be positive.
    face_width : float
        Face width in pixels. Must be positive.
    lower_third : float
        Lower facial third in pixels. Must be positive.
    middle_third : float
        Middle facial third in pixels. Must be positive.
    upper_third : float or None
        Upper facial third in pixels. Must be positive if provided.
    hairline_fallback_used : bool
        Whether the upper third was estimated from the middle third
        because no hairline was detected.

    Raises
    ------
    AnalysisError
        If any required measurement is zero, negative, or None.
    """

    def __init__(
        self,
        eye_width: float,
        inter_ocular_distance: float,
        nose_width: float,
        mouth_width: float,
        face_width: float,
        lower_third: float,
        middle_third: float,
        upper_third: Optional[float] = None,
        hairline_fallback_used: bool = False,
    ) -> None:
        self._measurements = FacialMeasurements(
            eye_width=eye_width,
            inter_ocular_distance=inter_ocular_distance,
            nose_width=nose_width,
            mouth_width=mouth_width,
            face_width=face_width,
            lower_third=lower_third,
            middle_third=middle_third,
            upper_third=upper_third,
            hairline_fallback_used=hairline_fallback_used,
        )
        self._validate_measurements()

    def _validate_measurements(self) -> None:
        """Ensure all required measurements are strictly positive.

        Raises
        ------
        AnalysisError
            If any required measurement is missing, zero, or negative.
        """
        required = {
            "eye_width": self._measurements.eye_width,
            "inter_ocular_distance": self._measurements.inter_ocular_distance,
            "nose_width": self._measurements.nose_width,
            "mouth_width": self._measurements.mouth_width,
            "face_width": self._measurements.face_width,
            "lower_third": self._measurements.lower_third,
            "middle_third": self._measurements.middle_third,
        }

        for name, value in required.items():
            if value is None or value <= 0:
                raise AnalysisError(
                    f"Measurement '{name}' must be a positive number. "
                    f"Got {value!r}."
                )

        if self._measurements.upper_third is not None:
            if self._measurements.upper_third <= 0:
                raise AnalysisError(
                    "Measurement 'upper_third' must be a positive number. "
                    f"Got {self._measurements.upper_third!r}."
                )

    def calculate(self) -> VisagismAnalysis:
        """Orchestrate the full visagism analysis.

        Computes the three reference blocks, identifies the block with
        the smallest total absolute deviation (the "best" block), and
        collects flagged deviations from that block only.

        Returns
        -------
        VisagismAnalysis
            Complete analysis result containing measurements, reference
            blocks, consensus, best block, and flagged deviations.
        """
        block_1 = self._compute_block_1()
        block_2 = self._compute_block_2()
        block_3 = self._compute_block_3()

        blocks = [block_1, block_2, block_3]
        consensus = self._compute_consensus(blocks)

        best_block = self._select_best_block(blocks)

        all_flagged: List[DeviationResult] = [
            dev for dev in best_block.deviations if dev.is_flagged
        ]

        return VisagismAnalysis(
            measurements=self._measurements,
            block_1_eye_width=block_1,
            block_2_inter_ocular=block_2,
            block_3_nose_width=block_3,
            consensus=consensus,
            all_flagged_deviations=all_flagged,
            best_block=best_block,
            best_block_name=best_block.block_name,
        )

    def _compute_block_1(self) -> ReferenceBlock:
        """Compute Block 1 — Eye Width Reference.

        Formulas
        --------
        - ideal_face_width = eye_width * 4
        - ideal_face_height = ideal_face_width * GOLDEN_RATIO
        - ideal_mouth_width = eye_width * 1.5
        - ideal_length_from_width = face_width * GOLDEN_RATIO

        Returns
        -------
        ReferenceBlock
            Populated reference block with deviations.
        """
        eye_width = self._measurements.eye_width
        face_width = self._measurements.face_width

        ideal_face_width = round(eye_width * 4, 2)
        ideal_face_height = round(ideal_face_width * GOLDEN_RATIO, 2)
        ideal_mouth_width = round(eye_width * 1.5, 2)
        ideal_length_from_width = round(face_width * GOLDEN_RATIO, 2)

        block = ReferenceBlock(
            block_name="Eye Width Reference",
            reference_measurement="eye_width",
            reference_value=round(eye_width, 2),
            ideal_face_width=ideal_face_width,
            ideal_face_height=ideal_face_height,
            ideal_mouth_width=ideal_mouth_width,
            ideal_length_from_width=ideal_length_from_width,
        )
        block.deviations = self._compute_block_deviations(block)
        return block

    def _compute_block_2(self) -> ReferenceBlock:
        """Compute Block 2 — Inter-Ocular Distance Reference.

        Formulas
        --------
        - ideal_face_width = inter_ocular_distance * 4
        - ideal_face_height = ideal_face_width * GOLDEN_RATIO
        - ideal_mouth_width = inter_ocular_distance * 1.5
        - ideal_length_from_width = face_width * GOLDEN_RATIO

        Returns
        -------
        ReferenceBlock
            Populated reference block with deviations.
        """
        inter_ocular = self._measurements.inter_ocular_distance
        face_width = self._measurements.face_width

        ideal_face_width = round(inter_ocular * 4, 2)
        ideal_face_height = round(ideal_face_width * GOLDEN_RATIO, 2)
        ideal_mouth_width = round(inter_ocular * 1.5, 2)
        ideal_length_from_width = round(face_width * GOLDEN_RATIO, 2)

        block = ReferenceBlock(
            block_name="Inter-Ocular Distance Reference",
            reference_measurement="inter_ocular_distance",
            reference_value=round(inter_ocular, 2),
            ideal_face_width=ideal_face_width,
            ideal_face_height=ideal_face_height,
            ideal_mouth_width=ideal_mouth_width,
            ideal_length_from_width=ideal_length_from_width,
        )
        block.deviations = self._compute_block_deviations(block)
        return block

    def _compute_block_3(self) -> ReferenceBlock:
        """Compute Block 3 — Nose Width Reference.

        Formulas
        --------
        - ideal_face_width = nose_width * 4
        - ideal_face_height = ideal_face_width * GOLDEN_RATIO
        - ideal_mouth_width = nose_width * 1.5
        - ideal_length_from_width = face_width * GOLDEN_RATIO

        Returns
        -------
        ReferenceBlock
            Populated reference block with deviations.
        """
        nose_width = self._measurements.nose_width
        face_width = self._measurements.face_width

        ideal_face_width = round(nose_width * 4, 2)
        ideal_face_height = round(ideal_face_width * GOLDEN_RATIO, 2)
        ideal_mouth_width = round(nose_width * 1.5, 2)
        ideal_length_from_width = round(face_width * GOLDEN_RATIO, 2)

        block = ReferenceBlock(
            block_name="Nose Width Reference",
            reference_measurement="nose_width",
            reference_value=round(nose_width, 2),
            ideal_face_width=ideal_face_width,
            ideal_face_height=ideal_face_height,
            ideal_mouth_width=ideal_mouth_width,
            ideal_length_from_width=ideal_length_from_width,
        )
        block.deviations = self._compute_block_deviations(block)
        return block

    def _compute_block_deviations(
        self, block: ReferenceBlock
    ) -> List[DeviationResult]:
        """Compute deviations for a reference block.

        Compares actual measurements against the block's ideal values.

        Parameters
        ----------
        block : ReferenceBlock
            The reference block containing ideal values.

        Returns
        -------
        list of DeviationResult
            Deviations for face width, face height, mouth width, and
            length from width (when available).
        """
        deviations: List[DeviationResult] = []

        deviations.append(
            self._compute_deviation(
                "face_width",
                self._measurements.face_width,
                block.ideal_face_width,
            )
        )

        # All blocks now compute ideal_length_from_width based on actual face width
        assert block.ideal_length_from_width is not None
        deviations.append(
            self._compute_deviation(
                "face_height",
                self._measurements.total_face_height,
                block.ideal_length_from_width,
            )
        )

        deviations.append(
            self._compute_deviation(
                "mouth_width",
                self._measurements.mouth_width,
                block.ideal_mouth_width,
            )
        )

        if block.ideal_length_from_width is not None:
            deviations.append(
                self._compute_deviation(
                    "length_from_width",
                    self._measurements.total_face_height,
                    block.ideal_length_from_width,
                )
            )

        return deviations

    @staticmethod
    def _compute_deviation(
        measurement_name: str,
        actual: Optional[float],
        ideal: float,
    ) -> DeviationResult:
        """Compute the deviation of an actual value from an ideal value.

        Parameters
        ----------
        measurement_name : str
            Name of the measurement.
        actual : float or None
            Actual measured value, or None if unavailable.
        ideal : float
            Ideal reference value.

        Returns
        -------
        DeviationResult
            Populated deviation result with percentage and flag status.
        """
        if actual is None:
            return DeviationResult(
                measurement_name=measurement_name,
                actual=None,
                ideal=round(ideal, 2),
                deviation_percent=None,
                is_flagged=False,
            )

        deviation_percent = round(((actual - ideal) / ideal) * 100, 2)
        is_flagged = abs(deviation_percent) > (DEVIATION_THRESHOLD * 100)

        return DeviationResult(
            measurement_name=measurement_name,
            actual=round(actual, 2),
            ideal=round(ideal, 2),
            deviation_percent=deviation_percent,
            is_flagged=is_flagged,
        )

    def _compute_consensus(self, blocks: List[ReferenceBlock]) -> ConsensusResult:
        """Average ideal values across all reference blocks.

        Parameters
        ----------
        blocks : list of ReferenceBlock
            The three computed reference blocks.

        Returns
        -------
        ConsensusResult
            Consensus ideals and deviations from actual measurements.
        """
        ideal_face_width = round(
            sum(b.ideal_face_width for b in blocks) / len(blocks), 2
        )
        ideal_face_height = round(
            sum(b.ideal_face_height for b in blocks) / len(blocks), 2
        )
        ideal_mouth_width = round(
            sum(b.ideal_mouth_width for b in blocks) / len(blocks), 2
        )

        length_values = [
            b.ideal_length_from_width
            for b in blocks
            if b.ideal_length_from_width is not None
        ]
        ideal_length_from_width: Optional[float] = None
        if length_values:
            ideal_length_from_width = round(sum(length_values) / len(length_values), 2)

        consensus = ConsensusResult(
            ideal_face_width=ideal_face_width,
            ideal_face_height=ideal_face_height,
            ideal_mouth_width=ideal_mouth_width,
            ideal_length_from_width=ideal_length_from_width,
        )

        consensus.deviations = self._compute_consensus_deviations(consensus)
        return consensus

    def _compute_consensus_deviations(
        self, consensus: ConsensusResult
    ) -> List[DeviationResult]:
        """Compute deviations for the consensus result.

        Parameters
        ----------
        consensus : ConsensusResult
            The consensus containing averaged ideal values.

        Returns
        -------
        list of DeviationResult
            Deviations for face width, face height, mouth width, and
            length from width (when available).
        """
        deviations: List[DeviationResult] = []

        deviations.append(
            self._compute_deviation(
                "face_width",
                self._measurements.face_width,
                consensus.ideal_face_width,
            )
        )

        deviations.append(
            self._compute_deviation(
                "face_height",
                self._measurements.total_face_height,
                consensus.ideal_face_height,
            )
        )

        deviations.append(
            self._compute_deviation(
                "mouth_width",
                self._measurements.mouth_width,
                consensus.ideal_mouth_width,
            )
        )

        if consensus.ideal_length_from_width is not None:
            deviations.append(
                self._compute_deviation(
                    "length_from_width",
                    self._measurements.total_face_height,
                    consensus.ideal_length_from_width,
                )
            )

        return deviations

    @staticmethod
    def _select_best_block(blocks: List[ReferenceBlock]) -> ReferenceBlock:
        """Select the reference block with the smallest total deviation.

        The total deviation is the sum of absolute deviation percentages
        for all deviations that have a computed percentage (i.e. where
        the actual measurement was available).

        Parameters
        ----------
        blocks : list of ReferenceBlock
            The three computed reference blocks.

        Returns
        -------
        ReferenceBlock
            The block with the minimum total absolute deviation.
        """
        best = blocks[0]
        best_score = float("inf")

        for block in blocks:
            total = sum(
                abs(dev.deviation_percent)
                for dev in block.deviations
                if dev.deviation_percent is not None
            )
            if total < best_score:
                best_score = total
                best = block

        return best

    @classmethod
    def from_landmarks(cls, landmarks: FacialLandmarks) -> VisagismCalculator:
        """Create a calculator from 68-point facial landmarks.

        Extracts the required measurements from the landmark coordinates
        using the dlib 68-point model indexing.

        Parameters
        ----------
        landmarks : FacialLandmarks
            Detected landmarks including the 68-point list and optional
            ``hairline_y``.

        Returns
        -------
        VisagismCalculator
            Initialised calculator with extracted measurements.

        Raises
        ------
        AnalysisError
            If the landmark list does not contain 68 points.
        """
        if len(landmarks.landmarks_68) != 68:
            raise AnalysisError(
                f"Expected 68 landmarks, got {len(landmarks.landmarks_68)}."
            )

        pts = landmarks.landmarks_68

        # Eye widths — average of left and right eyes
        left_eye_width = math.dist(pts[36], pts[39])
        right_eye_width = math.dist(pts[42], pts[45])
        eye_width = (left_eye_width + right_eye_width) / 2.0

        # Inter-ocular distance — inner corners of the eyes
        inter_ocular_distance = math.dist(pts[39], pts[42])

        # Nose width — nostril edges
        nose_width = math.dist(pts[31], pts[35])

        # Mouth width — outer mouth corners
        mouth_width = math.dist(pts[48], pts[54])

        # Face width — leftmost and rightmost jaw points
        face_width = math.dist(pts[0], pts[16])

        # Lower third — nose base (point 33) to chin (point 8)
        lower_third = abs(pts[33][1] - pts[8][1])

        # Middle third — eyebrow line to nose base
        eyebrow_points = pts[17:27]
        avg_eyebrow_y = sum(p[1] for p in eyebrow_points) / len(eyebrow_points)
        middle_third = abs(avg_eyebrow_y - pts[33][1])

        # Upper third — hairline to eyebrow line (if hairline available)
        upper_third: Optional[float] = None
        hairline_fallback_used = False
        if landmarks.hairline_y is not None:
            upper_third = abs(avg_eyebrow_y - landmarks.hairline_y)
        else:
            upper_third = middle_third
            hairline_fallback_used = True

        return cls(
            eye_width=eye_width,
            inter_ocular_distance=inter_ocular_distance,
            nose_width=nose_width,
            mouth_width=mouth_width,
            face_width=face_width,
            lower_third=lower_third,
            middle_third=middle_third,
            upper_third=upper_third,
            hairline_fallback_used=hairline_fallback_used,
        )
