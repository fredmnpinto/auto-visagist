"""Report formatter for visagism analysis results.

This module provides human-readable formatting for ``VisagismAnalysis``
results, supporting both console output and persistent text reports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from visagism.constants import THIRDS_DEVIATION_THRESHOLD
from visagism.visagism_calculator import (
    DeviationResult,
    ReferenceBlock,
    VisagismAnalysis,
)

REJECTED_FLAG_DESCRIPTIONS = {
    "large_eyes": "eyes are relatively large",
    "small_eyes": "eyes are relatively small",
    "wide_set_eyes": "eyes are relatively wide-set",
    "close_set_eyes": "eyes are relatively close-set",
    "large_nose": "nose is relatively large",
    "small_nose": "nose is relatively small",
}

_MEASUREMENT_FORMULAS = {
    "eye_width": "avg(dist(36,39), dist(42,45))",
    "inter_ocular_distance": "dist(39,42)",
    "nose_width": "dist(31,35)",
    "mouth_width": "dist(48,54)",
    "face_width": "dist(0,16)",
    "lower_third": "abs(33y - 8y)",
    "middle_third": "abs(avg_eyebrow_y - 33y)",
    "upper_third": "abs(avg_eyebrow_y - hairline_y)",
}

_BLOCK_FORMULAS = {
    "Eye Width Reference": (
        "ideal_width = eye_width × 4, "
        "ideal_height = ideal_width × 1.618, "
        "ideal_mouth = eye_width × 1.5"
    ),
    "Inter-Ocular Distance Reference": (
        "ideal_width = inter_ocular × 4, "
        "ideal_height = ideal_width × 1.618, "
        "ideal_mouth = inter_ocular × 1.5"
    ),
    "Nose Width Reference": (
        "ideal_width = nose_width × 4, "
        "ideal_height = ideal_width × 1.618, "
        "ideal_mouth = nose_width × 1.5"
    ),
}

_BLOCK_DISPLAY_NAMES = {
    "Eye Width Reference": "Block 1 (Eye Width)",
    "Inter-Ocular Distance Reference": "Block 2 (Inter-Ocular)",
    "Nose Width Reference": "Block 3 (Nose Width)",
}

_BLOCK_REFERENCE_LABELS = {
    "Eye Width Reference": "eye_width",
    "Inter-Ocular Distance Reference": "inter_ocular",
    "Nose Width Reference": "nose_width",
}


class ReportFormatter:
    """Format ``VisagismAnalysis`` results for console and file output."""

    @staticmethod
    def format_console(
        analysis: VisagismAnalysis, debug: bool = False
    ) -> str:
        """Return human-readable console output.

        Parameters
        ----------
        analysis : VisagismAnalysis
            Complete visagism analysis result.
        debug : bool, optional
            When True, include diagnostic detail lines woven into the
            existing sections. By default False.

        Returns
        -------
        str
            Multi-line formatted string suitable for printing to stdout.
        """
        lines: List[str] = []

        # === FACIAL MEASUREMENTS ===
        lines.append("=== FACIAL MEASUREMENTS ===")
        m = analysis.measurements
        lines.append(f"  Eye Width: {m.eye_width:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['eye_width']}"
            )
            if m.left_eye_width is not None and m.right_eye_width is not None:
                lines.append(
                    f"  → Intermediate: left={m.left_eye_width:.2f} px, "
                    f"right={m.right_eye_width:.2f} px"
                )
        lines.append(
            f"  Inter-Ocular Distance: {m.inter_ocular_distance:.2f} px"
        )
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['inter_ocular_distance']}"
            )
        lines.append(f"  Nose Width: {m.nose_width:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['nose_width']}"
            )
        lines.append(f"  Mouth Width: {m.mouth_width:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['mouth_width']}"
            )
        lines.append(f"  Face Width: {m.face_width:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['face_width']}"
            )
        lines.append(f"  Lower Third: {m.lower_third:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['lower_third']}"
            )
        lines.append(f"  Middle Third: {m.middle_third:.2f} px")
        if debug:
            lines.append(
                f"  → Formula: {_MEASUREMENT_FORMULAS['middle_third']}"
            )
            if m.avg_eyebrow_y is not None:
                lines.append(
                    f"  → Intermediate: avg_eyebrow_y={m.avg_eyebrow_y:.2f} px "
                    f"(landmarks 17-26)"
                )
        if m.upper_third is not None:
            if m.hairline_fallback_used:
                lines.append(
                    f"  Upper Third: {m.upper_third:.2f} px [estimated]"
                )
            else:
                lines.append(f"  Upper Third: {m.upper_third:.2f} px")
            if debug:
                lines.append(
                    f"  → Formula: {_MEASUREMENT_FORMULAS['upper_third']}"
                )
                intermediates: List[str] = []
                if m.avg_eyebrow_y is not None:
                    intermediates.append(
                        f"avg_eyebrow_y={m.avg_eyebrow_y:.2f} px"
                    )
                if m.hairline_y is not None:
                    intermediates.append(
                        f"hairline_y={m.hairline_y:.2f} px"
                    )
                if intermediates:
                    lines.append(
                        f"  → Intermediate: {', '.join(intermediates)}"
                    )
        else:
            lines.append("  Upper Third: N/A")
        total = m.total_face_height
        if total is not None:
            lines.append(f"  Total Face Height: {total:.2f} px")
        else:
            lines.append("  Total Face Height: N/A")
        lines.append("")

        # === PROPORTION ANALYSIS ===
        lines.append("=== PROPORTION ANALYSIS ===")
        lines.append("")
        lines.append(f"Best Reference: {analysis.best_block_name}")

        if debug:
            best_score = _block_score(analysis.best_block)
            lines.append(
                f"  → Selection: lowest score {best_score:.2f} "
                f"(sum of abs deviations)"
            )
            lines.append("  →")
            lines.append("  → All Reference Blocks:")
            for block in [
                analysis.block_1_eye_width,
                analysis.block_2_inter_ocular,
                analysis.block_3_nose_width,
            ]:
                score = _block_score(block)
                display_name = _BLOCK_DISPLAY_NAMES.get(
                    block.block_name, block.block_name
                )
                ref_label = _BLOCK_REFERENCE_LABELS.get(
                    block.block_name, block.reference_measurement
                )
                lines.append(
                    f"  → {display_name} — Score: {score:.2f}"
                )
                lines.append(
                    f"  →   Reference: {ref_label} = "
                    f"{block.reference_value:.2f} px"
                )
                lines.append(
                    f"  →   Formulas: {_BLOCK_FORMULAS.get(block.block_name, '')}"
                )
                lines.append(
                    f"  →   Ideal: face_width={block.ideal_face_width:.2f}, "
                    f"face_height={block.ideal_face_height:.2f}, "
                    f"mouth_width={block.ideal_mouth_width:.2f}"
                )
                lines.append("  →   Deviations:")
                for dev in block.deviations:
                    lines.append(
                        f"  →     {ReportFormatter._format_deviation(dev)}"
                    )
            lines.append("")
        else:
            lines.append("")

        # Deviations from Best Reference
        lines.append("Deviations from Best Reference:")
        flagged = [
            dev for dev in analysis.best_block.deviations if dev.is_flagged
        ]
        if flagged:
            for dev in flagged:
                lines.append(
                    f"  [FLAGGED] {dev.measurement_name}: "
                    f"actual={dev.actual}, ideal={dev.ideal}, "
                    f"dev={dev.deviation_percent}%"
                )
        else:
            lines.append(
                "  No significant deviations detected. "
                "All proportions are within the ideal range."
            )
        lines.append("")

        # Global Face Height (from width)
        lines.append("Global Face Height (from width):")
        gdev = analysis.global_face_height_deviation
        actual_str = f"{gdev.actual:.2f}" if gdev.actual is not None else "N/A"
        dev_str = (
            f"{gdev.deviation_percent:.2f}"
            if gdev.deviation_percent is not None
            else "N/A"
        )
        status = "[FLAGGED]" if gdev.is_flagged else "[OK]"
        lines.append(
            f"  Ideal: {analysis.ideal_face_height_from_width:.2f} px | "
            f"Actual: {actual_str} px | "
            f"Deviation: {dev_str}% {status}"
        )
        lines.append("")

        # Relative Feature Size
        lines.append("Relative Feature Size:")
        if analysis.rejected_reference_flags:
            for flag in analysis.rejected_reference_flags:
                desc = REJECTED_FLAG_DESCRIPTIONS.get(flag, flag)
                lines.append(f"  - {flag} ({desc})")
        else:
            lines.append("  No relative feature size flags.")
        lines.append("")

        # Facial Thirds
        if analysis.thirds_proportion_flags:
            lines.append("Facial Thirds:")
            m = analysis.measurements
            if m.upper_third is not None and m.total_face_height is not None:
                ideal_third = m.total_face_height / 3.0
                thirds = [
                    ("Upper", m.upper_third),
                    ("Middle", m.middle_third),
                    ("Lower", m.lower_third),
                ]
            else:
                partial_total = m.middle_third + m.lower_third
                ideal_third = partial_total / 2.0
                thirds = [
                    ("Middle", m.middle_third),
                    ("Lower", m.lower_third),
                ]

            for name, actual in thirds:
                if ideal_third > 0:
                    deviation = ((actual - ideal_third) / ideal_third) * 100
                    is_flagged = abs(deviation) > (
                        THIRDS_DEVIATION_THRESHOLD * 100
                    )
                else:
                    deviation = 0.0
                    is_flagged = False
                status = "[FLAGGED]" if is_flagged else "[OK]"
                lines.append(
                    f"  {name}: {actual:.2f} px "
                    f"(ideal: {ideal_third:.2f} px, {deviation:.2f}%) "
                    f"{status}"
                )
            lines.append(
                "  Status: Some thirds deviate from ideal 1:1:1 ratio."
            )
        else:
            lines.append("Facial Thirds: All thirds are well balanced.")

        return "\n".join(lines)

    @staticmethod
    def format_text_report(
        analysis: VisagismAnalysis,
        image_name: str,
        fallback_used: bool = False,
        debug: bool = False,
    ) -> str:
        """Return full text report with header.

        Parameters
        ----------
        analysis : VisagismAnalysis
            Complete visagism analysis result.
        image_name : str
            Name of the analysed image file.
        fallback_used : bool, optional
            Whether the hairline fallback was used, by default False.
        debug : bool, optional
            When True, include diagnostic detail lines. By default False.

        Returns
        -------
        str
            Multi-line formatted string including header, body, and footer.
        """
        lines: List[str] = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        lines.append("=" * 60)
        lines.append("FACIAL VISAGISM ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {timestamp}")
        lines.append(f"Image: {image_name}")
        if fallback_used:
            lines.append(
                "Note: Hairline not detected. Upper third estimated from "
                "middle third (may reduce accuracy)."
            )
        lines.append("")

        # Body (same as console)
        lines.append(ReportFormatter.format_console(analysis, debug=debug))

        # Footer
        lines.append("")
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def save_report(
        analysis: VisagismAnalysis,
        output_dir: Path,
        image_stem: str,
        fallback_used: bool = False,
        debug: bool = False,
    ) -> Path:
        """Save report to ``analysis_report_[timestamp].txt`` in ``output_dir``.

        Parameters
        ----------
        analysis : VisagismAnalysis
            Complete visagism analysis result.
        output_dir : Path
            Directory where the report file will be written.
        image_stem : str
            Stem of the input image filename (used in the report header).
        fallback_used : bool, optional
            Whether the hairline fallback was used, by default False.
        debug : bool, optional
            When True, include diagnostic detail lines. By default False.

        Returns
        -------
        Path
            Path to the saved report file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{timestamp}.txt"
        report_path = output_dir / filename

        report_content = ReportFormatter.format_text_report(
            analysis, image_name=image_stem, fallback_used=fallback_used,
            debug=debug,
        )
        report_path.write_text(report_content, encoding="utf-8")

        return report_path

    @staticmethod
    def _format_deviation(dev: DeviationResult) -> str:
        """Format a single deviation result as a compact string.

        Parameters
        ----------
        dev : DeviationResult
            The deviation to format.

        Returns
        -------
        str
            Compact representation, e.g.
            ``face_width: actual=180.00, ideal=180.80, dev=-0.44% [OK]``.
        """
        status = "[FLAGGED]" if dev.is_flagged else "[OK]"
        if dev.actual is None:
            return (
                f"{dev.measurement_name}: actual=N/A, "
                f"ideal={dev.ideal:.2f}, dev=N/A {status}"
            )
        return (
            f"{dev.measurement_name}: actual={dev.actual:.2f}, "
            f"ideal={dev.ideal:.2f}, dev={dev.deviation_percent}% {status}"
        )


def _block_score(block: ReferenceBlock) -> float:
    """Compute the sum of absolute deviation percentages for a block.

    Parameters
    ----------
    block : ReferenceBlock
        The reference block to score.

    Returns
    -------
    float
        Sum of absolute deviations for all deviations that have a
        computed percentage.
    """
    return sum(
        abs(dev.deviation_percent)
        for dev in block.deviations
        if dev.deviation_percent is not None
    )
