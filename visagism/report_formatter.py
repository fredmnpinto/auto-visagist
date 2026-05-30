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


class ReportFormatter:
    """Format ``VisagismAnalysis`` results for console and file output."""

    @staticmethod
    def format_console(analysis: VisagismAnalysis) -> str:
        """Return human-readable console output.

        Parameters
        ----------
        analysis : VisagismAnalysis
            Complete visagism analysis result.

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
        lines.append(f"  Inter-Ocular Distance: {m.inter_ocular_distance:.2f} px")
        lines.append(f"  Nose Width: {m.nose_width:.2f} px")
        lines.append(f"  Mouth Width: {m.mouth_width:.2f} px")
        lines.append(f"  Face Width: {m.face_width:.2f} px")
        lines.append(f"  Lower Third: {m.lower_third:.2f} px")
        lines.append(f"  Middle Third: {m.middle_third:.2f} px")
        if m.upper_third is not None:
            if m.hairline_fallback_used:
                lines.append(
                    f"  Upper Third: {m.upper_third:.2f} px [estimated]"
                )
            else:
                lines.append(f"  Upper Third: {m.upper_third:.2f} px")
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
        lines.append(ReportFormatter.format_console(analysis))

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
            analysis, image_name=image_stem, fallback_used=fallback_used
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
