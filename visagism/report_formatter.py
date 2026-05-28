"""Report formatter for visagism analysis results.

This module provides human-readable formatting for ``VisagismAnalysis``
results, supporting both console output and persistent text reports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from visagism.visagism_calculator import (
    DeviationResult,
    ReferenceBlock,
    VisagismAnalysis,
)


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
        if m.hairline_fallback_used:
            lines.append(f"  Upper Third: {m.upper_third:.2f} px [estimated]")
        else:
            lines.append(f"  Upper Third: {m.upper_third:.2f} px")
        total = m.total_face_height
        if total is not None:
            lines.append(f"  Total Face Height: {total:.2f} px")
        else:
            lines.append("  Total Face Height: N/A")
        lines.append("")

        # === BEST REFERENCE BLOCK ===
        lines.append(
            f"=== BEST REFERENCE BLOCK ({analysis.best_block_name}) ==="
        )
        lines.extend(
            ReportFormatter._format_block(analysis.best_block, prefix="  ")
        )
        lines.append("")

        # === FLAGGED DEVIATIONS ===
        lines.append("=== FLAGGED DEVIATIONS ===")
        flagged = [
            dev for dev in analysis.best_block.deviations if dev.is_flagged
        ]
        if flagged:
            for idx, dev in enumerate(flagged, start=1):
                lines.append(
                    f"  [{idx}] {dev.measurement_name}: "
                    f"actual={dev.actual}, ideal={dev.ideal}, "
                    f"dev={dev.deviation_percent}%"
                )
            lines.append(f"  Total flagged: {len(flagged)}")
        else:
            lines.append(
                "  No significant deviations detected. "
                "All proportions are within the ideal range."
            )

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
    def _format_block(block: ReferenceBlock, prefix: str = "") -> List[str]:
        """Format a single reference block as lines of text.

        Parameters
        ----------
        block : ReferenceBlock
            The reference block to format.
        prefix : str, optional
            String to prepend to each line (e.g. indentation).

        Returns
        -------
        list of str
            Lines representing the block.
        """
        lines: List[str] = []
        lines.append(
            f"{prefix}Ideal Face Width: {block.ideal_face_width:.2f} px"
        )
        lines.append(
            f"{prefix}Ideal Face Height: {block.ideal_face_height:.2f} px"
        )
        lines.append(
            f"{prefix}Ideal Mouth Width: {block.ideal_mouth_width:.2f} px"
        )
        if block.ideal_length_from_width is not None:
            lines.append(
                f"{prefix}Ideal Length from Width: "
                f"{block.ideal_length_from_width:.2f} px"
            )
        lines.append(f"{prefix}Deviations:")
        for dev in block.deviations:
            lines.append(
                f"{prefix}  {ReportFormatter._format_deviation(dev)}"
            )
        return lines

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
