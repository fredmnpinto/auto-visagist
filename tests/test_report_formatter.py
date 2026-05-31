"""Tests for the ReportFormatter module."""

from __future__ import annotations

from pathlib import Path

import pytest

from visagism.report_formatter import ReportFormatter
from visagism.visagism_calculator import (
    ConsensusResult,
    DeviationResult,
    FacialMeasurements,
    ReferenceBlock,
    VisagismAnalysis,
    VisagismCalculator,
)


class TestFormatConsole:
    """Test suite for ReportFormatter.format_console()."""

    @pytest.fixture
    def sample_analysis(self) -> VisagismAnalysis:
        """Return a VisagismAnalysis with known measurements."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        return calc.calculate()

    def test_includes_measurements_section(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains the facial measurements section."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== FACIAL MEASUREMENTS ===" in output
        assert "Eye Width: 45.20 px" in output
        assert "Face Width: 180.00 px" in output

    def test_includes_best_reference_block(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains the best reference within proportion analysis."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        assert "Best Reference:" in output
        assert sample_analysis.best_block_name in output

    def test_includes_flagged_deviations(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains deviations within proportion analysis."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        assert "Deviations from Best Reference:" in output
        # With these measurements, some deviations should be flagged
        assert (
            "[FLAGGED]" in output
            or "No significant deviations detected" in output
        )

    def test_no_flagged_shows_positive_message(self) -> None:
        """When no deviations are flagged, a positive message is shown."""
        # Manually construct an analysis with no flagged deviations
        # to test the formatter's handling of the empty-flagged case.
        measurements = FacialMeasurements(
            eye_width=45.0,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        block = ReferenceBlock(
            block_name="Test Block",
            reference_measurement="test",
            reference_value=10.0,
            ideal_face_width=40.0,
            ideal_face_height=60.0,
            ideal_mouth_width=15.0,
            deviations=[
                DeviationResult(
                    measurement_name="face_width",
                    actual=180.0,
                    ideal=180.0,
                    deviation_percent=0.0,
                    is_flagged=False,
                ),
            ],
        )
        consensus = ConsensusResult(
            ideal_face_width=40.0,
            ideal_face_height=60.0,
            ideal_mouth_width=15.0,
            deviations=[
                DeviationResult(
                    measurement_name="face_width",
                    actual=180.0,
                    ideal=180.0,
                    deviation_percent=0.0,
                    is_flagged=False,
                ),
            ],
        )
        analysis = VisagismAnalysis(
            measurements=measurements,
            block_1_eye_width=block,
            block_2_inter_ocular=block,
            block_3_nose_width=block,
            consensus=consensus,
            all_flagged_deviations=[],
            best_block=block,
            best_block_name=block.block_name,
        )
        output = ReportFormatter.format_console(analysis)
        assert "No significant deviations detected" in output
        assert "All proportions are within the ideal range." in output

    def test_fallback_shows_estimated_label(self) -> None:
        """When fallback is used, upper third shows [estimated]."""
        calc = VisagismCalculator(
            eye_width=45.0,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=62.0,
            hairline_fallback_used=True,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis)
        assert "Upper Third: 62.00 px [estimated]" in output

    def test_no_fallback_shows_normal_label(self) -> None:
        """When fallback is not used, upper third shows without [estimated]."""
        calc = VisagismCalculator(
            eye_width=45.0,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
            hairline_fallback_used=False,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis)
        assert "Upper Third: 67.00 px" in output
        assert "[estimated]" not in output.split("Upper Third:")[1].split("\n")[0]

    def test_includes_global_face_height_section(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains global face height within proportion analysis."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        assert "Global Face Height (from width):" in output
        assert "Ideal:" in output
        assert "Actual:" in output
        assert "Deviation:" in output

    def test_includes_relative_feature_size_section(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains relative feature size within proportion analysis."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        assert "Relative Feature Size:" in output

    def test_includes_facial_thirds_section(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Console output contains facial thirds within proportion analysis."""
        output = ReportFormatter.format_console(sample_analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        # Should show either detailed or compact facial thirds
        assert (
            "Facial Thirds:" in output
            or "Facial Thirds: All thirds are well balanced." in output
        )

    def test_total_face_height_na_when_upper_third_none(self) -> None:
        """When upper_third is None, total face height shows N/A."""
        calc = VisagismCalculator(
            eye_width=45.0,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=None,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis)
        assert "Total Face Height: N/A" in output

    def test_thirds_flags_displayed_when_present(self) -> None:
        """When thirds_proportion_flags is non-empty, detailed thirds are shown."""
        calc = VisagismCalculator(
            eye_width=10.0,
            inter_ocular_distance=20.0,
            nose_width=15.0,
            mouth_width=30.0,
            face_width=100.0,
            lower_third=30.0,
            middle_third=50.0,
            upper_third=50.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis)
        assert "=== PROPORTION ANALYSIS ===" in output
        assert "Facial Thirds:" in output
        # With these values, thirds should be flagged
        assert "[FLAGGED]" in output
        assert "Status: Some thirds deviate from ideal 1:1:1 ratio." in output

    def test_thirds_balanced_shows_compact_message(self) -> None:
        """When thirds are balanced, show compact message only."""
        calc = VisagismCalculator(
            eye_width=45.0,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=60.0,
            upper_third=60.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis)
        assert "Facial Thirds: All thirds are well balanced." in output
        assert "Status: Some thirds deviate" not in output

    def test_format_deviation_with_none_actual(self) -> None:
        """_format_deviation handles None actual value."""
        dev = DeviationResult(
            measurement_name="face_height",
            actual=None,
            ideal=100.0,
            deviation_percent=None,
            is_flagged=False,
        )
        formatted = ReportFormatter._format_deviation(dev)
        assert "actual=N/A" in formatted
        assert "dev=N/A" in formatted

    def test_debug_false_is_identical_to_current(self) -> None:
        """format_console(analysis, debug=False) equals format_console(analysis)."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        default_output = ReportFormatter.format_console(analysis)
        explicit_output = ReportFormatter.format_console(analysis, debug=False)
        assert default_output == explicit_output

    def test_debug_includes_measurement_formulas(self) -> None:
        """Every measurement has a → Formula: line when debug=True."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis, debug=True)
        assert "→ Formula: avg(dist(36,39), dist(42,45))" in output
        assert "→ Formula: dist(39,42)" in output
        assert "→ Formula: dist(31,35)" in output
        assert "→ Formula: dist(48,54)" in output
        assert "→ Formula: dist(0,16)" in output
        assert "→ Formula: abs(33y - 8y)" in output
        assert "→ Formula: abs(avg_eyebrow_y - 33y)" in output
        assert "→ Formula: abs(avg_eyebrow_y - hairline_y)" in output

    def test_debug_includes_intermediate_values(self) -> None:
        """Eye-width shows left/right; middle/upper third shows avg_eyebrow_y."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
            left_eye_width=44.5,
            right_eye_width=45.9,
            avg_eyebrow_y=95.0,
            hairline_y=28.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis, debug=True)
        assert "left=44.50 px, right=45.90 px" in output
        assert "avg_eyebrow_y=95.00 px" in output
        assert "hairline_y=28.00 px" in output

    def test_debug_omits_none_intermediates(self) -> None:
        """When hairline_y is None, no None value appears in output."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=62.0,
            hairline_fallback_used=True,
            left_eye_width=44.5,
            right_eye_width=45.9,
            avg_eyebrow_y=95.0,
            hairline_y=None,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis, debug=True)
        assert "None" not in output

    def test_debug_includes_all_reference_blocks(self) -> None:
        """Debug output contains all three block names with scores."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis, debug=True)
        assert "Block 1 (Eye Width)" in output
        assert "Block 2 (Inter-Ocular)" in output
        assert "Block 3 (Nose Width)" in output
        assert "Score:" in output

    def test_debug_no_separate_debug_section(self) -> None:
        """Assert === DEBUG is NOT present in debug output."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_console(analysis, debug=True)
        assert "=== DEBUG" not in output

    def test_debug_text_report_propagates(self) -> None:
        """format_text_report(..., debug=True) contains → markers."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        output = ReportFormatter.format_text_report(
            analysis, image_name="test.jpg", debug=True
        )
        assert "→ Formula:" in output
        assert "Block 1 (Eye Width)" in output

    def test_debug_save_report_propagates(self) -> None:
        """Saved file contains → markers when debug=True."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        analysis = calc.calculate()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = ReportFormatter.save_report(
                analysis, output_dir=tmp_path, image_stem="test", debug=True
            )
            content = report_path.read_text(encoding="utf-8")
            assert "→ Formula:" in content
            assert "Block 1 (Eye Width)" in content


class TestFormatTextReport:
    """Test suite for ReportFormatter.format_text_report()."""

    @pytest.fixture
    def sample_analysis(self) -> VisagismAnalysis:
        """Return a VisagismAnalysis with known measurements."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        return calc.calculate()

    def test_includes_header_with_image_name(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Text report header includes the image name."""
        output = ReportFormatter.format_text_report(
            sample_analysis, image_name="test_photo.jpg"
        )
        assert "FACIAL VISAGISM ANALYSIS REPORT" in output
        assert "Image: test_photo.jpg" in output

    def test_includes_timestamp(self, sample_analysis: VisagismAnalysis) -> None:
        """Text report header includes a timestamp."""
        output = ReportFormatter.format_text_report(
            sample_analysis, image_name="test.jpg"
        )
        assert "Timestamp:" in output

    def test_includes_fallback_status_when_true(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Text report notes fallback usage when True."""
        output = ReportFormatter.format_text_report(
            sample_analysis, image_name="test.jpg", fallback_used=True
        )
        assert "Hairline not detected" in output
        assert "may reduce accuracy" in output

    def test_omits_fallback_status_when_false(
        self, sample_analysis: VisagismAnalysis
    ) -> None:
        """Text report omits fallback note when False."""
        output = ReportFormatter.format_text_report(
            sample_analysis, image_name="test.jpg", fallback_used=False
        )
        assert "Hairline not detected" not in output

    def test_includes_footer(self, sample_analysis: VisagismAnalysis) -> None:
        """Text report ends with a footer."""
        output = ReportFormatter.format_text_report(
            sample_analysis, image_name="test.jpg"
        )
        assert "End of Report" in output


class TestSaveReport:
    """Test suite for ReportFormatter.save_report()."""

    @pytest.fixture
    def sample_analysis(self) -> VisagismAnalysis:
        """Return a VisagismAnalysis with known measurements."""
        calc = VisagismCalculator(
            eye_width=45.2,
            inter_ocular_distance=30.0,
            nose_width=25.0,
            mouth_width=50.0,
            face_width=180.0,
            lower_third=60.0,
            middle_third=62.0,
            upper_third=67.0,
        )
        return calc.calculate()

    def test_creates_file_with_correct_pattern(
        self, sample_analysis: VisagismAnalysis, tmp_path: Path
    ) -> None:
        """Saved report filename matches ``analysis_report_*.txt`` pattern."""
        report_path = ReportFormatter.save_report(
            sample_analysis, output_dir=tmp_path, image_stem="test"
        )
        assert report_path.exists()
        assert report_path.name.startswith("analysis_report_")
        assert report_path.suffix == ".txt"
        assert report_path.parent == tmp_path

    def test_content_matches_format_text_report(
        self, sample_analysis: VisagismAnalysis, tmp_path: Path
    ) -> None:
        """Saved file content equals ``format_text_report`` output."""
        report_path = ReportFormatter.save_report(
            sample_analysis,
            output_dir=tmp_path,
            image_stem="my_image",
            fallback_used=True,
        )
        expected = ReportFormatter.format_text_report(
            sample_analysis, image_name="my_image", fallback_used=True
        )
        actual = report_path.read_text(encoding="utf-8")
        assert actual == expected

    def test_creates_output_directory_if_missing(
        self, sample_analysis: VisagismAnalysis, tmp_path: Path
    ) -> None:
        """Output directory is created automatically if it does not exist."""
        nested = tmp_path / "nested" / "dir"
        report_path = ReportFormatter.save_report(
            sample_analysis, output_dir=nested, image_stem="test"
        )
        assert nested.exists()
        assert report_path.exists()
