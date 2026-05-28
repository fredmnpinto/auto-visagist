"""Unit tests for visagism.landmark_evaluator module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visagism.landmark_evaluator import (
    EvaluationReport,
    LandmarkEvaluator,
)
from visagism.landmark_ground_truth import LandmarkGroundTruth


class TestLandmarkEvaluatorMetrics:
    """Tests for the static metric computation methods."""

    def test_compute_per_landmark_errors(self) -> None:
        """Test Euclidean error computation for landmarks."""
        pred = [(0, 0)] * 68
        gt = [(3, 4)] * 68  # distance = 5
        errors, valid = LandmarkEvaluator.compute_per_landmark_errors(pred, gt)
        assert len(errors) == 68
        assert all(e == pytest.approx(5.0) for e in errors)
        assert valid == 68

    def test_compute_per_landmark_errors_with_missing(self) -> None:
        """Test that missing landmarks (-1, -1) are skipped."""
        pred = [(0, 0)] * 68
        gt = [(-1, -1)] * 68
        gt[0] = (3, 4)
        errors, valid = LandmarkEvaluator.compute_per_landmark_errors(pred, gt)
        assert errors[0] == pytest.approx(5.0)
        assert all(e == 0.0 for e in errors[1:])
        assert valid == 1

    def test_compute_inter_ocular_distance(self) -> None:
        """Test inter-ocular distance calculation."""
        gt = [(-1, -1)] * 68
        gt[36] = (0, 0)
        gt[45] = (3, 4)
        iod = LandmarkEvaluator.compute_inter_ocular_distance(gt)
        assert iod == pytest.approx(5.0)

    def test_compute_inter_ocular_distance_missing(self) -> None:
        """Test IOD returns 0 when landmarks are missing."""
        gt = [(-1, -1)] * 68
        iod = LandmarkEvaluator.compute_inter_ocular_distance(gt)
        assert iod == 0.0

    def test_compute_nme(self) -> None:
        """Test NME computation."""
        nme = LandmarkEvaluator.compute_nme(
            mean_error_px=5.0, inter_ocular_distance=50.0,
        )
        assert nme == pytest.approx(0.1)

    def test_compute_nme_zero_iod(self) -> None:
        """Test NME returns None for zero inter-ocular distance."""
        nme = LandmarkEvaluator.compute_nme(
            mean_error_px=5.0, inter_ocular_distance=0.0,
        )
        assert nme is None

    def test_compute_region_means(self) -> None:
        """Test per-region mean error computation."""
        errors = [0.0] * 68
        # jaw: indices 0-16, set to 10.0
        for i in range(17):
            errors[i] = 10.0
        # left_eye: indices 36-41, set to 5.0
        for i in range(36, 42):
            errors[i] = 5.0
        region_means = LandmarkEvaluator.compute_region_means(errors)
        assert region_means["jaw"] == pytest.approx(10.0)
        assert region_means["left_eye"] == pytest.approx(5.0)

    def test_compute_region_means_empty_region(self) -> None:
        """Test that regions with all zero errors are omitted."""
        errors = [0.0] * 68
        errors[0] = 10.0
        region_means = LandmarkEvaluator.compute_region_means(errors)
        assert "jaw" in region_means
        assert "left_eye" not in region_means


class TestLandmarkEvaluatorPair:
    """Tests for single-pair evaluation."""

    def test_evaluate_pair_perfect_match(self) -> None:
        """Test evaluation when prediction matches ground truth exactly."""
        landmarks = [(i, i * 2) for i in range(68)]
        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
            hairline_y=50,
        )
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
            hairline_y=50,
        )
        evaluator = LandmarkEvaluator(Path("/tmp"), Path("/tmp"))
        result = evaluator.evaluate_pair(pred, gt)
        assert result.mean_error_px == pytest.approx(0.0)
        assert result.nme == pytest.approx(0.0)
        assert result.hairline_error_px == pytest.approx(0.0)

    def test_evaluate_pair_with_errors(self) -> None:
        """Test evaluation with known errors."""
        pred_landmarks = [(0, 0)] * 68
        gt_landmarks = [(3, 4)] * 68  # distance = 5 for all
        # Set up IOD = 50 using landmarks 36 and 45
        # Both pred and gt have same offset so error remains 5 for these too
        pred_landmarks[36] = (0, 0)
        gt_landmarks[36] = (3, 4)   # error = 5
        pred_landmarks[45] = (50, 0)
        gt_landmarks[45] = (53, 4)  # error = 5, IOD = 50

        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=pred_landmarks,
        )
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=gt_landmarks,
        )
        evaluator = LandmarkEvaluator(Path("/tmp"), Path("/tmp"))
        result = evaluator.evaluate_pair(pred, gt)
        assert result.mean_error_px == pytest.approx(5.0)
        assert result.nme == pytest.approx(0.1)  # 5/50

    def test_evaluate_pair_missing_hairline(self) -> None:
        """Test that missing hairline returns None for hairline error."""
        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(0, 0)] * 68,
            hairline_y=None,
        )
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(0, 0)] * 68,
            hairline_y=50,
        )
        evaluator = LandmarkEvaluator(Path("/tmp"), Path("/tmp"))
        result = evaluator.evaluate_pair(pred, gt)
        assert result.hairline_error_px is None

    def test_evaluate_pair_hairline_error(self) -> None:
        """Test hairline absolute error computation."""
        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(0, 0)] * 68,
            hairline_y=40,
        )
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(0, 0)] * 68,
            hairline_y=50,
        )
        evaluator = LandmarkEvaluator(Path("/tmp"), Path("/tmp"))
        result = evaluator.evaluate_pair(pred, gt)
        assert result.hairline_error_px == pytest.approx(10.0)


class TestLandmarkEvaluatorBatch:
    """Tests for batch evaluation."""

    def test_batch_evaluate_empty_dirs(self, tmp_path: Path) -> None:
        """Test batch evaluation with empty directories."""
        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        evaluator = LandmarkEvaluator(pred_dir, gt_dir)
        report = evaluator.batch_evaluate()
        assert report.summary["num_images"] == 0

    def test_batch_evaluate_matched_pairs(self, tmp_path: Path) -> None:
        """Test batch evaluation with matched file pairs."""
        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Create matching pair
        landmarks = [(i, i) for i in range(68)]
        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
        )
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
        )
        pred.save(pred_dir / "subject_01.json")
        gt.save(gt_dir / "subject_01.json")

        evaluator = LandmarkEvaluator(pred_dir, gt_dir)
        report = evaluator.batch_evaluate()
        assert report.summary["num_images"] == 1
        assert report.summary["overall_mean_error_px"] == pytest.approx(0.0)

    def test_batch_evaluate_missing_gt(self, tmp_path: Path) -> None:
        """Test that missing ground truth is reported in skipped."""
        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        landmarks = [(i, i) for i in range(68)]
        pred = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
        )
        pred.save(pred_dir / "subject_01.json")

        evaluator = LandmarkEvaluator(pred_dir, gt_dir)
        report = evaluator.batch_evaluate()
        assert report.summary["num_images"] == 0
        assert any("No ground truth" in s for s in report.skipped_files)

    def test_batch_evaluate_malformed_json(self, tmp_path: Path) -> None:
        """Test graceful handling of malformed JSON."""
        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        (pred_dir / "bad.json").write_text("not json")
        (gt_dir / "bad.json").write_text("not json")

        evaluator = LandmarkEvaluator(pred_dir, gt_dir)
        report = evaluator.batch_evaluate()
        assert report.summary["num_images"] == 0
        assert any("Malformed JSON" in s for s in report.skipped_files)

    def test_batch_evaluate_multiple_images(self, tmp_path: Path) -> None:
        """Test batch evaluation with multiple image pairs."""
        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        for i, name in enumerate(["img1", "img2"]):
            landmarks = [(i, i) for _ in range(68)]
            pred = LandmarkGroundTruth(
                image_path=Path(f"/tmp/{name}.jpg"),
                image_width=100,
                image_height=100,
                landmarks_68=landmarks,
            )
            gt = LandmarkGroundTruth(
                image_path=Path(f"/tmp/{name}.jpg"),
                image_width=100,
                image_height=100,
                landmarks_68=[(0, 0)] * 68,
            )
            pred.save(pred_dir / f"{name}.json")
            gt.save(gt_dir / f"{name}.json")

        evaluator = LandmarkEvaluator(pred_dir, gt_dir)
        report = evaluator.batch_evaluate()
        assert report.summary["num_images"] == 2
        assert len(report.per_image) == 2


class TestLandmarkEvaluatorReport:
    """Tests for report generation and formatting."""

    def test_generate_console_table(self) -> None:
        """Test console table generation."""
        report = EvaluationReport(
            summary={
                "num_images": 2,
                "overall_mean_error_px": 5.0,
                "overall_nme": 0.05,
                "mean_hairline_error_px": 10.0,
            },
            per_image=[
                {
                    "image_stem": "img1",
                    "mean_error_px": 4.0,
                    "nme": 0.04,
                    "hairline_error_px": 8.0,
                    "per_region_means": {"jaw": 5.0},
                },
            ],
            per_region_overall={"jaw": 5.0},
            skipped_files=["missing: img3"],
        )
        table = LandmarkEvaluator.generate_console_table(report)
        assert "LANDMARK EVALUATION REPORT" in table
        assert "img1" in table
        assert "Jaw" in table
        assert "missing: img3" in table

    def test_report_save(self, tmp_path: Path) -> None:
        """Test saving report to JSON file."""
        report = EvaluationReport(
            summary={"num_images": 1, "overall_mean_error_px": 0.0},
            per_image=[],
            per_region_overall={},
            skipped_files=[],
        )
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["summary"]["num_images"] == 1

    def test_report_to_json(self) -> None:
        """Test report JSON serialization."""
        report = EvaluationReport(
            summary={"num_images": 1},
            per_image=[{"image_stem": "test"}],
            per_region_overall={},
            skipped_files=[],
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["summary"]["num_images"] == 1
