"""Landmark evaluation engine for comparing predictions against ground truth.

Mode 2 of the Landmark Evaluation Tool. Computes per-landmark Euclidean
errors, per-region means, Normalized Mean Error (NME), and hairline
errors. Generates both console tables and JSON reports.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from visagism.constants import REGION_INDICES, REGION_LABELS
from visagism.landmark_ground_truth import LandmarkGroundTruth
from visagism.types import LandmarksList


@dataclass
class ImageEvaluationResult:
    """Evaluation results for a single image pair.

    Parameters
    ----------
    image_stem : str
        Base filename (without extension) of the evaluated image.
    per_landmark_errors : list of float
        Euclidean error in pixels for each of the 68 landmarks.
    mean_error_px : float
        Mean Euclidean error across all 68 landmarks.
    nme : float or None
        Normalized Mean Error (mean error / inter-ocular distance).
        ``None`` if inter-ocular distance is zero.
    hairline_error_px : float or None
        Absolute hairline error in pixels. ``None`` if either ground
        truth or prediction lacks a hairline.
    per_region_means : dict
        Mean error per facial region.
    num_valid_landmarks : int
        Number of landmarks that were valid in both prediction and
        ground truth (not ``(-1, -1)``).
    """

    image_stem: str
    per_landmark_errors: list[float] = field(default_factory=list)
    mean_error_px: float = 0.0
    nme: float | None = None
    hairline_error_px: float | None = None
    per_region_means: dict[str, float] = field(default_factory=dict)
    num_valid_landmarks: int = 0


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across multiple images.

    Parameters
    ----------
    summary : dict
        High-level statistics including ``num_images``,
        ``overall_mean_error_px``, ``overall_nme``,
        ``mean_hairline_error_px``.
    per_image : list of dict
        Per-image results in serializable form.
    per_region_overall : dict
        Mean error per region across all images.
    skipped_files : list of str
        List of files that were skipped and why.
    """

    summary: dict[str, Any] = field(default_factory=dict)
    per_image: list[dict[str, Any]] = field(default_factory=list)
    per_region_overall: dict[str, float] = field(default_factory=dict)
    skipped_files: list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(asdict(self), indent=indent)

    def save(self, path: Path) -> None:
        """Save the report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


class LandmarkEvaluator:
    """Evaluates predicted landmarks against ground truth.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing prediction JSON files.
    ground_truth_dir : Path
        Directory containing ground truth JSON files.
    """

    def __init__(
        self,
        predictions_dir: Path,
        ground_truth_dir: Path,
    ) -> None:
        """Initialise the evaluator with source directories."""
        self._predictions_dir = predictions_dir
        self._ground_truth_dir = ground_truth_dir

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_per_landmark_errors(
        pred_landmarks: LandmarksList,
        gt_landmarks: LandmarksList,
    ) -> tuple[list[float], int]:
        """Compute Euclidean error for each landmark pair.

        Parameters
        ----------
        pred_landmarks : LandmarksList
            Predicted 68 landmarks.
        gt_landmarks : LandmarksList
            Ground truth 68 landmarks.

        Returns
        -------
        tuple
            ``(errors, valid_count)`` where ``errors`` is a list of 68
            float values and ``valid_count`` is the number of valid
            (non-``(-1, -1)``) pairs.
        """
        errors: list[float] = []
        valid_count = 0

        for i in range(68):
            pred_pt = pred_landmarks[i]
            gt_pt = gt_landmarks[i]

            if pred_pt == (-1, -1) or gt_pt == (-1, -1):
                errors.append(0.0)
                continue

            dist = math.dist(pred_pt, gt_pt)
            errors.append(dist)
            valid_count += 1

        return errors, valid_count

    @staticmethod
    def compute_inter_ocular_distance(gt_landmarks: LandmarksList) -> float:
        """Compute the inter-ocular distance from ground truth.

        Uses the distance between the outer corners of the left eye
        (landmark 36) and right eye (landmark 45).

        Parameters
        ----------
        gt_landmarks : LandmarksList
            Ground truth 68 landmarks.

        Returns
        -------
        float
            Inter-ocular distance in pixels. Returns 0.0 if either
            landmark is missing.
        """
        left_outer = gt_landmarks[36]
        right_outer = gt_landmarks[45]

        if left_outer == (-1, -1) or right_outer == (-1, -1):
            return 0.0

        return math.dist(left_outer, right_outer)

    @staticmethod
    def compute_nme(
        mean_error_px: float,
        inter_ocular_distance: float,
    ) -> float | None:
        """Compute Normalized Mean Error.

        NME = mean_error_px / inter_ocular_distance

        Parameters
        ----------
        mean_error_px : float
            Mean Euclidean error in pixels.
        inter_ocular_distance : float
            Inter-ocular distance in pixels.

        Returns
        -------
        float or None
            NME value, or ``None`` if inter-ocular distance is zero.
        """
        if inter_ocular_distance <= 0:
            return None
        return mean_error_px / inter_ocular_distance

    @staticmethod
    def compute_region_means(
        per_landmark_errors: list[float],
    ) -> dict[str, float]:
        """Compute mean error per facial region.

        Parameters
        ----------
        per_landmark_errors : list of float
            List of 68 per-landmark errors.

        Returns
        -------
        dict
            Mapping from region name to mean error. Regions with no
            valid landmarks are omitted.
        """
        region_means: dict[str, float] = {}

        for region, indices in REGION_INDICES.items():
            region_errors = [
                per_landmark_errors[i]
                for i in indices
                if per_landmark_errors[i] > 0
            ]
            if region_errors:
                region_means[region] = sum(region_errors) / len(region_errors)

        return region_means

    # ------------------------------------------------------------------
    # Pair / batch evaluation
    # ------------------------------------------------------------------

    def evaluate_pair(
        self,
        prediction: LandmarkGroundTruth,
        ground_truth: LandmarkGroundTruth,
    ) -> ImageEvaluationResult:
        """Evaluate a single prediction against ground truth.

        Parameters
        ----------
        prediction : LandmarkGroundTruth
            Predicted landmarks.
        ground_truth : LandmarkGroundTruth
            Ground truth landmarks.

        Returns
        -------
        ImageEvaluationResult
            Evaluation results for this image pair.
        """
        errors, valid_count = self.compute_per_landmark_errors(
            prediction.landmarks_68, ground_truth.landmarks_68,
        )

        mean_error = (
            sum(errors) / valid_count if valid_count > 0 else 0.0
        )

        iod = self.compute_inter_ocular_distance(ground_truth.landmarks_68)
        nme = self.compute_nme(mean_error, iod)

        # Hairline error
        hairline_error: float | None = None
        if (
            prediction.hairline_y is not None
            and ground_truth.hairline_y is not None
        ):
            hairline_error = abs(prediction.hairline_y - ground_truth.hairline_y)

        region_means = self.compute_region_means(errors)

        return ImageEvaluationResult(
            image_stem=ground_truth.image_path.stem,
            per_landmark_errors=errors,
            mean_error_px=mean_error,
            nme=nme,
            hairline_error_px=hairline_error,
            per_region_means=region_means,
            num_valid_landmarks=valid_count,
        )

    def batch_evaluate(self) -> EvaluationReport:
        """Run batch evaluation over all matched file pairs.

        Matches prediction and ground truth files by image stem
        (filename without extension). Skips unmatched or malformed
        files gracefully.

        Returns
        -------
        EvaluationReport
            Aggregated evaluation report.
        """
        # Build stem -> path mappings
        pred_files = {
            p.stem: p for p in self._predictions_dir.glob("*.json")
        }
        gt_files = {
            p.stem: p for p in self._ground_truth_dir.glob("*.json")
        }

        # Find common stems
        common_stems = sorted(set(pred_files.keys()) & set(gt_files.keys()))

        skipped: list[str] = []
        per_image_results: list[ImageEvaluationResult] = []

        # Report missing pairs
        for stem in sorted(set(pred_files.keys()) - set(gt_files.keys())):
            skipped.append(f"No ground truth for prediction: {stem}")
        for stem in sorted(set(gt_files.keys()) - set(pred_files.keys())):
            skipped.append(f"No prediction for ground truth: {stem}")

        for stem in common_stems:
            pred_path = pred_files[stem]
            gt_path = gt_files[stem]

            try:
                pred = LandmarkGroundTruth.load(pred_path)
                gt = LandmarkGroundTruth.load(gt_path)
            except Exception as exc:
                skipped.append(f"Malformed JSON for {stem}: {exc}")
                continue

            result = self.evaluate_pair(pred, gt)
            per_image_results.append(result)

        # Build aggregated report
        return self._build_report(per_image_results, skipped)

    def _build_report(
        self,
        results: list[ImageEvaluationResult],
        skipped: list[str],
    ) -> EvaluationReport:
        """Build an EvaluationReport from per-image results."""
        num_images = len(results)

        if num_images == 0:
            return EvaluationReport(
                summary={
                    "num_images": 0,
                    "overall_mean_error_px": 0.0,
                    "overall_nme": None,
                    "mean_hairline_error_px": None,
                },
                per_image=[],
                per_region_overall={},
                skipped_files=skipped,
            )

        # Overall mean error
        all_mean_errors = [r.mean_error_px for r in results]
        overall_mean = sum(all_mean_errors) / len(all_mean_errors)

        # Overall NME
        valid_nmes = [r.nme for r in results if r.nme is not None]
        overall_nme = sum(valid_nmes) / len(valid_nmes) if valid_nmes else None

        # Mean hairline error
        valid_hairline = [
            r.hairline_error_px for r in results
            if r.hairline_error_px is not None
        ]
        mean_hairline = (
            sum(valid_hairline) / len(valid_hairline)
            if valid_hairline else None
        )

        # Per-region overall
        region_sums: dict[str, list[float]] = {}
        for r in results:
            for region, mean_err in r.per_region_means.items():
                region_sums.setdefault(region, []).append(mean_err)

        per_region_overall = {
            region: sum(values) / len(values)
            for region, values in region_sums.items()
        }

        # Per-image serializable
        per_image = []
        for r in results:
            entry: dict[str, Any] = {
                "image_stem": r.image_stem,
                "mean_error_px": round(r.mean_error_px, 2),
                "nme": round(r.nme, 4) if r.nme is not None else None,
                "hairline_error_px": (
                    round(r.hairline_error_px, 1)
                    if r.hairline_error_px is not None else None
                ),
                "per_region_means": {
                    k: round(v, 2) for k, v in r.per_region_means.items()
                },
            }
            per_image.append(entry)

        summary = {
            "num_images": num_images,
            "overall_mean_error_px": round(overall_mean, 2),
            "overall_nme": round(overall_nme, 4) if overall_nme is not None else None,
            "mean_hairline_error_px": (
                round(mean_hairline, 1) if mean_hairline is not None else None
            ),
        }

        return EvaluationReport(
            summary=summary,
            per_image=per_image,
            per_region_overall={
                k: round(v, 2) for k, v in per_region_overall.items()
            },
            skipped_files=skipped,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    @staticmethod
    def generate_console_table(report: EvaluationReport) -> str:
        """Generate a human-readable console table from a report.

        Parameters
        ----------
        report : EvaluationReport
            The evaluation report to format.

        Returns
        -------
        str
            Multi-line string suitable for printing to console.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("LANDMARK EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        summary = report.summary
        lines.append("Summary:")
        lines.append(f"  Images evaluated: {summary['num_images']}")
        lines.append(
            f"  Overall mean error: {summary['overall_mean_error_px']} px"
        )
        nme = summary.get("overall_nme")
        lines.append(
            f"  Overall NME: {nme if nme is not None else 'N/A'}"
        )
        hairline = summary.get("mean_hairline_error_px")
        lines.append(
            f"  Mean hairline error: "
            f"{hairline if hairline is not None else 'N/A'} px"
        )
        lines.append("")

        # Per-region overall
        if report.per_region_overall:
            lines.append("Per-region mean errors:")
            for region, mean_err in sorted(report.per_region_overall.items()):
                label = REGION_LABELS.get(region, region)
                lines.append(f"  {label:20s}: {mean_err:6.2f} px")
            lines.append("")

        # Per-image
        if report.per_image:
            lines.append("Per-image results:")
            lines.append(
                f"  {'Image':<20s} {'Mean(px)':<10s} {'NME':<10s} "
                f"{'Hairline':<10s}"
            )
            lines.append("  " + "-" * 60)
            for entry in report.per_image:
                nme_str = (
                    f"{entry['nme']:.4f}"
                    if entry["nme"] is not None else "N/A"
                )
                hair_str = (
                    f"{entry['hairline_error_px']:.1f}"
                    if entry["hairline_error_px"] is not None else "N/A"
                )
                lines.append(
                    f"  {entry['image_stem']:<20s} "
                    f"{entry['mean_error_px']:<10.2f} "
                    f"{nme_str:<10s} {hair_str:<10s}"
                )
            lines.append("")

        # Skipped files
        if report.skipped_files:
            lines.append("Skipped files:")
            for msg in report.skipped_files:
                lines.append(f"  - {msg}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
