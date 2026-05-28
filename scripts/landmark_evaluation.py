"""CLI entry point for the Landmark Evaluation Tool.

Dual-mode tool for facial landmark ground truth labeling and evaluation.

Mode 1 (label): Interactive OpenCV GUI for manually annotating 68 facial
landmarks plus hairline position.

Mode 2 (evaluate): Batch comparison of predicted landmarks against ground
truth with error metrics and report generation.

Usage:
    python scripts/landmark_evaluation.py --mode label --input <image_or_dir>
    python scripts/landmark_evaluation.py --mode evaluate \
        --predictions-dir <dir> --ground-truth-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without package install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from visagism.errors import VisagismError  # noqa: E402
from visagism.landmark_evaluator import LandmarkEvaluator  # noqa: E402
from visagism.landmark_labeler import LandmarkLabeler  # noqa: E402


def _collect_image_paths(input_path: Path) -> list[Path]:
    """Collect image paths from a file or directory.

    Parameters
    ----------
    input_path : Path
        Path to an image file or directory containing images.

    Returns
    -------
    list of Path
        List of image file paths.
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(input_path.glob(ext))
            paths.extend(input_path.glob(ext.upper()))
        return sorted(set(paths))

    return []


def _run_label_mode(args: argparse.Namespace) -> None:
    """Run the interactive labeling GUI.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    input_path = Path(args.input)
    image_paths = _collect_image_paths(input_path)

    if not image_paths:
        print(f"Error: No valid images found at {input_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Labeling mode: {len(image_paths)} image(s)")
    print(f"Output directory: {output_dir.resolve()}")
    print("Predictions enabled: will pre-fill with dlib")

    labeler = LandmarkLabeler(
        image_paths=image_paths,
        output_dir=output_dir,
    )
    labeler.run()
    print("Labeling complete.")


def _run_evaluate_mode(args: argparse.Namespace) -> None:
    """Run batch evaluation and generate reports.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    predictions_dir = Path(args.predictions_dir)
    ground_truth_dir = Path(args.ground_truth_dir)

    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        sys.exit(1)
    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {ground_truth_dir}")
        sys.exit(1)

    print("Evaluation mode:")
    print(f"  Predictions: {predictions_dir.resolve()}")
    print(f"  Ground truth: {ground_truth_dir.resolve()}")

    evaluator = LandmarkEvaluator(predictions_dir, ground_truth_dir)
    report = evaluator.batch_evaluate()

    # Print console table
    table = LandmarkEvaluator.generate_console_table(report)
    print("\n" + table)

    # Save JSON report
    if args.report:
        report_path = Path(args.report)
        report.save(report_path)
        print(f"\nJSON report saved to: {report_path.resolve()}")


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        prog="landmark_evaluation.py",
        description=(
            "Landmark Evaluation Tool — dual-mode tool for facial "
            "landmark ground truth labeling and evaluation."
        ),
        epilog=(
            "Examples:\n"
            "  # Label a single image\n"
            "  python scripts/landmark_evaluation.py --mode label "
            "--input photo.jpg\n\n"
            "  # Label all images in a directory\n"
            "  python scripts/landmark_evaluation.py --mode label "
            "--input ./photos/\n\n"
            "  # Evaluate predictions against ground truth\n"
            "  python scripts/landmark_evaluation.py --mode evaluate "
            "--predictions-dir ./pred/ --ground-truth-dir ./gt/ "
            "--report report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["label", "evaluate"],
        required=True,
        help="Tool mode: 'label' for annotation, 'evaluate' for comparison",
    )

    # Label mode arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input image or directory (label mode)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ground_truth",
        help=(
            "Output directory for ground truth JSON "
            "(label mode, default: ground_truth)"
        ),
    )

    # Evaluate mode arguments
    parser.add_argument(
        "--predictions-dir", "-p",
        type=str,
        help="Directory containing prediction JSON files (evaluate mode)",
    )
    parser.add_argument(
        "--ground-truth-dir", "-g",
        type=str,
        help="Directory containing ground truth JSON files (evaluate mode)",
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Path to save JSON evaluation report (evaluate mode)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "label":
            if not args.input:
                parser.error("--input is required for label mode")
            _run_label_mode(args)
        elif args.mode == "evaluate":
            if not args.predictions_dir or not args.ground_truth_dir:
                parser.error(
                    "--predictions-dir and --ground-truth-dir are required "
                    "for evaluate mode"
                )
            _run_evaluate_mode(args)
    except VisagismError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
