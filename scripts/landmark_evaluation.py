"""CLI entry point for the Landmark Evaluation Tool.

Dual-mode tool for facial landmark ground truth labeling and evaluation.

Mode 1 (label): Interactive OpenCV GUI for manually annotating 68 facial
landmarks plus hairline position.

Mode 2 (evaluate): Batch comparison of live-detected landmarks against
ground truth with error metrics and report generation.

Usage:
    python scripts/landmark_evaluation.py --mode label --input <image_or_dir>
    python scripts/landmark_evaluation.py --mode evaluate \
        --images-dir <dir> --ground-truth-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without package install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from visagism.detector_factory import create_landmark_detector  # noqa: E402
from visagism.errors import VisagismError  # noqa: E402
from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.hairline_detector import HairlineDetector  # noqa: E402
from visagism.image_loader import ImageLoader  # noqa: E402
from visagism.landmark_evaluator import LandmarkEvaluator  # noqa: E402
from visagism.landmark_ground_truth import LandmarkGroundTruth  # noqa: E402
from visagism.landmark_labeler import LandmarkLabeler  # noqa: E402
from visagism.model_finder import ModelFinder  # noqa: E402


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
        paths: list[Path] = []
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

    output_dir = Path(args.output) if args.output else Path("data/ground_truth")
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
    """Run batch evaluation using live detection and generate reports.

    For each image in ``images_dir``, runs the full detection pipeline
    (face detection → landmarks → hairline), creates a
    ``LandmarkGroundTruth`` from the live predictions, and compares it
    against the corresponding ground truth file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    images_dir = Path(args.images_dir)
    ground_truth_dir = Path(args.ground_truth_dir)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {ground_truth_dir}")
        sys.exit(1)

    # Collect images
    image_paths = _collect_image_paths(images_dir)
    if not image_paths:
        print(f"Error: No valid images found in {images_dir}")
        sys.exit(1)

    # Initialise detection pipeline (once)
    model_path = ModelFinder.find()
    face_detector = FaceDetector()
    landmark_detector = create_landmark_detector(args.detector, model_path)
    hairline_detector = HairlineDetector()

    # Load ground truth files by stem (strip _gt suffix)
    gt_files = {}
    gt_original_stems = {}
    for p in ground_truth_dir.glob("*.json"):
        stem = p.stem
        original_stem = stem
        if stem.endswith("_gt"):
            stem = stem[:-3]  # Remove _gt suffix
        gt_files[stem] = p
        gt_original_stems[stem] = original_stem

    pairs: list[tuple[LandmarkGroundTruth, LandmarkGroundTruth]] = []
    skipped: list[str] = []

    # Report ground truth files without matching images
    image_stems = {p.stem for p in image_paths}
    for stem in sorted(set(gt_files.keys()) - image_stems):
        skipped.append(
            f"No image for ground truth: {gt_original_stems[stem]}"
        )

    for img_path in image_paths:
        stem = img_path.stem
        if stem not in gt_files:
            skipped.append(f"No ground truth for image: {stem}")
            continue

        try:
            # 1. Load image
            img_bgr, img_gray = ImageLoader.load(img_path)

            # 2. Detect face
            face_rect = face_detector.detect(img_gray, img_bgr.shape[:2])

            # 3. Detect landmarks
            landmarks = landmark_detector.detect(
                img_gray, face_rect, img_path,
            )

            # 4. Detect hairline
            result = hairline_detector.detect(img_gray, landmarks)
            hairline_y = result["hairline_y"]

            # 5. Create prediction ground truth from live detection
            pred_gt = LandmarkGroundTruth(
                image_path=img_path,
                image_width=img_bgr.shape[1],
                image_height=img_bgr.shape[0],
                landmarks_68=landmarks.landmarks_68,
                hairline_y=hairline_y,
            )

            # 6. Load ground truth
            gt = LandmarkGroundTruth.load(gt_files[stem])
            pairs.append((pred_gt, gt))

        except VisagismError as exc:
            skipped.append(f"Detection failed for {stem}: {exc}")
        except Exception as exc:
            skipped.append(f"Error processing {stem}: {exc}")

    print("Evaluation mode:")
    print(f"  Images directory: {images_dir.resolve()}")
    print(f"  Ground truth directory: {ground_truth_dir.resolve()}")
    print(f"  Images with ground truth: {len(pairs)}")
    print(f"  Skipped: {len(skipped)}")

    evaluator = LandmarkEvaluator()
    report = evaluator.evaluate_pairs(pairs)
    report.skipped_files.extend(skipped)

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
            "  # Evaluate live detections against ground truth\n"
            "  python scripts/landmark_evaluation.py --mode evaluate "
            "--images-dir ./photos/ --ground-truth-dir ./gt/ "
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
        default="data/ground_truth",
        help=(
            "Output directory for ground truth JSON "
            "(label mode, default: data/ground_truth)"
        ),
    )

    # Evaluate mode arguments
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing input images (evaluate mode)",
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
    parser.add_argument(
        "--detector",
        choices=["dlib", "nonml"],
        default="dlib",
        help="Landmark detector to use in evaluate mode (default: dlib)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "label":
            if not args.input:
                parser.error("--input is required for label mode")
            _run_label_mode(args)
        elif args.mode == "evaluate":
            if not args.images_dir or not args.ground_truth_dir:
                parser.error(
                    "--images-dir and --ground-truth-dir are required "
                    "for evaluate mode"
                )
            _run_evaluate_mode(args)
    except VisagismError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
