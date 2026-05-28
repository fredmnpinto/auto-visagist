"""Batch Canny edge detection viewer with morphological preprocessing.

This script processes all images in a directory, applies a single morphological
operation with a specified kernel size, runs Canny edge detection, and either
displays the result in an OpenCV window or saves it directly to disk.

When detectors are available, it also detects facial landmarks and estimates
the hairline position, overlaying the result on the original image.

Example usage
-------------
View close k=5 Canny for all test images with hairline overlay::

    python scripts/batch_canny_viewer.py --input test_images/ \\
        --operation close --kernel-size 5 --visualize

View open k=3 with custom Canny thresholds::

    python scripts/batch_canny_viewer.py --input test_images/ \\
        --operation open --kernel-size 3 \\
        --canny-low 20 --canny-high 60 --visualize

Just save without viewing::

    python scripts/batch_canny_viewer.py --input test_images/ \\
        --operation close --kernel-size 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from visagism.face_detector import FaceDetector  # noqa: E402
from visagism.hairline_detector import HairlineDetector  # noqa: E402
from visagism.landmark_detector import LandmarkDetector  # noqa: E402
from visagism.model_finder import ModelFinder  # noqa: E402

ImageArray = npt.NDArray[np.uint8]

# Supported image extensions
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# Map operation names to OpenCV morphological constants
_MORPH_OPS: dict[str, int] = {
    "close": cv2.MORPH_CLOSE,
    "open": cv2.MORPH_OPEN,
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE,
}


def _parse_operation(value: str) -> str:
    """Validate a single morphological operation name.

    Parameters
    ----------
    value : str
        Operation name to validate.

    Returns
    -------
    str
        Lowercase operation name.

    Raises
    ------
    argparse.ArgumentTypeError
        If the operation is not recognised.
    """
    op = value.strip().lower()
    if op not in _MORPH_OPS:
        valid = ", ".join(_MORPH_OPS.keys())
        raise argparse.ArgumentTypeError(
            f"Unknown operation: {value!r}. Valid: {valid}"
        )
    return op


def _parse_kernel_size(value: str) -> int:
    """Parse and validate a single kernel size.

    Parameters
    ----------
    value : str
        Kernel size as a string.

    Returns
    -------
    int
        Positive integer kernel size.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a positive integer.
    """
    try:
        ksize = int(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Kernel size must be an integer, got: {value!r}"
        ) from exc

    if ksize <= 0:
        raise argparse.ArgumentTypeError(
            f"Kernel size must be a positive integer, got: {ksize}"
        )
    return ksize


def _find_images(input_path: Path) -> list[Path]:
    """Find all supported image files in a directory.

    Parameters
    ----------
    input_path : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")

    images: list[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def _apply_morphology(
    img_gray: ImageArray,
    operation: str,
    kernel_size: int,
) -> ImageArray:
    """Apply a single morphological operation to a grayscale image.

    Parameters
    ----------
    img_gray : ImageArray
        Input grayscale image.
    operation : str
        Name of the morphological operation (close, open, dilate, erode).
    kernel_size : int
        Size of the square structuring element.

    Returns
    -------
    ImageArray
        Resulting grayscale image after the operation.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    op_code = _MORPH_OPS[operation]
    return cast(ImageArray, cv2.morphologyEx(img_gray, op_code, kernel))


def _run_canny(
    img_gray: ImageArray,
    low: int,
    high: int,
) -> ImageArray:
    """Run Canny edge detection on a grayscale image.

    Parameters
    ----------
    img_gray : ImageArray
        Input grayscale image.
    low : int
        Lower threshold for Canny hysteresis.
    high : int
        Upper threshold for Canny hysteresis.

    Returns
    -------
    ImageArray
        Binary edge map (0 or 255).
    """
    return cast(ImageArray, cv2.Canny(img_gray, low, high))


def _resize_to_height(img: ImageArray, target_height: int) -> ImageArray:
    """Resize an image to a target height, preserving aspect ratio.

    Parameters
    ----------
    img : ImageArray
        Input image.
    target_height : int
        Desired height in pixels.

    Returns
    -------
    ImageArray
        Resized image.
    """
    h, w = img.shape[:2]
    if h == target_height:
        return img
    scale = target_height / h
    new_w = int(w * scale)
    return cast(ImageArray, cv2.resize(img, (new_w, target_height)))


def _process_image(
    image_path: Path,
    operation: str,
    kernel_size: int,
    canny_low: int,
    canny_high: int,
    output_dir: Path,
    visualize: bool,
    face_detector: FaceDetector | None = None,
    landmark_detector: LandmarkDetector | None = None,
    hairline_detector: HairlineDetector | None = None,
) -> Path | None:
    """Process a single image and return the saved output path.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    operation : str
        Morphological operation name.
    kernel_size : int
        Kernel size for the morphological operation.
    canny_low : int
        Lower Canny threshold.
    canny_high : int
        Upper Canny threshold.
    output_dir : Path
        Directory where the Canny result will be saved.
    visualize : bool
        Whether to display the result in an OpenCV window.
    face_detector : FaceDetector or None
        Face detector instance. If None, hairline overlay is skipped.
    landmark_detector : LandmarkDetector or None
        Landmark detector instance. If None, hairline overlay is skipped.
    hairline_detector : HairlineDetector or None
        Hairline detector instance. If None, hairline overlay is skipped.

    Returns
    -------
    Path or None
        Path to the saved output image, or None if the image could not be
        processed.
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        warnings.warn(
            f"Cannot read image file (may be corrupted): {image_path}"
        )
        return None

    img_gray = cast(ImageArray, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    morphed = _apply_morphology(img_gray, operation, kernel_size)
    edges = _run_canny(morphed, canny_low, canny_high)

    output_path = output_dir / f"{image_path.stem}_canny.png"
    try:
        cv2.imwrite(str(output_path), edges)
    except OSError as exc:
        warnings.warn(f"Could not save {output_path}: {exc}")
        return None

    # Hairline detection (optional, for visualization only)
    hairline_y: int | None = None
    method = "none"

    if (
        face_detector is not None
        and landmark_detector is not None
        and hairline_detector is not None
    ):
        try:
            faces = face_detector.detect(img_gray, img_gray.shape)
            if not faces:
                warnings.warn(f"No face detected in {image_path}")
                return output_path

            # Use largest face (FaceDetector already returns single largest)
            face_rect = faces
            landmarks = landmark_detector.detect(
                img_gray, face_rect, image_path
            )
            result = hairline_detector.detect(
                morphed, landmarks,
                canny_low=canny_low,
                canny_high=canny_high,
                close_ksize=0,  # Already closed by batch viewer
            )
            hairline_y = result["hairline_y"]
            method = result["method"]
        except Exception as exc:
            warnings.warn(
                f"Hairline detection failed for {image_path}: {exc}"
            )

    if visualize:
        # Panel 1: Original with hairline overlay
        img_display = img_bgr.copy()
        if hairline_y is not None:
            cv2.line(
                img_display,
                (0, hairline_y),
                (img_display.shape[1], hairline_y),
                (0, 0, 255),  # Red
                2,
            )
        label = f"Hairline: y={hairline_y} ({method})"
        cv2.putText(
            img_display, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # Panel 2: Morphed image (post-operation)
        morphed_display = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            morphed_display, f"{operation.capitalize()} k={kernel_size}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # Panel 3: Full Canny
        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if hairline_y is not None:
            cv2.line(
                edges_display,
                (0, hairline_y),
                (edges_display.shape[1], hairline_y),
                (0, 0, 255),  # Red
                2,
            )
        cv2.putText(
            edges_display, "Full Canny", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # Resize panels to same height for horizontal stacking
        target_h = img_display.shape[0]
        morphed_display = _resize_to_height(morphed_display, target_h)
        edges_display = _resize_to_height(edges_display, target_h)

        # Stack horizontally
        combined = np.hstack([img_display, morphed_display, edges_display])

        window_name = (
            f"{image_path.name} — {operation.capitalize()} "
            f"k={kernel_size}"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, combined)
        print(
            f"Showing {image_path.name} — press any key for next image..."
        )
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return output_path


def _print_summary(
    processed_count: int,
    operation: str,
    kernel_size: int,
    canny_low: int,
    canny_high: int,
    saved_paths: list[Path],
) -> None:
    """Print a human-readable summary of the batch processing.

    Parameters
    ----------
    processed_count : int
        Number of images successfully processed.
    operation : str
        Morphological operation used.
    kernel_size : int
        Kernel size used.
    canny_low : int
        Lower Canny threshold.
    canny_high : int
        Upper Canny threshold.
    saved_paths : list[Path]
        List of saved output file paths.
    """
    print(f"\n{'=' * 60}")
    print("Batch Canny Edge Detection Summary")
    print(f"{'=' * 60}")
    print(f"  Images processed: {processed_count}")
    print(f"  Operation: {operation}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Canny thresholds: low={canny_low}, high={canny_high}")
    print(f"  Output directory: {saved_paths[0].parent.resolve()}")
    print("\n  Saved files:")
    for path in saved_paths:
        print(f"    - {path.name}")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Parse arguments and run the batch Canny viewer."""
    parser = argparse.ArgumentParser(
        description=(
            "Process all images in a directory with a morphological operation "
            "followed by Canny edge detection. Save results to disk and "
            "optionally display them in OpenCV windows."
        ),
    )
    parser.add_argument(
        "--operation",
        type=_parse_operation,
        required=True,
        help="Morphological operation: close, open, dilate, erode",
    )
    parser.add_argument(
        "--kernel-size",
        type=_parse_kernel_size,
        required=True,
        help="Size of the square structuring element (positive integer)",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=30,
        help="Lower threshold for Canny edge detection (default: 30)",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=100,
        help="Upper threshold for Canny edge detection (default: 100)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display Canny results in resizable OpenCV windows.",
    )
    args = parser.parse_args()

    try:
        image_paths = _find_images(args.input)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not image_paths:
        print(f"No images found in {args.input}")
        sys.exit(0)

    output_dir = Path("output") / "batch_canny"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detectors
    face_detector: FaceDetector | None = None
    landmark_detector: LandmarkDetector | None = None
    hairline_detector: HairlineDetector | None = None

    try:
        model_path = ModelFinder.find()
        face_detector = FaceDetector()
        landmark_detector = LandmarkDetector(model_path)
        hairline_detector = HairlineDetector()
    except Exception as exc:
        print(
            f"WARNING: Could not initialize detectors: {exc}\n"
            f"Hairline overlay will be disabled. "
            f"Canny processing will continue normally."
        )

    saved_paths: list[Path] = []
    skipped: list[str] = []

    for image_path in image_paths:
        result = _process_image(
            image_path=image_path,
            operation=args.operation,
            kernel_size=args.kernel_size,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            output_dir=output_dir,
            visualize=args.visualize,
            face_detector=face_detector,
            landmark_detector=landmark_detector,
            hairline_detector=hairline_detector,
        )
        if result is not None:
            saved_paths.append(result)
        else:
            skipped.append(image_path.name)

    if saved_paths:
        _print_summary(
            processed_count=len(saved_paths),
            operation=args.operation,
            kernel_size=args.kernel_size,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            saved_paths=saved_paths,
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name in skipped:
            print(f"  - {name}")

    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
