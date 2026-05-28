"""Standalone diagnostic script comparing morphological operations on images.

This script visualises the effect of morphological operations (especially
closing) on images before Canny edge detection. It produces side-by-side
comparisons to help decide whether to add morphological closing to the
hairline detection pipeline.

The script works without the dlib model — it loads images directly via
OpenCV and operates in headless mode by default.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]

# Map operation names to OpenCV morphological constants
_MORPH_OPS: dict[str, int] = {
    "close": cv2.MORPH_CLOSE,
    "open": cv2.MORPH_OPEN,
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE,
}


def _parse_kernel_sizes(value: str) -> list[int]:
    """Parse a comma-separated string of kernel sizes into a sorted list.

    Parameters
    ----------
    value : str
        Comma-separated kernel sizes (e.g. ``"3,5,7"``).

    Returns
    -------
    list[int]
        Sorted list of unique positive kernel sizes.

    Raises
    ------
    argparse.ArgumentTypeError
        If the string contains invalid values.
    """
    try:
        sizes = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Kernel sizes must be integers, got: {value!r}"
        ) from exc

    if any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError(
            f"Kernel sizes must be positive integers, got: {sizes}"
        )

    return sorted(set(sizes))


def _parse_operations(value: str) -> list[str]:
    """Parse a comma-separated string of operation names.

    Parameters
    ----------
    value : str
        Comma-separated operation names (e.g. ``"close,open"``).

    Returns
    -------
    list[str]
        List of valid operation names in the order provided.

    Raises
    ------
    argparse.ArgumentTypeError
        If an unknown operation is specified.
    """
    ops = [x.strip().lower() for x in value.split(",") if x.strip()]
    invalid = [op for op in ops if op not in _MORPH_OPS]
    if invalid:
        valid = ", ".join(_MORPH_OPS.keys())
        raise argparse.ArgumentTypeError(
            f"Unknown operation(s): {invalid}. Valid: {valid}"
        )
    return ops


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


def _compute_xor_diff(
    edges_original: ImageArray,
    edges_morphed: ImageArray,
) -> ImageArray:
    """Compute an XOR difference map between two edge images.

    Parameters
    ----------
    edges_original : ImageArray
        Edge map of the original image.
    edges_morphed : ImageArray
        Edge map of the morphed image.

    Returns
    -------
    ImageArray
        Binary image where white pixels indicate differing edges.
    """
    return cast(ImageArray, cv2.bitwise_xor(edges_original, edges_morphed))


def _build_morphology_figure(
    img_gray: ImageArray,
    results: list[tuple[str, int, ImageArray]],
    image_name: str,
) -> plt.Figure:
    """Build Figure 1: original vs each morphological operation (grayscale).

    Parameters
    ----------
    img_gray : ImageArray
        Original grayscale image.
    results : list[tuple[str, int, ImageArray]]
        List of (operation, kernel_size, morphed_image) tuples.
    image_name : str
        Name of the input image for the figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure with the comparison grid.
    """
    n_results = len(results)
    n_cols = 3
    n_rows = (n_results + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        f"Morphological Operations — {image_name}",
        fontsize=14,
        fontweight="bold",
    )

    # Original image
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")

    for idx, (op, ksize, morphed) in enumerate(results, start=1):
        axes[idx].imshow(morphed, cmap="gray")
        axes[idx].set_title(f"{op.capitalize()} (k={ksize})")
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_results + 1, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _build_canny_figure(
    edges_original: ImageArray,
    canny_results: list[tuple[str, int, ImageArray]],
    image_name: str,
) -> plt.Figure:
    """Build Figure 2: original Canny vs each operation's Canny.

    Parameters
    ----------
    edges_original : ImageArray
        Canny edge map of the original image.
    canny_results : list[tuple[str, int, ImageArray]]
        List of (operation, kernel_size, edges_morphed) tuples.
    image_name : str
        Name of the input image for the figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure with the Canny comparison grid.
    """
    n_results = len(canny_results)
    n_cols = 3
    n_rows = (n_results + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        f"Canny Edge Detection — {image_name}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(edges_original, cmap="gray")
    axes[0].set_title("Original Canny")
    axes[0].axis("off")

    for idx, (op, ksize, edges) in enumerate(canny_results, start=1):
        axes[idx].imshow(edges, cmap="gray")
        axes[idx].set_title(f"{op.capitalize()} + Canny (k={ksize})")
        axes[idx].axis("off")

    for idx in range(n_results + 1, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _build_diff_figure(
    edges_original: ImageArray,
    diff_results: list[tuple[str, int, ImageArray]],
    image_name: str,
) -> plt.Figure:
    """Build Figure 3: XOR difference maps showing where edges differ.

    Parameters
    ----------
    edges_original : ImageArray
        Canny edge map of the original image (used for pixel counts).
    diff_results : list[tuple[str, int, ImageArray]]
        List of (operation, kernel_size, diff_map) tuples.
    image_name : str
        Name of the input image for the figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure with the difference map grid.
    """
    n_results = len(diff_results)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        f"Edge Difference Maps (XOR) — {image_name}",
        fontsize=14,
        fontweight="bold",
    )

    original_pixels = int(np.count_nonzero(edges_original))

    for idx, (op, ksize, diff) in enumerate(diff_results):
        axes[idx].imshow(diff, cmap="gray")
        diff_pixels = int(np.count_nonzero(diff))
        pct = (
            (diff_pixels / original_pixels * 100)
            if original_pixels > 0
            else 0.0
        )
        axes[idx].set_title(
            f"{op.capitalize()} (k={ksize})\n"
            f"Diff pixels: {diff_pixels} ({pct:.1f}% of original)"
        )
        axes[idx].axis("off")

    for idx in range(n_results, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_path : Path
        Destination file path.
    """
    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        warnings.warn(f"Could not save figure to {output_path}: {exc}")
    finally:
        plt.close(fig)


def _print_summary(
    image_path: Path,
    img_shape: tuple[int, ...],
    operations: list[str],
    kernel_sizes: list[int],
    with_canny: bool,
    canny_low: int,
    canny_high: int,
    output_dir: Path,
    saved_paths: list[Path],
) -> None:
    """Print a human-readable summary of the comparison to the console.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    img_shape : tuple[int, ...]
        Shape of the loaded image.
    operations : list[str]
        List of operations performed.
    kernel_sizes : list[int]
        List of kernel sizes used.
    with_canny : bool
        Whether Canny edge detection was included.
    canny_low : int
        Lower Canny threshold.
    canny_high : int
        Upper Canny threshold.
    output_dir : Path
        Directory where outputs were saved.
    saved_paths : list[Path]
        List of saved figure file paths.
    """
    print(f"\n{'=' * 60}")
    print("Morphological Operation Comparison")
    print(f"{'=' * 60}")
    print(f"  Image: {image_path.name}")
    print(f"  Dimensions: {img_shape[1]}x{img_shape[0]} px")
    print(f"  Operations: {', '.join(operations)}")
    print(f"  Kernel sizes: {', '.join(str(k) for k in kernel_sizes)}")
    print(f"  Canny: {'enabled' if with_canny else 'disabled'}")
    if with_canny:
        print(f"  Canny thresholds: low={canny_low}, high={canny_high}")
    print(f"  Output directory: {output_dir.resolve()}")
    print("\n  Saved figures:")
    for path in saved_paths:
        print(f"    - {path.name}")
    print(f"{'=' * 60}\n")


def process_image(
    image_path: Path,
    output_dir: Path,
    operations: list[str],
    kernel_sizes: list[int],
    with_canny: bool,
    canny_low: int,
    canny_high: int,
    visualize: bool,
) -> None:
    """Load an image and generate all comparison figures.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    output_dir : Path
        Directory where figures will be saved.
    operations : list[str]
        List of morphological operation names.
    kernel_sizes : list[int]
        List of kernel sizes.
    with_canny : bool
        Whether to include Canny edge detection comparisons.
    canny_low : int
        Lower Canny threshold.
    canny_high : int
        Upper Canny threshold.
    visualize : bool
        Whether to display the saved figures using OpenCV windows.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be loaded.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read image file (may be corrupted): {image_path}")

    img_gray = cast(ImageArray, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply all operation + kernel combinations
    morph_results: list[tuple[str, int, ImageArray]] = []
    canny_results: list[tuple[str, int, ImageArray]] = []
    diff_results: list[tuple[str, int, ImageArray]] = []

    edges_original: ImageArray | None = None
    if with_canny:
        edges_original = _run_canny(img_gray, canny_low, canny_high)

    for op in operations:
        for ksize in kernel_sizes:
            morphed = _apply_morphology(img_gray, op, ksize)
            morph_results.append((op, ksize, morphed))

            if with_canny and edges_original is not None:
                edges_morphed = _run_canny(morphed, canny_low, canny_high)
                canny_results.append((op, ksize, edges_morphed))
                diff = _compute_xor_diff(edges_original, edges_morphed)
                diff_results.append((op, ksize, diff))

    saved_paths: list[Path] = []

    # Figure 1: Morphology comparison
    fig1 = _build_morphology_figure(img_gray, morph_results, image_path.name)
    path1 = output_dir / f"{image_path.stem}_morphology.png"
    _save_figure(fig1, path1)
    saved_paths.append(path1)

    # Figure 2: Canny comparison
    if with_canny and edges_original is not None:
        fig2 = _build_canny_figure(edges_original, canny_results, image_path.name)
        path2 = output_dir / f"{image_path.stem}_canny.png"
        _save_figure(fig2, path2)
        saved_paths.append(path2)

        # Figure 3: Difference maps
        fig3 = _build_diff_figure(edges_original, diff_results, image_path.name)
        path3 = output_dir / f"{image_path.stem}_diff.png"
        _save_figure(fig3, path3)
        saved_paths.append(path3)

    _print_summary(
        image_path=image_path,
        img_shape=img_gray.shape,
        operations=operations,
        kernel_sizes=kernel_sizes,
        with_canny=with_canny,
        canny_low=canny_low,
        canny_high=canny_high,
        output_dir=output_dir,
        saved_paths=saved_paths,
    )

    if visualize:
        # Show original grayscale
        window_name = "Original Grayscale"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img_gray)
        print(f"Showing {window_name} — press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        if with_canny and edges_original is not None:
            # Show original Canny
            window_name = "Original Canny"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, edges_original)
            print(f"Showing {window_name} — press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

            # Show each operation's Canny result
            for op, ksize, edges in canny_results:
                window_name = f"{op.capitalize()} k={ksize} Canny"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, edges)
                print(f"Showing {window_name} — press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
        else:
            # Show each operation's morphed result (when no --canny)
            for op, ksize, morphed in morph_results:
                window_name = f"{op.capitalize()} k={ksize}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, morphed)
                print(f"Showing {window_name} — press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

        # Show saved comparison PNGs
        for path in saved_paths:
            img = cv2.imread(str(path))
            if img is not None:
                window_name = f"Comparison - {path.name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, img)
                print(f"Showing {path.name} — press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

        cv2.destroyAllWindows()


def main() -> None:
    """Parse arguments and run the morphological comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare morphological operations on images before Canny edge "
            "detection. Saves comparison figures to disk."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input image (JPG or PNG).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/morphology_comparison"),
        help=(
            "Output directory for comparison figures "
            "(default: output/morphology_comparison/)"
        ),
    )
    parser.add_argument(
        "--kernel-sizes",
        type=_parse_kernel_sizes,
        default=_parse_kernel_sizes("3,5,7"),
        help="Comma-separated list of kernel sizes (default: 3,5,7)",
    )
    parser.add_argument(
        "--operations",
        type=_parse_operations,
        default=_parse_operations("close,open,dilate,erode"),
        help=(
            "Comma-separated list of operations: close,open,dilate,erode "
            "(default: close,open,dilate,erode)"
        ),
    )
    parser.add_argument(
        "--canny",
        action="store_true",
        help="Also run Canny edge detection and produce edge-comparison figures.",
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
        "--visualize",
        action="store_true",
        help="Display saved comparison figures using OpenCV windows.",
    )
    args = parser.parse_args()

    try:
        process_image(
            image_path=args.input,
            output_dir=args.output,
            operations=args.operations,
            kernel_sizes=args.kernel_sizes,
            with_canny=args.canny,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            visualize=args.visualize,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
