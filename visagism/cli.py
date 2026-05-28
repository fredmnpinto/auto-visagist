"""Command-line argument parsing for the Facial Visagism Analysis System."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CliConfig:
    """Configuration parsed from command-line arguments.

    Parameters
    ----------
    input_path : Path
        Path to the input image file.
    output_dir : Path
        Directory for output files (default: ``output``).
    model_path : Path or None
        Explicit path to the dlib model file (optional).
    visualize : bool
        Whether to display the visualization window.
    save_viz : bool
        Whether to save the visualization to a file.
    kernel_size : int
        Morphological closing kernel size for hairline detection.
    canny_low : int
        Lower Canny threshold for hairline detection.
    canny_high : int
        Upper Canny threshold for hairline detection.
    """

    input_path: Path
    output_dir: Path = field(default_factory=lambda: Path("output"))
    model_path: Path | None = None
    visualize: bool = False
    save_viz: bool = False
    kernel_size: int = 7
    canny_low: int = 30
    canny_high: int = 60


class CliParser:
    """Parses command-line arguments into a ``CliConfig`` dataclass."""

    @staticmethod
    def parse(argv: list[str] | None = None) -> CliConfig:
        """Parse command-line arguments.

        Parameters
        ----------
        argv : list of str or None
            Argument list to parse. If None, uses ``sys.argv``.

        Returns
        -------
        CliConfig
            Parsed configuration.
        """
        parser = argparse.ArgumentParser(
            prog="visagism.py",
            description=(
                "Facial Visagism Analysis System — analyzes frontal face "
                "photographs to detect facial landmarks, calculate proportions, "
                "classify face shape, and compare against the golden ratio."
            ),
            epilog=(
                "Examples:\n"
                "  python visagism.py --input photo.jpg --visualize\n"
                "  python visagism.py --input photo.jpg --save-viz --output ./results\n"
                "  python visagism.py --input photo.jpg --kernel-size 5 --canny-low 20 --canny-high 60"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--input", "-i",
            required=True,
            type=str,
            help="Path to input image file (JPG or PNG)",
        )
        parser.add_argument(
            "--output", "-o",
            default="output",
            type=str,
            help="Output directory for results (default: output)",
        )
        parser.add_argument(
            "--model", "-m",
            type=str,
            default=None,
            help="Path to dlib shape predictor model file",
        )
        parser.add_argument(
            "--visualize",
            action="store_true",
            help="Display landmark visualization window",
        )
        parser.add_argument(
            "--save-viz",
            action="store_true",
            help="Save landmark visualization to output file",
        )
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=7,
            help="Morphological closing kernel size for hairline detection (default: 7, 0=disable)",
        )
        parser.add_argument(
            "--canny-low",
            type=int,
            default=30,
            help="Lower Canny threshold for hairline detection (default: 30)",
        )
        parser.add_argument(
            "--canny-high",
            type=int,
            default=60,
            help="Upper Canny threshold for hairline detection (default: 60)",
        )

        args = parser.parse_args(argv)

        return CliConfig(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            model_path=Path(args.model) if args.model else None,
            visualize=args.visualize,
            save_viz=args.save_viz,
            kernel_size=args.kernel_size,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
