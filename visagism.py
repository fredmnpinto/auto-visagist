#!/usr/bin/env python3
"""Facial Visagism Analysis System — CLI entry point.

Usage::

    python visagism.py --input <path> [--visualize] [--save-viz]

Examples::

    python visagism.py --input photo.jpg
    python visagism.py --input photo.jpg --visualize
    python visagism.py --input photo.jpg --save-viz --output ./results
    python visagism.py --input photo.jpg --model path/to/model.dat --visualize
"""

from __future__ import annotations

import dataclasses
import sys

from visagism.cli import CliParser
from visagism.errors import VisagismError
from visagism.face_detector import FaceDetector
from visagism.hairline_detector import HairlineDetector
from visagism.image_loader import ImageLoader
from visagism.landmark_detector import LandmarkDetector
from visagism.landmark_visualizer import LandmarkVisualizer
from visagism.model_finder import ModelFinder
from visagism.report_formatter import ReportFormatter
from visagism.visagism_calculator import VisagismCalculator


def main() -> None:
    """Run the Facial Visagism Analysis pipeline.

    Parses command-line arguments, loads the image, detects a face,
    detects landmarks, estimates hairline, and optionally visualizes.
    """
    try:
        config = CliParser.parse()
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        img_bgr, img_gray = ImageLoader.load(config.input_path)
        print(
            f"Loaded image: {config.input_path} "
            f"({img_bgr.shape[1]}x{img_bgr.shape[0]})"
        )

        # Detect face
        detector = FaceDetector()
        face_rect = detector.detect(img_gray, img_bgr.shape[:2])
        print(
            f"Face detected at ({face_rect[0]}, {face_rect[1]}) "
            f"size {face_rect[2]}x{face_rect[3]}"
        )

        # Detect landmarks (always runs)
        model_path = ModelFinder.find(config.model_path)
        landmark_detector = LandmarkDetector(model_path)
        landmarks = landmark_detector.detect(
            img_gray, face_rect, config.input_path
        )
        print(f"Detected {len(landmarks.landmarks_68)} facial landmarks")

        # Check for non-frontal pose
        if not LandmarkDetector.check_pose(landmarks.landmarks_68):
            print(
                "Warning: Image appears non-frontal. "
                "For best results, use a frontal photo."
            )

        # Detect hairline (always runs)
        hairline_detector = HairlineDetector()
        result = hairline_detector.detect(
            img_gray, landmarks,
            canny_low=config.canny_low,
            canny_high=config.canny_high,
            close_ksize=config.kernel_size,
        )
        hairline_y = result["hairline_y"]
        method = result["method"]
        landmarks = dataclasses.replace(landmarks, hairline_y=hairline_y)
        print(f"Estimated hairline at y={hairline_y} (method={method}, "
              f"close_k={config.kernel_size}, canny={config.canny_low}/{config.canny_high})")

        # Run visagism analysis
        calculator = VisagismCalculator.from_landmarks(landmarks)
        analysis = calculator.calculate()

        if analysis.measurements.hairline_fallback_used:
            print("Warning: Hairline not detected. Upper third estimated from middle third (may reduce accuracy).")

        # Console output
        console_output = ReportFormatter.format_console(analysis)
        print(console_output)

        # Save report
        report_path = ReportFormatter.save_report(
            analysis, config.output_dir, config.input_path.stem,
            fallback_used=analysis.measurements.hairline_fallback_used
        )
        print(f"Saved analysis report: {report_path}")

        # Visualize if requested
        if config.visualize or config.save_viz:
            viz = LandmarkVisualizer()
            annotated = viz.draw_landmarks(img_bgr, landmarks)

            if config.save_viz:
                save_path = viz.save(
                    annotated, config.output_dir, config.input_path.stem
                )
                print(f"Saved visualization: {save_path}")

            if config.visualize:
                viz.show(annotated)

        print("Done.")

    except VisagismError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
