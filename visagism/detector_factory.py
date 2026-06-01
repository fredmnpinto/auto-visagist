"""Factory for creating landmark detectors.

Provides a single entry point for instantiating either the dlib-based
``LandmarkDetector`` or the classical-CV ``NonMLLandmarkDetector``.
"""

from __future__ import annotations

from pathlib import Path

from visagism.errors import DetectionError
from visagism.landmark_detector import LandmarkDetector
from visagism.nonml_landmark_detector import NonMLLandmarkDetector


SUPPORTED_DETECTORS: frozenset[str] = frozenset({"dlib", "nonml"})


def create_landmark_detector(
    detector_type: str,
    model_path: Path,
) -> LandmarkDetector | NonMLLandmarkDetector:
    """Create a landmark detector of the specified type.

    Parameters
    ----------
    detector_type : str
        Detector type.  Must be ``"dlib"`` or ``"nonml"``.
    model_path : Path
        Path to the dlib shape predictor model file (required for
        ``"dlib"``, ignored for ``"nonml"`` but must still be provided).

    Returns
    -------
    LandmarkDetector or NonMLLandmarkDetector
        Instance of the requested detector.

    Raises
    ------
    DetectionError
        If *detector_type* is not supported.
    """
    if detector_type == "dlib":
        return LandmarkDetector(model_path)
    if detector_type == "nonml":
        return NonMLLandmarkDetector(model_path)

    raise DetectionError(
        f"Unsupported detector type: '{detector_type}'. "
        f"Use one of: {', '.join(sorted(SUPPORTED_DETECTORS))}."
    )
