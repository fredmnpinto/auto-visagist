"""Custom exception hierarchy for the Facial Visagism Analysis System.

All domain-specific exceptions inherit from VisagismError.
The CLI entry point catches VisagismError and prints user-friendly messages
without exposing raw Python tracebacks.
"""

from __future__ import annotations


class VisagismError(Exception):
    """Base exception for all visagism analysis errors.

    Parameters
    ----------
    message : str or None
        Custom error message. If None, uses the class default MESSAGE.
    """

    MESSAGE: str = "An unknown visagism error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.MESSAGE)


class ImageError(VisagismError):
    """Errors related to image loading and validation."""

    MESSAGE: str = "An error occurred while processing the image."


class UnsupportedFormatError(ImageError):
    """Raised when the input file format is not supported."""

    MESSAGE: str = "Unsupported image format. Use JPG or PNG."


class CorruptedImageError(ImageError):
    """Raised when the image file cannot be read or is corrupted."""

    MESSAGE: str = "Cannot read image file. File may be corrupted."


class DetectionError(VisagismError):
    """Errors related to face or landmark detection."""

    MESSAGE: str = "Detection error occurred."


class NoFaceDetectedError(DetectionError):
    """Raised when no face is detected in the input image."""

    MESSAGE: str = (
        "No face detected. Please provide a clear frontal photo."
    )


class AnalysisError(VisagismError):
    """Errors related to facial analysis calculations."""

    MESSAGE: str = "Analysis calculation error occurred."


class InvalidLandmarksError(AnalysisError):
    """Raised when landmark data is invalid or incomplete."""

    MESSAGE: str = "Invalid landmark data for analysis."


class ModelError(VisagismError):
    """Errors related to model files."""

    MESSAGE: str = "Model error occurred."


class ModelNotFoundError(ModelError):
    """Raised when the dlib shape predictor model file is not found."""

    MESSAGE: str = (
        "dlib shape predictor model not found. "
        "Download and place the file at:\n"
        "  data/shape_predictor_68_face_landmarks.dat\n"
        "or set the DLIB_MODEL_PATH environment variable.\n"
        "Download URL:\n"
        "  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
        "Then decompress: bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"
    )


class LandmarkToolError(VisagismError):
    """Base exception for landmark evaluation tool errors."""

    MESSAGE: str = "Landmark evaluation tool error occurred."


class LabelerError(LandmarkToolError):
    """Errors related to the interactive landmark labeling GUI."""

    MESSAGE: str = "Landmark labeling error occurred."


class EvaluationError(LandmarkToolError):
    """Errors related to landmark evaluation and comparison."""

    MESSAGE: str = "Landmark evaluation error occurred."


class GroundTruthError(LandmarkToolError):
    """Errors related to ground truth data loading and validation."""

    MESSAGE: str = "Ground truth data error occurred."
