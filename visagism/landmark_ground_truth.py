"""Ground truth data model for facial landmark annotation.

Provides a dataclass for storing manually-annotated 68-point facial
landmarks plus an optional hairline position, along with JSON
serialization and deserialization utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visagism.constants import REGION_INDICES
from visagism.errors import GroundTruthError
from visagism.types import LandmarksList


@dataclass
class LandmarkGroundTruth:
    """Container for manually-annotated facial landmark ground truth.

    Parameters
    ----------
    image_path : Path
        Path to the input image file.
    image_width : int
        Width of the image in pixels.
    image_height : int
        Height of the image in pixels.
    landmarks_68 : LandmarksList
        List of 68 (x, y) landmark coordinates. Missing landmarks are
        stored as ``(-1, -1)``.
    hairline_y : int or None
        Optional y-coordinate of the hairline. ``None`` if not labeled.
    corrected_landmarks : list of int
        Indices of landmarks that were manually corrected by the user
        (as opposed to left as dlib predictions).
    timestamp : str
        ISO-8601 timestamp of when the annotation was created or last
        modified.
    """

    image_path: Path
    image_width: int
    image_height: int
    landmarks_68: LandmarksList = field(default_factory=list)
    hairline_y: int | None = None
    corrected_landmarks: list[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(
        timezone.utc
    ).isoformat())

    def __post_init__(self) -> None:
        """Validate landmark count after initialization."""
        if len(self.landmarks_68) != 68:
            raise GroundTruthError(
                f"Expected exactly 68 landmarks, got {len(self.landmarks_68)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the ground truth to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with JSON-serializable values.
        """
        return {
            "image_path": str(self.image_path),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "landmarks_68": self.landmarks_68,
            "hairline_y": self.hairline_y,
            "corrected_landmarks": self.corrected_landmarks,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the ground truth to a JSON string.

        Parameters
        ----------
        indent : int
            Indentation level for pretty-printing.

        Returns
        -------
        str
            JSON representation of the ground truth.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save the ground truth to a JSON file.

        Parameters
        ----------
        path : Path
            Destination file path.

        Raises
        ------
        GroundTruthError
            If the file cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        except OSError as exc:
            raise GroundTruthError(
                f"Failed to save ground truth to {path}: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LandmarkGroundTruth:
        """Create a ``LandmarkGroundTruth`` instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the ground truth fields.

        Returns
        -------
        LandmarkGroundTruth
            Parsed ground truth instance.

        Raises
        ------
        GroundTruthError
            If required fields are missing or invalid.
        """
        required = {"image_path", "image_width", "image_height"}
        missing = required - set(data.keys())
        if missing:
            raise GroundTruthError(
                f"Missing required fields: {sorted(missing)}"
            )

        landmarks = data.get("landmarks_68", [])
        if not isinstance(landmarks, list):
            raise GroundTruthError("landmarks_68 must be a list")

        # Pad or trim to exactly 68 landmarks
        if len(landmarks) < 68:
            landmarks = landmarks + [[-1, -1]] * (68 - len(landmarks))
        elif len(landmarks) > 68:
            landmarks = landmarks[:68]

        # Ensure each landmark is a tuple of ints
        parsed_landmarks: LandmarksList = []
        for pt in landmarks:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                parsed_landmarks.append((int(pt[0]), int(pt[1])))
            else:
                parsed_landmarks.append((-1, -1))

        hairline = data.get("hairline_y")
        if hairline is not None:
            hairline = int(hairline)

        corrected = data.get("corrected_landmarks", [])
        if not isinstance(corrected, list):
            corrected = []
        parsed_corrected = [int(i) for i in corrected if isinstance(i, int)]

        return cls(
            image_path=Path(data["image_path"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            landmarks_68=parsed_landmarks,
            hairline_y=hairline,
            corrected_landmarks=parsed_corrected,
            timestamp=data.get("timestamp", datetime.now(
                timezone.utc
            ).isoformat()),
        )

    @classmethod
    def from_json(cls, json_str: str) -> LandmarkGroundTruth:
        """Parse a JSON string into a ``LandmarkGroundTruth`` instance.

        Parameters
        ----------
        json_str : str
            JSON string to parse.

        Returns
        -------
        LandmarkGroundTruth
            Parsed ground truth instance.

        Raises
        ------
        GroundTruthError
            If the JSON is malformed or missing required fields.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise GroundTruthError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path) -> LandmarkGroundTruth:
        """Load a ground truth from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Returns
        -------
        LandmarkGroundTruth
            Loaded ground truth instance.

        Raises
        ------
        GroundTruthError
            If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            raise GroundTruthError(
                f"Failed to load ground truth from {path}: {exc}"
            ) from exc
        return cls.from_json(content)

    def get_region_for_landmark(self, index: int) -> str:
        """Return the region name for a given landmark index.

        Parameters
        ----------
        index : int
            0-indexed landmark index (0-67).

        Returns
        -------
        str
            Region name (e.g., ``"jaw"``, ``"left_eye"``), or
            ``"unknown"`` if the index is out of range.
        """
        for region, indices in REGION_INDICES.items():
            if index in indices:
                return region
        return "unknown"

    def is_landmark_placed(self, index: int) -> bool:
        """Check whether a landmark has been placed (not ``(-1, -1)``).

        Parameters
        ----------
        index : int
            0-indexed landmark index.

        Returns
        -------
        bool
            ``True`` if the landmark has a valid position.
        """
        if not (0 <= index < 68):
            return False
        pt = self.landmarks_68[index]
        return pt != (-1, -1)

    def completion_count(self) -> tuple[int, int]:
        """Return the number of placed landmarks and total.

        Returns
        -------
        tuple
            ``(placed_count, total_count)`` where total is always 68.
        """
        placed = sum(1 for i in range(68) if self.is_landmark_placed(i))
        return placed, 68

    def default_output_path(self, output_dir: Path | None = None) -> Path:
        """Generate the default output path for this ground truth.

        The file is saved as ``<image_stem>_gt.json`` in the
        ``data/ground_truth/`` directory (or the specified output_dir).

        Parameters
        ----------
        output_dir : Path or None
            Custom output directory. If None, uses ``data/ground_truth/``
            relative to the image's parent directory.

        Returns
        -------
        Path
            Default file path for saving this ground truth.
        """
        if output_dir is None:
            output_dir = self.image_path.parent / "data" / "ground_truth"
        return output_dir / f"{self.image_path.stem}_gt.json"
