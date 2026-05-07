"""Locate the dlib shape predictor model file on disk.

Searches in a predefined order:
1. CLI --model argument (if provided)
2. DLIB_MODEL_PATH environment variable
3. ./data/shape_predictor_68_face_landmarks.dat (relative to CWD)
4. ~/.dlib/shape_predictor_68_face_landmarks.dat
"""

from __future__ import annotations

import os
from pathlib import Path

from visagism.constants import DEFAULT_MODEL_RELATIVE_PATH, ENV_DLIB_MODEL_PATH
from visagism.errors import ModelNotFoundError


class ModelFinder:
    """Locates the dlib shape predictor model file on disk.

    Searches candidate locations in a predefined priority order.
    """

    @staticmethod
    def find(model_override: Path | None = None) -> Path:
        """Locate the dlib shape predictor model file.

        Searches in this order:
        1. *model_override* if provided
        2. ``DLIB_MODEL_PATH`` environment variable
        3. ``./data/shape_predictor_68_face_landmarks.dat``
        4. ``~/.dlib/shape_predictor_68_face_landmarks.dat``

        Parameters
        ----------
        model_override : Path or None
            Explicit path to the model file, typically from CLI ``--model``.

        Returns
        -------
        Path
            Absolute path to the existing model file.

        Raises
        ------
        ModelNotFoundError
            If the model file cannot be found in any search location.
        """
        candidates: list[Path] = []

        # 1. CLI override
        if model_override is not None:
            candidates.append(model_override.resolve())

        # 2. Environment variable
        env_path = os.environ.get(ENV_DLIB_MODEL_PATH)
        if env_path is not None:
            candidates.append(Path(env_path).resolve())

        # 3. ./data/shape_predictor_68_face_landmarks.dat (relative to CWD)
        candidates.append(Path.cwd() / DEFAULT_MODEL_RELATIVE_PATH)

        # 4. ~/.dlib/shape_predictor_68_face_landmarks.dat
        candidates.append(
            Path.home() / ".dlib" / "shape_predictor_68_face_landmarks.dat"
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        raise ModelNotFoundError(
            f"dlib shape predictor model not found in any search location.\n"
            f"Tried:\n"
            f"  {chr(10).join(f'  - {p}' for p in candidates)}\n"
            f"Download URL:\n"
            f"  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            f"Then decompress: bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"
        )
