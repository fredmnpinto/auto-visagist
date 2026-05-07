# AGENTS.md

## Project

Facial Visagism Analysis System — Python 3.12, OpenCV 4.x (GTK2), dlib 19.x, NumPy, Matplotlib.
MSc Computer Vision assignment @ Universidade de Aveiro, 2025/2026.

## Development Environment

- **Nix flake**: `../flake.nix` — enter with `nix develop` or auto-load via direnv (`.envrc` uses `use flake`)
- **Pre-installed in devshell**: Python 3.12, numpy, matplotlib, opencv (with GTK2 for `cv2.imshow`)
- **Adding dependencies**: All dependencies go in `../flake.nix`. Never `pip install` manually.
  Currently missing from flake (add as needed): dlib, pytest, pytest-cov, flake8, mypy
- **Download dlib model** (not in repo, ~100MB, gitignored):
  ```
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  # Place at: data/shape_predictor_68_face_landmarks.dat
  ```
- **Dependencies** tracked in `requirements.txt` (create one documenting all deps including those from flake)

## Commands

| Action | Command |
|--------|---------|
| Enter devshell | `nix develop` |
| Install deps | `pip install -r requirements.txt` |
| Run all tests | `python -m pytest tests/ -v` |
| Run single test | `python -m pytest tests/test_X.py::TestY::test_z -v` |
| Run by keyword | `python -m pytest tests/ -k "keyword" -v` |
| Coverage | `python -m pytest --cov=src --cov-report=term` |
| Lint | `python -m flake8 src/ tests/` |
| Typecheck | `python -m mypy src/` |
| Run CLI | `python visagism_analyzer.py --input <path> [--visualize]` |

## Code Style

### Imports

One group per category, separated by blank line. Order: standard library → third-party → local.

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from visagism.constants import GOLDEN_RATIO, SUPPORTED_FORMATS
```

### Formatting

- PEP 8 (4-space indent, 88-character line limit)
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants and enums
- Two blank lines between top-level definitions, one between methods
- One space around operators, after commas

### Type Annotations

Use `from __future__ import annotations` at the top of every module. Always annotate public function signatures. Use expressive type aliases.

```python
Point = Tuple[int, int]
Landmarks = List[Point]
ImageArray = npt.NDArray[np.uint8]


def detect_facial_landmarks(
    img_gray: ImageArray,
    detector: dlib.get_frontal_face_detector,
    predictor: dlib.shape_predictor,
) -> Optional[Landmarks]:
    ...
```

### Naming Conventions

**Functions/methods** — descriptive verbs: `detect_facial_landmarks()`, `calculate_width_height_ratio()`, `classify_face_shape()`, `generate_report()`.

**Classes** — nouns describing the component: `FaceDetector`, `LandmarkDetector`, `ProportionCalculator`, `FaceShapeClassifier`, `VisagismAnalyzer`, `ReportGenerator`.

**Constants** — explicit naming: `GOLDEN_RATIO = 1.618`, `SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}`, `MIN_RESOLUTION = (200, 200)`, `DEVIATION_THRESHOLD = 0.10`, `OUTPUT_DIR = "output"`.

**CV variable prefixes** — descriptive prefixes indicating what the variable holds:
- `img_` for image arrays: `img_color`, `img_gray`, `img_annotated`
- `gray_` for grayscale: `gray_face`
- `bbox_` for bounding boxes: `bbox_face` (tuple x, y, w, h)
- `pts_` for point collections: `pts_jaw`, `pts_leye`, `pts_reye`, `pts_nose`, `pts_mouth`, `pts_lbrow`, `pts_rbrow`
- `landmarks_` for the full 68-point list

**Private helpers** — prefix with `_`: `_validate_image()`, `_calculate_angle()`.

### Docstrings

NumPy style for all public functions. Required for every module, class, and public method.

```python
def calculate_width_height_ratio(face_width: float, face_height: float) -> float:
    """Calculate the width-to-height ratio of a face.

    Parameters
    ----------
    face_width : float
        Face width in pixels. Must be positive.
    face_height : float
        Face height in pixels. Must be positive.

    Returns
    -------
    float
        Width-to-height ratio rounded to 2 decimal places.

    Raises
    ------
    ValueError
        If face_width or face_height is zero or negative.
    """
```

### Error Handling

Use a custom exception hierarchy for domain-specific errors. Never expose raw Python tracebacks to the CLI user.

```python
class VisagismError(Exception):
    """Base exception for all visagism analysis errors."""

class ImageError(VisagismError): ...
class UnsupportedFormatError(ImageError): ...
class CorruptedImageError(ImageError): ...

class DetectionError(VisagismError): ...
class NoFaceDetectedError(DetectionError): ...
class NonFrontalPoseError(DetectionError): ...

class AnalysisError(VisagismError): ...
class InvalidLandmarksError(AnalysisError): ...
class CalculationError(AnalysisError): ...
```

- CLI entry point catches `VisagismError` and prints user-friendly messages
- Exit code 0 = success, 1 = error
- Log internal details (not stack traces) if logging is configured

### Testing

- pytest framework
- Test files mirror source: `tests/test_<module>.py`
- Test classes: `Test<Component>` (e.g., `TestFaceDetector`)
- Test methods: `test_<scenario>` (e.g., `test_detect_single_face`, `test_no_face_returns_none`)
- Use pytest fixtures for reusable test data
- Coverage target ≥85%

## Critical Rules for Agents

1. **Read `docs/specs/functional-spec.md` before implementing** — it is the single source of truth
2. **Run tests before finishing** — ensure all tests pass
3. **No scope creep** — stick to Must/Should requirements; flag anything outside scope
4. **Document any non-original code** — per assignment rules, reference all external code/libraries
5. **Always reference original code sources** — add comments pointing to adapted code
