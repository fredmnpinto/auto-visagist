# Facial Visagism Analysis System

**MSc Computer Vision assignment at Universidade de Aveiro, 2025/2026**

A Python-based computer vision system for facial proportion analysis, face shape classification, and golden ratio comparison using 68-point facial landmarks and automatic hairline detection.

---

## Overview

The Facial Visagism Analysis System is an academic project developed as part of the MSc Computer Vision curriculum at Universidade de Aveiro. It addresses the problem of automated facial analysis by combining classical computer vision techniques with established visagism methodology.

The system performs:

- **Facial proportion analysis** — Measures key facial ratios (width-to-height, forehead-to-lower-face, etc.) using detected landmarks
- **Face shape classification** — Categorizes faces into 7 standard shapes based on geometric proportions
- **Golden ratio comparison** — Compares facial proportions against the golden ratio (1.618) and reports deviations
- **Hairline estimation** — Automatically detects the hairline to enable complete facial height measurements

This tool is designed for research and educational purposes in the field of computer vision and facial analysis.

---

## How It Works

The system follows a modular pipeline:

```
Input Image → Face Detection → 68-Point Landmarks → Hairline Detection → 
Proportion Calculation → Face Shape Classification → Golden Ratio Analysis → 
Visualization & Report
```

1. **Face Detection** — Uses dlib's HOG-based frontal face detector to locate the face in the image
2. **Landmark Detection** — Applies dlib's 68-point shape predictor to identify key facial features (jaw, eyes, nose, mouth, eyebrows)
3. **Hairline Detection** — Estimates the hairline position using Canny edge detection (primary method) with vertical intensity gradient fallback in the forehead region
4. **Proportion Calculation** — Computes facial width, height, and regional proportions from landmarks and hairline
5. **Face Shape Classification** — Classifies the face into one of 7 categories using ratio-based rules
6. **Golden Ratio Analysis** — Compares measured proportions against the golden ratio and calculates deviations
7. **Visualization** — Optionally renders annotated images with colored landmark regions and saves results

---

## Key Features

- **68-point facial landmark detection** using dlib's pre-trained shape predictor
- **Automatic hairline estimation** via Canny edge detection and morphological closing
- **7-category face shape classification**: Oval, Round, Square, Oblong, Heart, Triangle, Diamond
- **Golden ratio deviation analysis** with per-metric reporting
- **Landmark visualization** with region-colored overlays (jaw, eyes, nose, mouth, eyebrows, hairline)
- **Diagnostic tools** for hairline detection debugging and step-by-step visualization
- **Landmark evaluation tool** with interactive ground-truth labeling and batch evaluation against predictions

---

## Technology Stack

| Component | Version / Tool |
|-----------|---------------|
| Python | 3.12 |
| OpenCV | 4.x (GTK2 enabled) |
| dlib | 19.x |
| NumPy | Latest (via Nix) |
| Matplotlib | Latest (via Nix) |
| pytest | Latest (via Nix) |
| Environment | Nix flake |

---

## Installation & Setup

### Prerequisites

- [Nix](https://nixos.org/download.html) package manager installed, **or**
- Manual Python 3.12 installation with all dependencies listed above

### Enter Development Shell

```bash
# Enter the Nix devshell (all dependencies pre-installed)
nix develop

# Or auto-load via direnv (if .envrc is configured)
direnv allow
```

### Download dlib Model

The dlib 68-point facial landmark model is **not included in the repository** (~100 MB). Download it as follows:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mkdir -p data/models
mv shape_predictor_68_face_landmarks.dat data/models/
```

The system will automatically locate the model under `data/models/` at runtime.

---

## Usage

### Main CLI

Analyze a single image:

```bash
python visagism.py --input <image_path> [--visualize] [--save-viz] [--hairline]
```

Examples:

```bash
python visagism.py --input photo.jpg
python visagism.py --input photo.jpg --visualize
python visagism.py --input photo.jpg --save-viz --output ./results
python visagism.py --input photo.jpg --model path/to/model.dat --visualize
```

### Landmark Evaluation — Label Mode

Interactively create or correct ground-truth landmark labels:

```bash
python scripts/landmark_evaluation.py --mode label --input <image_or_dir>
```

- Predictions are shown as starting points (gray circles)
- Click any landmark to correct its position
- Press `r` to reset all landmarks to the current prediction
- Use arrow keys, `n` / `p` for image navigation
- Labels auto-save to `data/ground_truth/<stem>_gt.json`

### Landmark Evaluation — Evaluate Mode

Batch-evaluate predicted landmarks against ground truth:

```bash
python scripts/landmark_evaluation.py --mode evaluate \
    --images-dir <dir> --ground-truth-dir <dir> --report report.json
```

This runs the live detection pipeline on each image, compares predicted landmarks against ground truth, and reports mean pixel error, NME, per-region errors, and hairline error.

---

## Results / Evaluation

### Hairline Detection Evaluation

> **Important**: Both the Canny and No-Canny pipelines use the **exact same dlib 68-point shape predictor** for facial landmark detection. The 68 facial landmarks are identical in both methods. The **only difference** is the hairline estimation algorithm. This comparison evaluates **hairline detection only**.

#### 68-Point Landmark Accuracy (identical in both methods)

Since both pipelines use the same dlib model, landmark accuracy is identical:

| Metric | Value |
|--------|-------|
| Mean error | 1.8 px |
| NME | 0.0044 |

### Hairline Detection Comparison

Evaluation on a 9-image ground-truth test set:

| Metric | No-Canny | Canny |
|--------|----------|-------|
| Mean error | 100.3 px | **38.3 px** |
| Median error | 82.0 px | **49.0 px** |
| ≤ 3 px accuracy | 0/9 (0%) | **4/9 (44.4%)** |
| ≤ 50 px accuracy | 3/9 (33.3%) | **6/9 (66.7%)** |

#### Per-Image Hairline Errors

| Image | No-Canny | Canny | Winner |
|-------|----------|-------|--------|
| woman_10 | 25 px | **2 px** | Canny |
| woman_11 | **85 px** | 90 px | No-Canny |
| woman_12 | 107 px | **0 px** | Canny |
| woman_13 | 143 px | **0 px** | Canny |
| woman_6 | 46 px | 46 px | Tie |
| woman_7 | 62 px | **14 px** | Canny |
| woman_8 | 158 px | **29 px** | Canny |
| woman_9 | **137 px** | 164 px | No-Canny |
| woman_makeup_1 | 140 px | **0 px** | Canny |

#### Key Findings

- **Canny achieves 62% lower mean hairline error** compared to the No-Canny approach
- Canny wins on **6/9 images**, No-Canny on **2/9**, with **1 tie**
- **Decision**: Canny is used as the primary hairline detection method; No-Canny is retained as a fallback for false-positive edge cases

---

## Project Structure

```
.
├── visagism/              # Core Python package
│   ├── face_detector.py
│   ├── landmark_detector.py
│   ├── hairline_detector.py
│   ├── landmark_evaluator.py
│   ├── landmark_ground_truth.py
│   ├── landmark_labeler.py
│   ├── landmark_visualizer.py
│   ├── image_loader.py
│   ├── model_finder.py
│   ├── cli.py
│   ├── constants.py
│   ├── errors.py
│   ├── types.py
│   └── __init__.py
├── scripts/               # CLI utilities and evaluation tools
│   ├── landmark_evaluation.py
│   ├── demo_hairline_steps.py
│   ├── diagnose_hairline.py
│   ├── batch_canny_viewer.py
│   ├── compare_canny_vs_no_canny.py
│   ├── evaluate_both_correct.py
│   ├── evaluate_no_canny.py
│   ├── export_predictions.py
│   ├── visualize_ground_truth.py
│   ├── analyze_3px_threshold.py
│   ├── analyze_hairline_3px.py
│   ├── compare_morphology.py
│   └── download_model.sh
├── tests/                 # pytest test suite
│   ├── test_face_detector.py
│   ├── test_landmark_detector.py
│   ├── test_hairline_detector.py
│   ├── test_hairline_e2e.py
│   ├── test_landmark_evaluator.py
│   ├── test_landmark_labeler.py
│   ├── test_landmark_ground_truth.py
│   ├── test_landmark_visualizer.py
│   ├── test_image_loader.py
│   ├── test_model_finder.py
│   ├── test_cli.py
│   ├── test_demo_hairline_steps.py
│   ├── test_batch_canny_viewer.py
│   ├── conftest.py
│   └── __init__.py
├── docs/                  # Documentation
│   └── specs/
│       └── functional-spec.md
├── data/                  # dlib model and generated outputs
│   ├── models/            # dlib shape predictor (not in repo)
│   ├── ground_truth/      # Manual annotations
│   │   └── images/        # Ground-truth source images
│   └── output/            # Generated visualizations and reports
├── images/                # Sample input images
├── test_images/           # Test set images
├── flake.nix              # Nix development environment
├── flake.lock             # Nix lock file
├── setup.cfg              # pytest / coverage / flake8 configuration
├── .envrc                 # direnv auto-load configuration
└── visagism.py            # Main entry point
```

---

## Development Environment

This project uses a **Nix flake** to provide a fully reproducible development environment. All Python dependencies (OpenCV with GTK2, dlib, NumPy, Matplotlib, pytest, flake8, mypy) are pre-installed — no `pip install` needed.

To auto-load the environment when entering the project directory:

```bash
direnv allow
```

This reads the `.envrc` file and activates the Nix devshell automatically.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_hairline_detector.py::TestHairlineDetector::test_detect_returns_hairline -v

# Run tests by keyword
python -m pytest tests/ -k "hairline" -v

# Coverage report
python -m pytest --cov=visagism --cov-report=term

# Linting
python -m flake8 visagism/ tests/

# Type checking
python -m mypy visagism/
```

**Coverage target**: ≥ 85%

---

## Assignment Context

- **Course**: MSc Computer Vision
- **Institution**: Universidade de Aveiro
- **Academic Year**: 2025/2026
- **Team Size**: 2 students
- **Target Level**: Level 1–2 (single solution with dataset acquisition and comparison methodology)

This project demonstrates the full computer vision pipeline: dataset collection, algorithm implementation, quantitative evaluation against ground truth, and comparative analysis of two hairline detection approaches (Canny edge detection vs. vertical intensity gradient analysis). The facial landmark detection (68 points) is identical in both approaches, using dlib's pre-trained shape predictor.

---

## References

- **dlib 68-point facial landmark model**: [http://dlib.net/face_landmark_detection.py.html](http://dlib.net/face_landmark_detection.py.html)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Visagism Methodology**: Classical facial analysis techniques based on geometric proportions and the golden ratio

---

*Developed for academic purposes. All external code and libraries are referenced per assignment requirements.*
