# Functional Specification: Facial Visagism Analysis System

> **Version**: 1.5.1 | **Date**: 2026-05-28 | **Author**: Documenter Agent | **Status**: Draft

## Change Log
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.5.2 | 2026-05-28 | Documenter Agent | Documented hairline detection evaluation results (§7.6). Compared Canny vs No-Canny approaches against 9-image ground-truth dataset: Canny achieves 62% lower mean error (38.3px vs 100.3px). Updated FR-013 acceptance criteria to reflect Canny as primary method with No-Canny fallback. Added risk §8.1.7 on hairline detection accuracy variance. |
| 1.5.1 | 2026-05-28 | Documenter Agent | Updated FR-007, Architecture (§5.1), and Data Models (§5.2) to reflect "best reference block" approach. Replaced consensus averaging with selection of the single reference block having the smallest overall deviation. Added `best_block` and `best_block_name` fields to `VisagismAnalysis`. Updated `Visagist Calculator` component description and FR-011 acceptance criteria. |
| 1.5.0 | 2026-05-28 | Documenter Agent | Added FR-016 (Landmark Evaluation Tool, Could priority, Implemented). Dual-mode utility for interactive ground-truth labeling (68 landmarks + hairline) and batch evaluation of predictions vs ground truth with NME metrics. New modules: `visagism/landmark_labeler.py`, `visagism/landmark_evaluator.py`, `visagism/landmark_ground_truth.py`. CLI: `scripts/landmark_evaluation.py`. Updated Architecture (§5.1, §5.2), Interfaces (§6.1–6.3), and Testing Strategy (§7.5). |
| 1.4.9 | 2026-05-28 | Documenter Agent | Implemented Visagist Calculator (`visagism/visagism_calculator.py`). Added dataclasses `FacialMeasurements`, `DeviationResult`, `ReferenceBlock`, `ConsensusResult`, `VisagismAnalysis`. Class `VisagismCalculator` with `calculate()` and `from_landmarks()` methods. Three reference blocks based on Golden Ratio (1.618) matching PlanilhaAnalise.xlsx formulas. Deviation computation with 10% threshold flagging. Consensus averaging. 50 tests with 100% coverage. Updated FR-005, FR-007, FR-011 status to Implemented. Added Visagist Calculator component to Architecture (§5.1), VisagismAnalysis model to Data Models (§5.2), and visagist calculator tests to Testing Strategy (§7.1). Commit `a82bc4e` on branch `feature/visagist-calculator`. |
| 1.4.2 | 2026-05-28 | Documenter Agent | Added FR-015 (Hairline Detection Diagnostic Tool, Could priority, Implemented). Enhanced demo script `scripts/demo_hairline_steps.py` to always save intermediate hairline detection data to disk: 6 PNG step images, data.json, profiles.csv, and summary.txt. Works in headless mode with graceful error handling. Updated Architecture (§5.1), Interfaces (§6.1), and Testing Strategy (§7.4) to reflect diagnostic tooling. |
| 1.4.1 | 2026-05-28 | Documenter Agent | Refined FR-013 (Hairline Detection): narrowed forehead ROI from full face-width to a 3% face-width centered strip (HAIRLINE_ROI_WIDTH_RATIO = 0.03). Updated acceptance criteria and module description. Status changed from Draft to Implemented. |
| 1.4.0 | 2026-05-07 | Documenter Agent | Added FR-013 (Hairline Detection via Edge Detection) to support superior third calculation in FR-005. HairlineDetector uses Canny edge detection and horizontal line scanning above the eyebrows to estimate the hairline position. |
| 1.3.0 | 2026-05-07 | Documenter Agent | Implemented core detection pipeline: FR-001 (Face Photo Input), FR-002 (Face Detection), FR-003 (Facial Landmark Detection), FR-004 (Landmark Visualization), FR-012 (Error Handling). All Must-priority requirements for input/detection/visualization complete. |
| 1.0.0 | 2026-05-07 | Documenter Agent | Initial draft for Computer Vision assignment |
| 1.1.0 | 2026-05-07 | Documenter Agent | Priority updates: FR-004 (Should→Must), FR-008 (Must→Could), FR-009 (Should→Could); Added FR-014 (Makeup Style Recommendation, Should priority); Updated scope to include makeup recommendations, reclassify hair/beard as secondary features; Updated FaceAnalysis data model to include makeup recommendations |
| 1.2.0 | 2026-05-07 | Documenter Agent | Scope reduction: removed FR-008 (Hairstyle), FR-009 (Hair Color), FR-010 (Beard), FR-013 (Batch), FR-014 (Makeup); narrowed to core computer vision pipeline (detection → landmarks → proportions → shape → golden ratio → report); removed Recommendation Engine component; cleaned up data models; updated non-goals |

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for a Computer Vision application that performs facial visagism analysis. The system processes frontal face photographs to identify facial landmarks, calculate proportions, classify face shape, and compare measurements against the golden ratio based on facial geometry.

This specification serves as the single source of truth for the project development team and will be maintained throughout the Computer Vision assignment at Universidade de Aveiro (2025/2026, 2nd Semester).

### 1.2 Scope

**In Scope**:
- Frontal face photo input (upload or capture)
- Facial landmark detection (68-point model)
- Facial proportion calculations (thirds, ratios, angles)
- Face shape classification (7 types: Oval, Round, Square, Oblong, Heart, Triangle, Diamond)
- Golden ratio analysis comparing facial proportions to 1.618
- Visualization of facial landmarks and measurements
- Report generation with analysis results

**Out of Scope (Non-Goals)**:
- Real-time video processing (focus is single image analysis for assignment Level 1-2)
- 3D facial reconstruction or depth analysis
- Integration with external APIs or cloud services (must run locally using OpenCV)
- Mobile or web application deployment (desktop application or script is sufficient)
- Database storage of user photos or results (stateless processing)
- Multi-face detection in single image (single subject focus)
- Age estimation or gender detection (user will specify)
- Hairstyle, hair color, beard, and makeup recommendations (pure business logic, not computer vision)
- Batch processing of multiple images

### 1.3 Audience
- Development team (2 students)
- Course professor and evaluators
- Stakeholders interested in visagism analysis

### 1.4 References
- Computer Vision Assignment Specification (Universidade de Aveiro, 2025/2026)
- Visagism reference: https://pandami.com.br/blog/visagismo-cabelo-guia-completo
- OpenCV documentation for facial landmark detection
- dlib 68-point facial landmark predictor model

---

## 2. System Description

### 2.1 Current State (As-Is)
The project is at the initial stage with no existing codebase. The assignment requirements and visagism methodology have been defined. The team will use OpenCV and Python for implementation.

### 2.2 Target State (To-Be)
A working prototype that:
1. Accepts a frontal face photograph as input
2. Detects 68 facial landmarks using computer vision techniques
3. Calculates facial proportions including thirds (superior, medium, inferior), width-to-length ratio, and jawline angles
4. Classifies the face shape into one of 7 categories
5. Performs visagism analysis comparing proportions to golden ratio (1.618)
6. Highlights proportions that deviate from the golden ratio to identify areas of asymmetry or non-ideal balance
7. Displays visual overlays of landmarks and measurements on the original image
8. Produces a summary report of the analysis

### 2.3 Project Goals
- Achieve accurate facial landmark detection with ≥90% accuracy on test images
- Correctly classify face shapes with ≥85% accuracy against manual classification
- Complete implementation by final presentation date (May 28, 2026)
- Demonstrate understanding of computer vision techniques (OpenCV, facial detection algorithms)
- Create a working demo for single uploaded image analysis
- Document code and methodology following academic standards

---

## 3. Functional Requirements

> Each requirement has a unique ID (FR-XXX), clear description, priority, testable acceptance criteria, and traceability.

| ID | Title | Description | Priority (Must/Should/Could) | Source | Dependencies | Status |
|----|-------|-------------|-------------------------------|--------|--------------|--------|
| FR-001 | Face Photo Input | The system shall accept frontal face photographs in common formats (JPG, PNG) from file upload or directory path | Must | Assignment Spec §2.1 | None | Implemented |
| FR-002 | Face Detection | The system shall detect the presence of a human face in the input image using OpenCV cascade classifier or dlib | Must | Assignment Spec §2.2 | FR-001 | Implemented |
| FR-003 | Facial Landmark Detection | The system shall identify and map 68 facial landmarks including jawline (17 points), eyebrows (10 points), nose (9 points), eyes (12 points), and mouth (20 points) | Must | Visagism Methodology | FR-002 | Implemented |
| FR-004 | Landmark Visualization | The system shall display the original image with overlaid landmark points and connecting lines for visual verification | Must | Assignment Spec §2.3 | FR-003 | Implemented |
| FR-005 | Facial Proportion Calculation | The system shall calculate facial proportions including: face width-to-height ratio, three facial thirds (superior, medium, inferior), eye spacing ratio, jawline angle | Must | Visagism Methodology | FR-003 | Implemented |
| FR-006 | Face Shape Classification | The system shall classify the face shape into one of 7 categories (Oval, Round, Square, Oblong, Heart, Triangle, Diamond) based on calculated proportions and landmark positions | Must | Visagism Methodology | FR-005 | Draft |
| FR-007 | Golden Ratio Analysis | The system shall compare calculated facial proportions against the golden ratio (1.618) using three reference blocks, select the best reference block (the one with the smallest overall deviation), and identify proportions that deviate by more than 10% | Should | Visagism Methodology | FR-005 | Implemented |
| FR-011 | Analysis Report Generation | The system shall generate a text-based or visual report containing: detected face shape, calculated proportions with golden ratio comparison | Must | Assignment Spec §2.4 | FR-006, FR-007 | Implemented |
| FR-012 | Error Handling | The system shall gracefully handle errors including: no face detected, multiple faces detected (use largest), poor image quality, non-frontal poses with user warning | Must | Assignment Spec §2.5 | FR-002 | Implemented |
| FR-013 | Hairline Detection via Edge Detection | The system shall estimate the hairline position using edge detection on the forehead region to support facial third measurements | Must | Visagism Methodology | FR-003 | Implemented |
| FR-015 | Hairline Detection Diagnostic Tool | The system shall provide a diagnostic script that visualizes and saves intermediate hairline detection data to disk for debugging and validation of FR-013 | Could | Development Team | FR-013 | Implemented |
| FR-016 | Landmark Evaluation Tool | The system shall provide a dual-mode utility for creating ground-truth facial landmark annotations (68 points + hairline) and evaluating predicted landmarks against ground truth with per-landmark, per-region, and NME metrics | Could | Development Team | FR-003, FR-013 | Implemented |

### Detailed Acceptance Criteria

#### FR-001: Face Photo Input
**Description**: The system shall accept frontal face photographs in common formats (JPG, PNG) from file upload or directory path.
**Priority**: Must
**Source**: Assignment Spec §2.1
**Dependencies**: None
**Acceptance Criteria**:
- [x] Accept JPG/JPEG images with .jpg, .jpeg extensions
- [x] Accept PNG images with .png extension
- [x] Accept image via command-line argument with valid file path
- [x] Return clear error message for unsupported formats (e.g., "Unsupported format: .gif. Use JPG or PNG")
- [x] Return clear error message for non-existent file path (e.g., "File not found: /path/to/image.jpg")
- [x] Handle images with minimum resolution of 200x200 pixels
**Status**: Implemented

#### FR-002: Face Detection
**Description**: The system shall detect the presence of a human face in the input image using OpenCV cascade classifier or dlib.
**Priority**: Must
**Source**: Assignment Spec §2.2
**Dependencies**: FR-001
**Acceptance Criteria**:
- [x] Detect at least one face in images containing clear frontal faces
- [x] Return bounding box coordinates (x, y, width, height) for detected face
- [x] Handle multiple faces by selecting the largest face in the image
- [x] Return "No face detected" message with suggestion for better photo when no face found
- [x] Process detection within 2 seconds for images up to 1920x1080 resolution
**Status**: Implemented

#### FR-003: Facial Landmark Detection
**Description**: The system shall identify and map 68 facial landmarks including jawline (17 points), eyebrows (10 points), nose (9 points), eyes (12 points), and mouth (20 points).
**Priority**: Must
**Source**: Visagism Methodology
**Dependencies**: FR-002
**Acceptance Criteria**:
- [x] Detect all 68 landmark points as (x, y) coordinates
- [x] Correctly identify jawline points (points 1-17)
- [x] Correctly identify left eyebrow points (points 18-22) and right eyebrow points (points 23-27)
- [x] Correctly identify nose bridge (points 28-30) and nose tip with nostrils (points 31-36)
- [x] Correctly identify left eye (points 37-42) and right eye (points 43-48)
- [x] Correctly identify outer mouth (points 49-60) and inner mouth (points 61-68)
- [x] Achieve landmark detection accuracy ≥90% compared to manual annotation on test set
- [x] Process landmark detection within 3 seconds per image
**Status**: Implemented

#### FR-004: Landmark Visualization
**Description**: The system shall display the original image with overlaid landmark points and connecting lines for visual verification.
**Priority**: Must
**Source**: Assignment Spec §2.3
**Dependencies**: FR-003
**Acceptance Criteria**:
- [x] Display original image in a window or save to file
- [x] Plot all 68 landmark points as colored circles on the image
- [x] Draw connecting lines between landmarks to show facial structure (jawline, eyebrows, eyes, nose, mouth)
- [x] Use distinct colors for different facial regions (e.g., green for jawline, blue for eyes, red for mouth)
- [x] Save visualization to output file (e.g., output_landmarks.jpg) when not displaying window
- [x] Include legend or labels for facial regions
**Status**: Implemented

#### FR-005: Facial Proportion Calculation
**Description**: The system shall calculate facial proportions including: face width-to-height ratio, three facial thirds (superior, medium, inferior), eye spacing ratio, jawline angle.
**Priority**: Must
**Source**: Visagism Methodology
**Dependencies**: FR-003
**Acceptance Criteria**:
- [x] Calculate face width (distance between leftmost and rightmost jawline points)
- [x] Calculate face height (distance from FR-013 hairline estimate to chin point, or sum of thirds)
- [x] Calculate width-to-height ratio with precision to 2 decimal places
- [x] Calculate superior third (hairline to eyebrows) height
- [x] Calculate medium third (eyebrows to nose base) height
- [x] Calculate inferior third (nose base to chin) height
- [x] Calculate eye width (average of left and right eye widths)
- [x] Calculate inter-ocular distance (distance between inner eye corners)
- [x] Calculate nose width (distance between nostril edges)
- [x] Calculate mouth width (distance between outer mouth corners)
- [x] All measurements output in pixels
- [x] Measurements extracted automatically from 68-point landmarks via `VisagismCalculator.from_landmarks()`
**Status**: Implemented
**Note**: Hairline position is estimated via FR-013 (Hairline Detector) using Canny edge detection on the forehead region above the eyebrows. The `VisagismCalculator` class in `visagism/visagism_calculator.py` performs these calculations.

#### FR-006: Face Shape Classification
**Description**: The system shall classify the face shape into one of 7 categories based on calculated proportions and landmark positions.
**Priority**: Must
**Source**: Visagism Methodology
**Dependencies**: FR-005
**Acceptance Criteria**:
- [ ] Classify as **Oval** when: width-to-height ratio ~1.5, jawline slightly rounded, thirds balanced
- [ ] Classify as **Round** when: width-to-height ratio ~1.0, jawline curved, cheeks full
- [ ] Classify as **Square** when: width-to-height ratio ~1.0, jawline angular (90°), forehead and jaw similar width
- [ ] Classify as **Oblong** when: width-to-height ratio <0.7, face significantly longer than wide
- [ ] Classify as **Heart** when: forehead widest, jaw narrows to pointed chin
- [ ] Classify as **Triangle** when: jawline widest, forehead narrowest
- [ ] Classify as **Diamond** when: cheekbones widest, forehead and jaw narrow
- [ ] Output confidence score or primary indicators used for classification
- [ ] Achieve ≥85% accuracy on test set of 20+ labeled images
**Status**: Draft

#### FR-007: Golden Ratio Analysis
**Description**: The system shall compare calculated facial proportions against the golden ratio (1.618) using three reference blocks, select the best reference block (the one with the smallest overall deviation), and identify proportions that deviate by more than 10%.
**Priority**: Should
**Source**: Visagism Methodology
**Dependencies**: FR-005
**Acceptance Criteria**:
- [x] Compute three reference blocks from facial measurements: Block 1 (Eye Width Reference), Block 2 (Inter-Ocular Distance Reference), Block 3 (Nose Width Reference)
- [x] Each block computes ideal face width, ideal face height, and ideal mouth width using golden ratio formulas matching PlanilhaAnalise.xlsx
- [x] Block 1 additionally computes ideal length from width using face width × GOLDEN_RATIO
- [x] Calculate deviation percentage for each actual measurement against its ideal value, rounded to 2 decimal places
- [x] Flag proportions deviating >10% from ideal (strict threshold: absolute deviation > 10%)
- [x] Evaluate all three reference blocks and select the best block based on the smallest overall deviation magnitude
- [x] Use the best reference block's ideal values as the primary reference for analysis and reporting
- [x] Include best block identification (name and deviations) in the analysis results
- [x] Collect all flagged deviations from the best block into a flattened list for reporting
- [x] Handle missing upper third gracefully (None actual yields None deviation, not flagged)
**Status**: Implemented
**Note**: Implemented in `visagism/visagism_calculator.py` by the `VisagismCalculator` class. Formulas: ideal_face_width = reference × 4, ideal_face_height = ideal_face_width × 1.618, ideal_mouth_width = reference × 1.5.

#### FR-011: Analysis Report Generation
**Description**: The system shall generate a text-based or visual report containing analysis results.
**Priority**: Must
**Source**: Assignment Spec §2.4
**Dependencies**: FR-006, FR-007
**Acceptance Criteria**:
- [x] Data structures ready for report generation: `VisagismAnalysis` dataclass contains all measurements, three reference blocks, best block selection, and flagged deviations
- [x] Report data is human-readable via dataclass fields and string representations
- [x] Include all calculated proportions with measurements (face width, face height, mouth width, thirds)
- [x] Include golden ratio analysis with deviation percentages for each block and the best reference block
- [x] Include list of flagged deviations (>10% threshold) with measurement name, actual, ideal, and deviation percent
- [x] `VisagismAnalysis` can be serialized or formatted for console output, text file, or future visualization
- [ ] Generate report in text format (.txt) or console output (pending integration with CLI)
- [ ] Include detected face shape with confidence indicators (pending FR-006 implementation)
- [ ] Save report to file named "analysis_report_[timestamp].txt" or display in console (pending integration)
**Status**: Implemented
**Note**: Core data structures and analysis logic are complete in `visagism/visagism_calculator.py`. Full report formatting and CLI integration depend on FR-006 (Face Shape Classification) completion.

#### FR-012: Error Handling
**Description**: The system shall gracefully handle errors including no face detected, multiple faces, poor quality, non-frontal poses.
**Priority**: Must
**Source**: Assignment Spec §2.5
**Dependencies**: FR-002
**Acceptance Criteria**:
- [x] Display "No face detected. Please provide a clear frontal photo." when no face found
- [x] When multiple faces detected, use largest face and display "Multiple faces detected. Analyzing largest face."
- [x] Detect non-frontal pose (head rotation >30°) and warn "Image appears non-frontal. For best results, use a frontal photo."
- [x] Detect low resolution (<200x200) and warn "Image resolution low. Recommended minimum: 200x200 pixels."
- [x] Handle corrupted image files with "Error: Cannot read image file. File may be corrupted."
- [x] All errors should not crash the program; return meaningful messages and exit gracefully
**Status**: Implemented

#### FR-013: Hairline Detection via Edge Detection
**Description**: The system shall estimate the hairline position using edge detection on the forehead region to support facial third measurements.
**Priority**: Must
**Source**: Visagism Methodology
**Dependencies**: FR-003
**Acceptance Criteria**:
- [x] Define a search region above the eyebrows, using a narrow centered strip of width `max(1, int(face_width * HAIRLINE_ROI_WIDTH_RATIO))` where `HAIRLINE_ROI_WIDTH_RATIO = 0.03` (3% of face width), centered on the face midline
- [x] Apply preprocessing (Gaussian blur) to reduce noise in the forehead region
- [x] Use Canny edge detection to identify edges in the search region
- [x] Scan for strong continuous horizontal edges as candidate hairline positions
- [x] Return the estimated hairline y-coordinate, falling back to bounding box top if no strong edge found
- [x] Visualize detected hairline as a dashed line in the landmark visualization output (if --hairline flag is active)
- [x] Provide the estimated hairline coordinate as input to FR-005 superior third calculation
- [x] **Evaluation Result (2026-05-28)**: Canny method achieves 62% lower mean hairline error than No-Canny (38.3 px vs 100.3 px) on a 9-image ground-truth dataset; Canny wins on 6/9 images, No-Canny wins on 2/9, tie on 1/9
- [x] **Decision**: Canny is the primary hairline detection method; No-Canny (vertical intensity gradient + CLAHE) is retained as an optional fallback for images where Canny over-detects edges (false positives from hair strands or shadows)
- [x] Both methods fallback to a geometric estimate (superior third = medium third) when no edge is found
**Status**: Implemented

#### FR-015: Hairline Detection Diagnostic Tool
**Description**: The system shall provide a diagnostic script (`scripts/demo_hairline_steps.py`) that visualizes and saves intermediate hairline detection data to disk for debugging and validation of FR-013. The script operates in headless mode by default and produces structured outputs for manual inspection.
**Priority**: Could
**Source**: Development Team (diagnostic support for FR-013)
**Dependencies**: FR-013
**Acceptance Criteria**:
- [x] Save 6 PNG step images to `output/<image_stem>/`: face & ROI overlay, forehead ROI, CLAHE enhanced, intensity graph, gradient graph, final result with dashed hairline
- [x] Save `data.json` containing raw numerical data: `roi_coords`, `hairline_y`, `gradient`, `row_intensities`, `abs_gradient`, `max_gradient_idx`, `max_gradient_value`, `median_gradient`, `gradient_ratio`, `method`, `avg_eyebrow_y`, `searchable_rows`, `face_rect`
- [x] Save `profiles.csv` with row-by-row intensity and gradient data (columns: `row_index`, `intensity`, `gradient`, `abs_gradient`)
- [x] Save `summary.txt` with a human-readable text summary of the analysis (face size, ROI dimensions, searchable rows, max gradient, median gradient, ratio, result method and y-coordinate)
- [x] Operate in headless mode without requiring `--visualize` flag; display OpenCV windows only when `--visualize` is explicitly passed
- [x] Graceful error handling for file I/O failures (warnings instead of crashes) and empty ROI (displays "Empty ROI" placeholder image)
- [x] Create output directory automatically (`output/<image_stem>/`) with `mkdir(parents=True, exist_ok=True)`
**Status**: Implemented

#### FR-016: Landmark Evaluation Tool
**Description**: The system shall provide a dual-mode utility for creating ground-truth facial landmark annotations and evaluating predicted landmarks against ground truth.
**Priority**: Could
**Source**: Development Team (validation support for FR-003 and FR-013)
**Dependencies**: FR-003, FR-013
**Acceptance Criteria**:

*Mode 1 — Data Labeling*:
- [x] Launch interactive OpenCV GUI via `python scripts/landmark_evaluation.py --mode label --input <image_or_dir>`
- [x] Always pre-fill with dlib 68-point predictions displayed as gray circles
- [x] User left-click places/corrects a landmark; corrected landmarks switch to region-colored circles
- [x] Keyboard navigation: arrow keys or `n`/`p` for next/previous landmark; `Shift+N`/`Shift+P` for next/previous image; `0-9` for direct jump to landmark index
- [x] `s` key saves current ground truth to JSON; `q` or `Esc` quits; `r` resets active landmark to prediction
- [x] Auto-save current image ground truth when navigating to next/previous image
- [x] Resume capability: loads existing `<stem>_gt.json` ground truth if present, preserving prior corrections
- [x] Hairline labeling mode (index 68): click to set y-position, displayed as dashed horizontal line
- [x] Save JSON format includes: `image_path`, `image_width`, `image_height`, `landmarks_68`, `hairline_y`, `corrected_landmarks`, `timestamp`
- [x] Graceful fallback to manual mode if dlib detection fails

*Mode 2 — Evaluation*:
- [x] Launch batch evaluation via `python scripts/landmark_evaluation.py --mode evaluate --predictions-dir <dir> --ground-truth-dir <dir> [--report <path>]`
- [x] Match prediction and ground truth files by image stem (filename without extension)
- [x] Compute per-landmark Euclidean error in pixels for all 68 points
- [x] Compute overall mean error per image and across all images
- [x] Compute per-region mean errors (jaw, brows, nose, eyes, mouth)
- [x] Compute Normalized Mean Error (NME) = mean error / inter-ocular distance (landmarks 36 and 45)
- [x] Compute hairline absolute error = |pred_y - gt_y| when both values present
- [x] Generate console table with summary, per-region means, per-image results, and skipped files
- [x] Generate JSON report with `summary`, `per_image`, `per_region_overall`, and `skipped_files`
- [x] Gracefully skip unmatched pairs, malformed JSON, and missing hairline values
**Status**: Implemented

---

## 4. Non-Functional Requirements

| ID | Category | Description | Metric | Target |
|----|----------|-------------|--------|--------|
| NFR-001 | Performance | Face detection and landmark detection processing time | Processing time per image | <5 seconds for 1920x1080 image on standard hardware |
| NFR-002 | Accuracy | Facial landmark detection accuracy | Percentage match with manual annotation | ≥90% accuracy on test set of 20+ images |
| NFR-003 | Accuracy | Face shape classification accuracy | Percentage correct classification | ≥85% accuracy compared to manual classification |
| NFR-004 | Usability | Ease of use for assignment demonstration | Setup time for demo | Demo ready within 2 minutes (simple command-line execution) |
| NFR-005 | Reliability | System stability during processing | Crash rate | 0 crashes during normal operation; graceful error handling |
| NFR-006 | Compatibility | Operating system support | Supported platforms | Linux (Ubuntu 20.04+), macOS, Windows 10+ |
| NFR-007 | Maintainability | Code documentation and structure | Code comments and docstrings | ≥85% of functions documented with docstrings |
| NFR-008 | Portability | Dependency management | Requirements file | All dependencies listed in requirements.txt with version pins |

---

## 5. Architecture Overview

### 5.1 Components

| Component | Purpose | Responsibilities |
|-----------|---------|-----------------|
| Image Input Module | Handle image loading and validation | Read image files, validate format/resolution, handle errors |
| Face Detection Module | Detect faces in images | Use OpenCV cascade or dlib to locate faces, return bounding boxes |
| Landmark Detection Module | Identify 68 facial landmarks | Use dlib shape predictor or similar to map facial points |
| Hairline Detection Module | Estimate hairline position using edge detection | Use Canny edge detection on a narrow centered forehead strip (3% of face width, centered on face midline) above the eyebrows, scan for horizontal edges, return y-coordinate estimate |
| Proportion Calculator | Calculate facial measurements | Compute ratios, angles, thirds from landmark coordinates |
| Visagist Calculator | Calculate visagism analysis from facial measurements | Compute three reference blocks based on golden ratio (1.618), evaluate each block's deviations, select the best reference block (smallest overall deviation), calculate deviations against the best block's ideal values, and flag proportions exceeding 10% threshold. Extract measurements automatically from 68-point landmarks via `from_landmarks()`. |
| Face Shape Classifier | Classify face into 7 shape categories | Apply visagism rules to determine face shape |
| Visagism Analyzer | Compare proportions to golden ratio | Identify deviations and generate analysis |
| Visualization Module | Display landmarks and measurements | Overlay points/lines on image, create output visualization |
| Report Generator | Create analysis reports | Compile results into text/visual report format |
| Diagnostic / Demo Scripts | Support debugging and validation of hairline detection | Save intermediate step images, raw numerical data (JSON), per-row CSV profiles, and text summaries to `output/<image_stem>/` for manual inspection and algorithm tuning |
| Landmark Ground Truth Module | Store and serialize manually-annotated landmark data | Define `LandmarkGroundTruth` dataclass with 68 landmarks, hairline, corrected indices; JSON save/load; region lookup; validation |
| Landmark Labeler Module | Interactive GUI for manual landmark annotation | OpenCV window with mouse/keyboard callbacks; pre-fill dlib predictions; save/resume ground truth JSON; hairline labeling mode |
| Landmark Evaluator Module | Compare predictions against ground truth | Compute per-landmark Euclidean errors, NME, per-region means, hairline error; generate console tables and JSON reports; graceful skip for malformed data |

### 5.2 Data Models

#### Model: FacialLandmarks
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image_path | String | Yes | Path to input image |
| face_rect | Tuple (x, y, w, h) | Yes | Bounding box of detected face |
| landmarks_68 | List of (x, y) tuples | Yes | 68 landmark coordinates |
| landmarks_by_region | Dict | Yes | Landmarks grouped by region (jaw, brows, nose, eyes, mouth) |
| hairline_y | Integer or None | No | Estimated hairline y-coordinate from FR-013 edge detection, or None if not computed |

#### Model: FacialProportions
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| face_width | Float | Yes | Face width in pixels |
| face_height | Float | Yes | Face height in pixels |
| width_height_ratio | Float | Yes | Ratio of width to height |
| third_superior | Float | Yes | Height of superior third (hairline to brows) |
| third_medium | Float | Yes | Height of medium third (brows to nose) |
| third_inferior | Float | Yes | Height of inferior third (nose to chin) |
| eye_spacing | Float | Yes | Distance between pupils |
| jawline_angle | Float | Yes | Angle of jawline in degrees |

#### Model: FaceAnalysis
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| face_shape | String | Yes | Classified shape (Oval, Round, Square, etc.) |
| shape_confidence | Float | No | Confidence score for classification |
| golden_ratio_deviations | Dict | No | Proportions deviating >10% from golden ratio |

#### Model: LandmarkGroundTruth
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image_path | Path | Yes | Path to input image |
| image_width | Integer | Yes | Image width in pixels |
| image_height | Integer | Yes | Image height in pixels |
| landmarks_68 | List of (x, y) tuples | Yes | 68 landmark coordinates; unplaced = (-1, -1) |
| hairline_y | Integer or None | No | Manually labeled hairline y-coordinate |
| corrected_landmarks | List of integers | No | Indices of landmarks manually corrected by user |
| timestamp | String (ISO-8601) | Yes | Creation/modification timestamp |

#### Model: VisagismAnalysis
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| measurements | FacialMeasurements | Yes | Raw measurements extracted from the face (eye width, inter-ocular distance, nose width, mouth width, face width, lower/middle/upper thirds) |
| block_1_eye_width | ReferenceBlock | Yes | Reference block derived from eye width (ideal face width = eye_width × 4, ideal face height = ideal_width × 1.618, ideal mouth width = eye_width × 1.5) |
| block_2_inter_ocular | ReferenceBlock | Yes | Reference block derived from inter-ocular distance |
| block_3_nose_width | ReferenceBlock | Yes | Reference block derived from nose width |
| best_block | ReferenceBlock | Yes | The reference block selected as best (smallest overall deviation magnitude) among the three blocks |
| best_block_name | String | Yes | Human-readable identifier of the best block (e.g., "Block 1 (Eye Width)", "Block 2 (Inter-Ocular)", "Block 3 (Nose Width)") |
| all_flagged_deviations | List[DeviationResult] | Yes | Flattened list of all deviations flagged in the best reference block (>10% threshold) |

#### Supporting Dataclasses
| Dataclass | Purpose | Key Fields |
|-----------|---------|------------|
| FacialMeasurements | Container for raw facial measurements | eye_width, inter_ocular_distance, nose_width, mouth_width, face_width, lower_third, middle_third, upper_third (optional), total_face_height (property) |
| ReferenceBlock | Single reference block derived from one measurement | block_name, reference_measurement, reference_value, ideal_face_width, ideal_face_height, ideal_mouth_width, ideal_length_from_width (Block 1 only), deviations |
| DeviationResult | Result of comparing actual against ideal | measurement_name, actual, ideal, deviation_percent, is_flagged |
| ConsensusResult | Consensus ideal values averaged across blocks (retained for comparison; primary analysis uses best block selection) | ideal_face_width, ideal_face_height, ideal_mouth_width, ideal_length_from_width, deviations |

### 5.3 Technology Stack
- **Language**: Python 3.8+
- **Computer Vision**: OpenCV 4.x (opencv-python)
- **Landmark Detection**: dlib 19.x with 68-point shape predictor model
- **Numerical Computation**: NumPy
- **Image Processing**: PIL/Pillow (optional, for advanced image operations)
- **Visualization**: Matplotlib (optional, for plotting landmarks)

### 5.4 External Dependencies
- **dlib 68-point shape predictor model**: File `shape_predictor_68_face_landmarks.dat` (download separately, ~100MB)
- **OpenCV Haar Cascade**: Built-in `haarcascade_frontalface_default.xml`

---

## 6. Interfaces

### 6.1 User Interface
- **Primary Interface**: Command-line interface (CLI)
- **Diagnostic Scripts**: `scripts/demo_hairline_steps.py` for step-by-step hairline detection visualization and disk output; `scripts/diagnose_hairline.py` for batch diagnostic figures
- **Landmark Evaluation Tool**: `scripts/landmark_evaluation.py` for interactive ground-truth labeling and batch evaluation
- **Usage**: `python visagism.py --input <image_path> [--output <output_dir>] [--visualize] [--save-viz] [--hairline]`
- **Demo Script Usage**: `python scripts/demo_hairline_steps.py --input <image_path> [--visualize]`
- **Landmark Evaluation Usage**:
  ```bash
  # Label a single image or directory
  python scripts/landmark_evaluation.py --mode label --input photos/face.jpg
  python scripts/landmark_evaluation.py --mode label --input ./photos/ --output ./ground_truth/

  # Evaluate predictions against ground truth
  python scripts/landmark_evaluation.py --mode evaluate \
      --predictions-dir ./predictions/ --ground-truth-dir ./ground_truth/ \
      --report evaluation_report.json
  ```
- **Example**:
  ```bash
  # Process single image
  python visagism.py --input photos/face.jpg --visualize --hairline

  # Run hairline diagnostic (headless, saves to output/<image_stem>/)
  python scripts/demo_hairline_steps.py --input photos/face.jpg
  ```

### 6.2 Input Format
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| --input | String (path) | Yes | Path to image file |
| --output | String (path) | No | Output directory for results (default: ./output) |
| --visualize | Flag | No | Display landmark visualization window |
| --save-viz | Flag | No | Save visualization to file without displaying |
| --hairline | Flag | No | Enable hairline detection visualization (dashed line) |
| --mode | String | Yes (landmark_evaluation.py) | Tool mode: `label` or `evaluate` |
| --input | String (path) | Yes (label mode) | Path to image file or directory |
| --output | String (path) | No (label mode) | Output directory for ground truth JSON (default: ./ground_truth) |
| --predictions-dir | String (path) | Yes (evaluate mode) | Directory containing prediction JSON files |
| --ground-truth-dir | String (path) | Yes (evaluate mode) | Directory containing ground truth JSON files |
| --report | String (path) | No (evaluate mode) | Path to save JSON evaluation report |

### 6.3 Output Format
- **Console Output**: Analysis summary with face shape and golden ratio comparisons
- **Report File**: `analysis_report_[timestamp].txt` containing full analysis
- **Visualization File**: `landmarks_[original_filename].jpg` with overlaid landmarks
- **Diagnostic Outputs** (from `scripts/demo_hairline_steps.py`):
  - `output/<image_stem>/step01_face_and_roi.png` through `step06_final_result.png`
  - `output/<image_stem>/data.json` — raw numerical hairline detection data
  - `output/<image_stem>/profiles.csv` — per-row intensity and gradient profiles
  - `output/<image_stem>/summary.txt` — human-readable text summary
- **Ground Truth Files** (from `scripts/landmark_evaluation.py --mode label`):
  - `ground_truth/<image_stem>_gt.json` — manually annotated 68 landmarks, hairline, and metadata
- **Evaluation Reports** (from `scripts/landmark_evaluation.py --mode evaluate`):
  - Console table with overall mean error, NME, per-region means, per-image breakdown, and skipped files
  - JSON report with `summary`, `per_image`, `per_region_overall`, `skipped_files`

### 6.4 Error Handling
- All errors displayed in clear, user-friendly messages
- No stack traces shown to end user (logged internally if needed)
- Exit codes: 0 = success, 1 = error (no face detected, invalid input, etc.)

---

## 7. Testing Strategy

### 7.1 Unit Tests
- **Coverage Target**: ≥85% code coverage
- **Framework**: pytest
- **Test Areas**:
  - Image input validation (valid/invalid formats, missing files)
  - Face detection with known test images
  - Landmark detection accuracy against manual annotations
  - Proportion calculations with known measurements
  - Visagist calculator: 50 tests covering `VisagismCalculator` initialization, validation, three reference block formulas, deviation computation with 10% threshold flagging, consensus averaging, `from_landmarks()` factory method, and all dataclasses (`FacialMeasurements`, `DeviationResult`, `ReferenceBlock`, `ConsensusResult`, `VisagismAnalysis`) — 100% line coverage
  - Face shape classification against labeled dataset
  - Golden ratio deviation calculations
  - Error handling scenarios

### 7.2 Integration Tests
- **End-to-end pipeline**: Input image → detection → landmarks → proportions → shape → golden ratio → report
- **Visualization**: Landmark overlay matches expected output

### 7.3 Validation Tests
- **Test Dataset**: 20+ diverse face images with manual annotations
  - Various face shapes (3+ per category)
  - Different ages, genders, ethnicities
  - Various lighting conditions and image qualities
- **Accuracy Metrics**:
  - Landmark detection: Compare to manual (x, y) coordinates (±5 pixels tolerance)
  - Face shape: Compare to manual classification
  - Proportions: Verify calculations manually

### 7.4 Demo Testing
- **Pre-demo checklist**: Verify on 5 test images before presentation
- **Fallback images**: Have 3 guaranteed-working images for live demo
- **Performance**: Verify processing time <5 seconds per image
- **Diagnostic validation**: Use `scripts/demo_hairline_steps.py` to inspect intermediate hairline detection outputs (step images, `data.json`, `profiles.csv`, `summary.txt`) and confirm algorithm behaviour on test images

### 7.5 Landmark Evaluation Tool Testing
- **Unit tests** for `LandmarkGroundTruth` serialization/deserialization (`tests/test_landmark_ground_truth.py`): save/load round-trip, validation of landmark count, padding/truncation of malformed input, region lookup, completion counting
- **Unit tests** for `LandmarkEvaluator` metrics (`tests/test_landmark_evaluator.py`): per-landmark Euclidean errors, NME calculation, inter-ocular distance computation, per-region mean errors, hairline absolute error, report aggregation, graceful handling of missing landmarks (-1, -1) and zero inter-ocular distance
- **Unit tests** for `LandmarkLabeler` state management (`tests/test_landmark_labeler.py`): navigation (next/prev landmark and image), reset to prediction, save path generation, ground truth resume, completion tracking
- **Integration tests**: End-to-end label → save → load → evaluate pipeline using temporary directories
- **Edge cases**: Malformed JSON, unmatched file pairs, missing hairline values, empty prediction/ground truth directories

### 7.6 Hairline Detection Evaluation

A controlled evaluation was performed on 2026-05-28 to compare two hairline detection implementations against manually-annotated ground truth (9 images, 68 landmarks + hairline per image).

#### 7.6.1 Methodology

| Method | Technique | Fallback |
|--------|-----------|----------|
| **Canny** (primary) | Canny edge detection + morphological closing on a 1-pixel-wide center strip above the eyebrows | Geometric estimate (superior third = medium third) |
| **No-Canny** (fallback) | Vertical intensity gradient + CLAHE enhancement on a narrow forehead ROI | Geometric estimate (superior third = medium third) |

Both methods share identical dlib 68-point landmarks; only the hairline estimation algorithm differs.

#### 7.6.2 68 Facial Landmark Results

The 68 facial landmarks are identical between both methods because both rely on the same dlib shape predictor.

| Metric | Value |
|--------|-------|
| Overall mean error | 1.8 px |
| Overall NME | 0.0044 |

#### 7.6.3 Hairline Detection Comparison

| Metric | No-Canny | Canny |
|--------|----------|-------|
| Mean hairline error | 100.3 px | 38.3 px |
| Median hairline error | 82.0 px | 49.0 px |
| Images within 3 px | 0/9 (0%) | 4/9 (44.4%) |
| Images within 10 px | 1/9 (11.1%) | 4/9 (44.4%) |
| Images within 50 px | 3/9 (33.3%) | 6/9 (66.7%) |

#### 7.6.4 Per-Image Breakdown

| Image | No-Canny Error | Canny Error | Winner |
|-------|----------------|-------------|--------|
| woman_10 | 25 px | **2 px** | Canny |
| woman_11 | **85 px** | 90 px | No-Canny |
| woman_12 | 107 px | **0 px** | Canny |
| woman_13 | 143 px | **0 px** | Canny |
| woman_6 | 46 px | 46 px | Tie |
| woman_7 | 62 px | **14 px** | Canny |
| woman_8 | 158 px | **29 px** | Canny |
| woman_9 | **137 px** | 164 px | No-Canny |
| woman_makeup_1 | 140 px | **0 px** | Canny |

#### 7.6.5 Key Findings

1. **Canny achieves 62% lower mean hairline error** (38.3 px vs 100.3 px).
2. **Canny wins on 6/9 images**, No-Canny wins on 2/9, and 1/9 is a tie.
3. **Canny achieves ≤3 px accuracy on 44% of images** vs 0% for No-Canny.
4. **No-Canny performs better when Canny over-detects edges** — false positives from hair strands, shadows, or textured backgrounds cause Canny to place the hairline too low.
5. **Both methods use identical dlib landmarks**; only the hairline detection step differs.

#### 7.6.6 Decision

- **Primary method**: Canny edge detection (lower mean error, higher consistency).
- **Fallback method**: No-Canny gradient method is retained for images where Canny produces false-positive edges.
- **Future work**: Adaptive Canny thresholds or a hybrid confidence score could further reduce the 38.3 px mean error.

---

## 8. Risks & Constraints

### 8.1 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| dlib shape predictor model file unavailable (100MB download) | High - Cannot detect landmarks | Provide alternative using OpenCV only; include download script in repo |
| Poor landmark detection on non-frontal or occluded faces | Medium - Reduced accuracy | Document limitations; warn user about photo requirements |
| Face shape classification accuracy below 85% | Medium - Assignment grade impact | Use ensemble of rules; manual tuning with test dataset |
| Processing time exceeds 5 seconds per image | Low - Demo inconvenience | Optimize code; use faster detection methods; pre-load models |
| Test dataset insufficient for validation | Medium - Cannot verify accuracy | Use public datasets (e.g., Labeled Faces in the Wild) for additional test cases |
| Assignment scope creep (adding too many features) | Medium - Miss deadline | Strictly follow Non-Goals section; focus on Must-priority requirements |
| Hairline detection accuracy variance across image conditions | Medium - Superior third measurement error | Canny is primary method (38.3 px mean error); retain No-Canny fallback for images with false-positive edges; document limitation that hairline is an estimate, not ground truth |

### 8.2 Constraints
- **Timeline**: Final presentation by May 28, 2026 (3 weeks from start)
- **Team Size**: 2 students (limited development resources)
- **Hardware**: Standard laptop computing power (no GPU acceleration required)
- **Assignment Level**: Targeting Level 1-2 (single solution or dataset acquisition with comparison)
- **Academic Integrity**: Must reference all non-original code/libraries explicitly (per assignment warning)
- **Technology**: Must use OpenCV (assignment requirement)
- **Deployment**: Must run locally (no cloud services for assignment submission)

### 8.3 Assumptions
- User will provide clear, frontal face photographs
- Single face per image (or largest face can be selected)
- dlib 68-point model file will be available for download
- Python 3.8+ and pip available on development machines
- Test images can be acquired or generated for validation

---

## 9. Appendices

### 9.1 Glossary
- **Visagism**: Science of personal image analysis through facial proportion study (from French "visage" = face)
- **68-Point Landmark Model**: Standard facial landmark detection model identifying 68 key points
- **Facial Thirds**: Division of face into three horizontal sections (superior, medium, inferior)
- **Golden Ratio (Phi)**: Mathematical ratio 1.618, considered aesthetically pleasing
- **Face Shape**: Classification based on facial geometry (Oval, Round, Square, Oblong, Heart, Triangle, Diamond)

### 9.2 Reference Images
- Example landmark positions: https://pandami.com.br/blog/visagismo-cabelo-guia-completo
- dlib 68-point model visualization: http://dlib.net/face_landmark_detection.py.html

### 9.3 Related Documents
- Computer Vision Assignment Specification (Universidade de Aveiro, 2025/2026)
- OpenCV Documentation: https://docs.opencv.org/
- dlib Documentation: http://dlib.net/

---

*End of Functional Specification*
