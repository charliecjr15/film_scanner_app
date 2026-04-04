# Film Scanner App (Current Version)

## Overview

Film Scanner App is a desktop application designed to convert camera-scanned film negatives into positive images. It provides a GUI-based workflow for previewing, processing, and exporting scanned film frames.

This version represents an **experimental pipeline** focused on automated negative inversion and batch processing, with early-stage attempts at film-like color rendering.

---

## ⚠️ Current State (Important)

This version is **not production-ready** and does **not match lab-quality tools (e.g., Negative Lab Pro)**.

Known characteristics:

* Inconsistent color reproduction across frames
* Weak handling of orange mask removal
* Border/perforation contamination affecting results
* Over-reliance on per-frame auto-adjustments
* Limited roll-level consistency

This README documents the system as it exists today.

---

## Features

### Core Functionality

* Load and preview scanned film images
* Batch processing workflow
* Negative → positive conversion
* Histogram generation
* Image export (TIFF/JPEG)

### UI Components

* Main preview window
* Film controls panel
* Batch queue handling
* Export controls

### Processing Pipeline (Current)

1. Image load (assumed linear or near-linear input)
2. Frame detection (basic / partial)
3. Negative inversion
4. Color balancing (auto-based)
5. Tone adjustments
6. Output rendering

---

## Project Structure

```
film_scanner_app/
│
├── main.py
├── scanner/
│   ├── app_controller.py
│   │
│   ├── core/
│   │   ├── pipeline.py
│   │   ├── negative.py
│   │   ├── frame_detector.py
│   │
│   ├── workers/
│   │   ├── preview_worker.py
│   │
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── film_controls.py
│   │
│   └── utils/
│
└── assets/
```

---

## Key Modules

### `pipeline.py`

Main processing pipeline:

* Coordinates image transformation
* Calls frame detection and negative conversion
* Produces final preview/output

### `negative.py`

Handles:

* Negative inversion logic
* Color balancing attempts
* Preset-based adjustments (incomplete)

### `frame_detector.py`

Responsible for:

* Detecting film frame boundaries
* (Currently unreliable for complex borders/perforations)

### `preview_worker.py`

* Runs processing in background thread
* Prevents UI blocking during preview

### UI Layer

* `main_window.py`: Main application window
* `film_controls.py`: Controls for processing parameters

---

## Current Limitations

### 1. Color Accuracy

* No true film stock modeling
* No roll-level calibration
* Orange mask removal is unstable

### 2. Frame Detection

* Borders and perforations bleed into analysis
* Light leaks affect exposure calculations

### 3. Processing Strategy

* Per-frame auto adjustments cause inconsistency
* No shared calibration between images

### 4. Highlight Handling

* Clipping and washed highlights in bright areas

### 5. Shadow Handling

* Loss of detail or muddy tones

---

## Known Errors

Common issues encountered:

### Import Errors

```
ImportError: cannot import name 'detect_film_frame'
ImportError: cannot import name 'list_negative_presets'
```

Cause:

* Missing or outdated functions in core modules

---

## Dependencies

* Python 3.10+
* OpenCV
* NumPy
* PyQt / PySide (depending on build)

---

## Usage

Run the app:

```bash
python main.py
```

Typical workflow:

1. Load images
2. Preview conversion
3. Adjust controls (limited effect)
4. Export results

---

## Design Philosophy (Current Version)

This version attempts:

* Fully automatic negative conversion
* Minimal user interaction
* Fast preview feedback

However, this approach leads to:

* Inconsistent results
* Poor handling of real-world film variations

---

## Future Direction (Planned Rewrite)

The next version will move toward:

* Roll-based calibration
* LUT-based inversion
* Proper orange mask handling
* Edge-aware border rejection
* Film stock profiles
* Deterministic (non-chaotic) processing

---

## Summary

This version is:

* A working prototype
* A foundation for UI and workflow
* Not suitable for final-quality scans

It should be treated as:

> A stepping stone toward a rebuilt, production-quality film scanning engine.

---

## Author Notes

This project is actively evolving.
The current implementation highlights the challenges of:

* Film negative inversion
* Color science
* Robust automation

A full redesign of the processing pipeline is planned.
