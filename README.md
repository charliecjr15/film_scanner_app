# Film Scanner App

A desktop film scanning workflow app built with **Python** and **PySide6** for converting camera-captured film frames into usable positive images with a fast preview-to-export workflow.

## Overview

Film Scanner App is designed around one simple loop:

**Import → Auto-correct → Adjust → Compare → Export**

The goal is to make film conversion feel fast, dependable, and easy to use. The app is built for users who want a strong automatic starting point, then only a few manual tweaks before export.

## Current feature set

### Core workflow
- Import individual image files or entire folders
- Queue-based workflow for multiple frames
- Fast preview rendering
- Export current image or batch export the full queue

### Film support
- Color negative conversion
- Black and white negative conversion
- Slide positive workflow

### Auto-correction and processing
- Automatic frame detection and auto crop
- Optional border inclusion
- Content-mask estimation for safer image statistics
- Midtone-based exposure normalization
- Automatic color balancing
- Gray picker for neutral reference correction
- Filmic contrast finish
- Highlight and shadow protection
- Sharpening

### Stock-aware negative presets
- Balanced
- Neutral Lab
- Kodak Gold
- Kodak Portra 400
- Fuji 400H
- CineStill 800T

### Color management
- Standard images with embedded ICC profiles are converted into the working space
- RAW files are developed into **sRGB**
- Working pipeline is managed in **sRGB**
- JPEG and TIFF exports embed an **sRGB ICC profile**

### User controls
- Film type
- Stock / preset selection
- Exposure
- Temperature
- Tint
- Contrast
- Saturation
- Black point
- White point
- Sharpness
- Rotation
- Manual crop
- Gray picker

### Output
- JPEG export
- TIFF export
- Embedded sRGB ICC profile in exported files

## Project structure

```text
film_scanner_app/
├── main.py
├── requirements.txt
├── README.md
├── config/
│   └── default_settings.json
└── scanner/
    ├── app_controller.py
    ├── settings_manager.py
    ├── core/
    │   ├── color.py
    │   ├── frame_detector.py
    │   ├── histogram.py
    │   ├── image_io.py
    │   ├── negative.py
    │   ├── pipeline.py
    │   ├── sharpening.py
    │   ├── tone.py
    │   ├── transforms.py
    │   └── utils.py
    ├── models/
    │   ├── document_state.py
    │   └── image_job.py
    ├── ui/
    │   ├── film_controls.py
    │   ├── histogram_widget.py
    │   ├── main_window.py
    │   ├── preview_widget.py
    │   └── queue_panel.py
    └── workers/
        ├── export_worker.py
        └── preview_worker.py
```

## Requirements

- Python **3.11+** recommended
- macOS, Windows, or Linux
- A virtual environment is recommended

## Installation

Clone the repository:

```bash
git clone https://github.com/charliecjr15/film_scanner_app.git
cd film_scanner_app
```

Create and activate a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python main.py
```

## Basic workflow

1. Add image files or a folder.
2. Select the correct film type:
   - `color_negative`
   - `bw_negative`
   - `slide_positive`
3. Choose a stock preset if you are working with color negative film.
4. Let the app generate a preview.
5. Fine-tune with the controls if needed.
6. Use compare/preview tools in the interface.
7. Export the current image or export the full queue.

## Processing pipeline summary

The current processing flow is built around a practical, scanner-style workflow:

1. Read image into the working color space
2. Apply transforms such as rotation and flips
3. Detect frame and resolve crop
4. Estimate content mask
5. Convert film image:
   - invert negative if needed
   - apply stock-aware base estimation for color negatives
6. Normalize exposure around midtones
7. Auto-balance color
8. Apply optional gray picker correction
9. Apply user adjustments
10. Apply highlight/shadow protection and filmic contrast
11. Export with embedded output ICC profile

## Supported input formats

Typical standard image support includes:
- JPEG
- PNG
- TIFF

RAW support depends on `rawpy` and the installed LibRaw support. Common RAW extensions may include:
- DNG
- NEF
- CR2
- CR3
- ARW
- RAF
- RW2
- ORF
- PEF
- SRW
- and others defined in the project

## V4 / V4.5 direction

The current version focuses on:

- faster preview-to-export workflow
- stronger default auto-correction
- safer content-aware statistics
- better base estimation for color negatives
- stock-aware starting looks
- color-managed processing and output ICC embedding

## Known limitations

This project is still evolving. Current limitations may include:

- stock presets are heuristic and not lab-calibrated
- no true spectral orange-mask removal yet
- no custom user-loaded ICC profile selection yet
- no per-stock sensitometric calibration yet
- results still depend heavily on capture consistency, lighting, and white balance discipline

## Roadmap ideas

Potential next upgrades:

- selectable external ICC profiles
- more advanced output color spaces
- better compare tools / split view
- saved presets
- session save/load
- stronger batch workflow
- more film-stock-specific tone and color packs
- improved border-aware masking and crop logic

## Contributing

This project is actively evolving. If you want to contribute:

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Commit clearly
5. Open a pull request with a concise explanation

EXAMPLES

Film_Scanner_App
<img width="512" height="345" alt="Screenshot 2026-04-02 at 10 11 58" src="https://github.com/user-attachments/assets/04183aed-0687-4ee9-aa3b-7ddc9b07ec3b" />

Negitave Lab Pro
<img width="312" height="471" alt="Screenshot 2026-04-02 at 10 07 31" src="https://github.com/user-attachments/assets/a4bfc611-064e-4328-9729-8b551f94c665" />

## License

Add a license file if you want to define reuse, modification, and distribution terms clearly.
