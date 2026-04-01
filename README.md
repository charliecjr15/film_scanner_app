# Film Scanner App

A desktop film scanning workflow app built with **Python** and **PySide6** for quickly turning camera-captured film frames into usable positive images.

## What it does

Film Scanner App is designed to make film conversion feel fast and dependable:

- import a film image in seconds
- auto-correct to a usable starting point
- make a few quick adjustments
- compare the result in the preview
- export confidently to JPEG or TIFF

The app currently supports:

- **Color negative** conversion
- **Black and white negative** conversion
- **Slide positive** workflow
- **Auto crop** and optional border inclusion
- **Manual crop** adjustment
- **Gray picker** for neutral balancing
- **Exposure, temperature, tint, contrast, saturation** controls
- **Black point / white point** controls
- **Sharpening**
- **Batch export**
- **Histogram preview**

## Project structure

```text
film_scanner_app/
├── main.py
├── requirements.txt
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
- A virtual environment is strongly recommended

## Install

Clone the repo:

```bash
git clone https://github.com/charliecjr15/film_scanner_app.git
cd film_scanner_app
```

Create and activate a virtual environment:

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
2. Select the film type:
   - `color_negative`
   - `bw_negative`
   - `slide_positive`
3. Let the app generate a preview.
4. Fine-tune with the control panel.
5. Export the current image or export the full queue.

## V4 goals

Version 4 focuses on a simple core workflow:

**Import → Auto-correct → Adjust → Compare → Export**

The V4 direction is intended to improve:

- more dependable default color conversion
- better midtone-based exposure normalization
- cleaner highlight and shadow handling
- fewer manual corrections per frame
- faster preview-to-export workflow

## Current export options

- JPEG
- TIFF

## Known limitations

This project is still evolving. Current limitations may include:

- color conversion still depends on capture consistency
- no stock-specific film profiles yet
- no advanced orange-mask modeling yet
- UI and auto-correction workflow are still being refined

## Roadmap ideas

- stronger border-aware correction
- improved film base estimation
- better compare mode / split view
- saved presets
- session save/load
- more robust batch workflow

## Contributing

This project is currently being actively shaped and refactored. If you want to contribute, open an issue or submit a pull request with a clear explanation of the change.

## License

Add a license file if you want to make reuse terms explicit.
