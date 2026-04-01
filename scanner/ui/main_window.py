from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QHBoxLayout, QVBoxLayout,
    QMessageBox, QLabel, QSplitter, QComboBox, QGroupBox, QScrollArea
)

from scanner.settings_manager import SettingsManager
from scanner.models.image_job import ImageJob
from scanner.models.document_state import DocumentState
from scanner.ui.preview_widget import PreviewWidget
from scanner.ui.film_controls import FilmControls
from scanner.ui.queue_panel import QueuePanel
from scanner.ui.histogram_widget import HistogramWidget
from scanner.workers.preview_worker import PreviewWorker
from scanner.workers.export_worker import ExportWorker
from scanner.core.image_io import RAW_EXTS

SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff"
} | RAW_EXTS

IMAGE_DIALOG_FILTER = (
    "Supported Images (*.jpg *.jpeg *.png *.tif *.tiff "
    "*.dng *.nef *.cr2 *.cr3 *.crw *.arw *.raf *.rw2 *.orf *.pef *.srw *.erf *.kdc *.mos *.3fr *.iiq *.mrw *.x3f);;"
    "JPEG/PNG/TIFF (*.jpg *.jpeg *.png *.tif *.tiff);;"
    "RAW Files (*.dng *.nef *.cr2 *.cr3 *.crw *.arw *.raf *.rw2 *.orf *.pef *.srw *.erf *.kdc *.mos *.3fr *.iiq *.mrw *.x3f);;"
    "All Files (*)"
)

class MainWindow(QMainWindow):
    def __init__(self, settings: SettingsManager) -> None:
        super().__init__()
        self.settings = settings
        self.state = DocumentState()
        self.thread_pool = QThreadPool.globalInstance()
        self.preview_request_id = 0

        self.setWindowTitle("Film Scanner V3")
        self.resize(
            int(self.settings.get("window_width", 1280)),
            int(self.settings.get("window_height", 820))
        )

        self.setMinimumSize(980, 640)

        container = QWidget()
        self.setCentralWidget(container)
        root = QHBoxLayout(container)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        self.queue_panel = QueuePanel()
        self.preview = PreviewWidget()
        self.controls = FilmControls()

        self.queue_panel.setMinimumWidth(180)
        self.queue_panel.setMaximumWidth(260)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(self.controls)

        hist_group = QGroupBox("Histogram")
        hist_layout = QVBoxLayout(hist_group)
        self.histogram_widget = HistogramWidget()
        self.histogram_widget.setMinimumHeight(120)
        self.histogram_widget.setMaximumHeight(180)
        hist_layout.addWidget(self.histogram_widget)
        right_layout.addWidget(hist_group)

        export_group = QGroupBox("Batch Export Format")
        export_layout = QVBoxLayout(export_group)
        self.batch_export_format = QComboBox()
        self.batch_export_format.addItems(["jpeg", "tiff"])
        export_layout.addWidget(self.batch_export_format)
        right_layout.addWidget(export_group)

        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setMinimumWidth(260)
        right_scroll.setMaximumWidth(340)
        right_scroll.setWidget(right_content)

        splitter.addWidget(self.queue_panel)
        splitter.addWidget(self.preview)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 760, 300])

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._start_preview_worker)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.queue_panel.add_files_clicked.connect(self.add_files)
        self.queue_panel.add_folder_clicked.connect(self.add_folder)
        self.queue_panel.remove_clicked.connect(self.remove_selected)
        self.queue_panel.clear_clicked.connect(self.clear_queue)
        self.queue_panel.selection_changed.connect(self.select_index)

        self.controls.changed.connect(self.on_controls_changed)
        self.controls.export_clicked.connect(self.export_current)
        self.controls.export_all_clicked.connect(self.export_all)
        self.controls.reset_clicked.connect(self.reset_current)
        self.controls.rotate_left_clicked.connect(self.rotate_left)
        self.controls.rotate_right_clicked.connect(self.rotate_right)
        self.controls.manual_crop_toggled.connect(self._on_manual_crop_toggled)
        self.controls.gray_picker_toggled.connect(self._on_gray_picker_toggled)

        self.preview.crop_changed.connect(self.on_manual_crop_changed)
        self.preview.gray_point_chosen.connect(self.on_gray_point_chosen)

    def closeEvent(self, event) -> None:
        self.settings.set("window_width", self.width())
        self.settings.set("window_height", self.height())
        self.settings.save()
        super().closeEvent(event)

    def _refresh_queue_names(self) -> None:
        self.queue_panel.set_items([job.display_name() for job in self.state.queue])
        if self.state.selected_index >= 0:
            self.queue_panel.set_current_index(self.state.selected_index)

    def _default_job(self, path: str) -> ImageJob:
        return ImageJob(
            source_path=path,
            film_type=self.settings.get("default_film_type", "color_negative"),
            auto_crop_enabled=bool(self.settings.get("default_auto_crop", True)),
            include_border=bool(self.settings.get("default_include_border", False)),
            exposure=float(self.settings.get("default_exposure", 0.0)),
            temp=float(self.settings.get("default_temp", 0.0)),
            tint=float(self.settings.get("default_tint", 0.0)),
            contrast=float(self.settings.get("default_contrast", 0.0)),
            saturation=float(self.settings.get("default_saturation", 0.0)),
            black_point=float(self.settings.get("default_black_point", 0.0)),
            white_point=float(self.settings.get("default_white_point", 1.0)),
            sharpness=float(self.settings.get("default_sharpness", 0.25)),
        )

    def current_job(self) -> ImageJob | None:
        return self.state.current_job

    def add_files(self) -> None:
        start_dir = self.settings.get("last_open_dir", "")
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image Files",
            start_dir,
            IMAGE_DIALOG_FILTER
        )
        if not paths:
            return

        self.settings.set("last_open_dir", str(Path(paths[0]).parent))
        for path in paths:
            self.state.queue.append(self._default_job(path))

        if self.state.selected_index == -1 and self.state.queue:
            self.state.selected_index = 0

        self._refresh_queue_names()
        self.push_job_to_controls()
        self.schedule_preview()
        self.status_label.setText(f"Added {len(paths)} file(s)")

    def add_folder(self) -> None:
        start_dir = self.settings.get("last_open_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Add Folder", start_dir)
        if not folder:
            return

        self.settings.set("last_open_dir", folder)
        files = sorted([
            str(p) for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ])

        for path in files:
            self.state.queue.append(self._default_job(path))

        if self.state.selected_index == -1 and self.state.queue:
            self.state.selected_index = 0

        self._refresh_queue_names()
        self.push_job_to_controls()
        self.schedule_preview()
        self.status_label.setText(f"Added {len(files)} file(s)")

    def remove_selected(self) -> None:
        idx = self.state.selected_index
        if idx < 0 or idx >= len(self.state.queue):
            return

        del self.state.queue[idx]
        if not self.state.queue:
            self.state.selected_index = -1
            self.preview.clear()
            self.preview.setText("Import images to begin")
            self.histogram_widget.set_histogram(None)
        else:
            self.state.selected_index = min(idx, len(self.state.queue) - 1)

        self._refresh_queue_names()
        self.push_job_to_controls()
        self.schedule_preview()

    def clear_queue(self) -> None:
        self.state.queue.clear()
        self.state.selected_index = -1
        self._refresh_queue_names()
        self.preview.clear()
        self.preview.setText("Import images to begin")
        self.histogram_widget.set_histogram(None)
        self.status_label.setText("Queue cleared")

    def select_index(self, index: int) -> None:
        if index < 0 or index >= len(self.state.queue):
            return
        self.state.selected_index = index
        self.push_job_to_controls()
        self.schedule_preview()

    def push_job_to_controls(self) -> None:
        job = self.current_job()
        if job is None:
            return

        self.controls.film_type.setCurrentText(job.film_type)
        self.controls.auto_crop.setChecked(job.auto_crop_enabled)
        self.controls.include_border.setChecked(job.include_border)
        self.controls.exposure.setValue(int(job.exposure * 10))
        self.controls.temp.setValue(int(job.temp * 10))
        self.controls.tint.setValue(int(job.tint * 10))
        self.controls.contrast.setValue(int(job.contrast * 20))
        self.controls.saturation.setValue(int(job.saturation * 20))
        self.controls.black_point.setValue(int(job.black_point * 100))
        self.controls.white_point.setValue(int(job.white_point * 100))
        self.controls.sharpness.setValue(int(job.sharpness * 100))

        self.preview.set_crop_rect_normalized(job.normalized_crop_rect)
        self.preview.set_gray_point_normalized(job.gray_pick_normalized)

    def pull_controls_to_job(self) -> None:
        job = self.current_job()
        if job is None:
            return

        job.film_type = self.controls.film_type.currentText()
        job.auto_crop_enabled = self.controls.auto_crop.isChecked()
        job.include_border = self.controls.include_border.isChecked()
        job.exposure = self.controls.exposure.value() / 10.0
        job.temp = self.controls.temp.value() / 10.0
        job.tint = self.controls.tint.value() / 10.0
        job.contrast = self.controls.contrast.value() / 20.0
        job.saturation = self.controls.saturation.value() / 20.0
        job.black_point = self.controls.black_point.value() / 100.0
        job.white_point = self.controls.white_point.value() / 100.0
        job.sharpness = self.controls.sharpness.value() / 100.0

    def on_controls_changed(self) -> None:
        if self.current_job() is None:
            return
        self.pull_controls_to_job()
        self.schedule_preview()

    def rotate_left(self) -> None:
        job = self.current_job()
        if job is None:
            return
        job.rotation = (job.rotation - 90) % 360
        self.schedule_preview()

    def rotate_right(self) -> None:
        job = self.current_job()
        if job is None:
            return
        job.rotation = (job.rotation + 90) % 360
        self.schedule_preview()

    def reset_current(self) -> None:
        job = self.current_job()
        if job is None:
            return

        path = job.source_path
        self.state.queue[self.state.selected_index] = self._default_job(path)
        self.push_job_to_controls()
        self.schedule_preview()
        self.status_label.setText("Current image reset")

    def _on_manual_crop_toggled(self, enabled: bool) -> None:
        self.preview.set_manual_crop_enabled(enabled)
        if enabled:
            self.controls.gray_picker.setChecked(False)

    def _on_gray_picker_toggled(self, enabled: bool) -> None:
        self.preview.set_gray_picker_enabled(enabled)
        if enabled:
            self.controls.manual_crop.setChecked(False)

    def on_manual_crop_changed(self, norm_rect: tuple[float, float, float, float]) -> None:
        job = self.current_job()
        if job is None:
            return

        job.normalized_crop_rect = norm_rect
        job.auto_crop_enabled = False
        self.controls.auto_crop.setChecked(False)
        self.schedule_preview()

    def on_gray_point_chosen(self, norm_point: tuple[float, float]) -> None:
        job = self.current_job()
        if job is None:
            return

        job.gray_pick_normalized = norm_point
        self.preview.set_gray_point_normalized(norm_point)
        self.controls.gray_picker.setChecked(False)
        self.schedule_preview()

    def schedule_preview(self) -> None:
        if self.current_job() is None:
            return
        self.preview_request_id += 1
        self.preview_timer.start(120)
        self.status_label.setText("Rendering preview...")

    def _start_preview_worker(self) -> None:
        job = self.current_job()
        if job is None:
            return

        self.pull_controls_to_job()
        worker_job = copy.deepcopy(job)
        request_id = self.preview_request_id

        worker = PreviewWorker(worker_job, request_id)
        worker.signals.finished.connect(self._on_preview_finished)
        worker.signals.error.connect(self._on_preview_error)
        self.thread_pool.start(worker)

    def _on_preview_finished(self, image, hist, request_id: int) -> None:
        if request_id != self.preview_request_id:
            return

        self.preview.set_image(image)
        job = self.current_job()
        if job is not None:
            self.preview.set_crop_rect_normalized(job.normalized_crop_rect)
            self.preview.set_gray_point_normalized(job.gray_pick_normalized)

        self.histogram_widget.set_histogram(hist)
        self.status_label.setText("Preview ready")

    def _on_preview_error(self, trace: str, request_id: int) -> None:
        if request_id != self.preview_request_id:
            return
        QMessageBox.critical(self, "Preview Error", trace)

    def export_current(self) -> None:
        job = self.current_job()
        if job is None:
            return

        self.pull_controls_to_job()

        start_dir = self.settings.get("last_export_dir", str(Path(job.source_path).parent))
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Current Image",
            str(Path(start_dir) / f"{Path(job.source_path).stem}_scan"),
            "JPEG (*.jpg);;TIFF (*.tiff)"
        )
        if not path:
            return

        export_dir = str(Path(path).parent)
        self.settings.set("last_export_dir", export_dir)
        self.settings.save()

        export_format = "jpeg"
        suffix = Path(path).suffix.lower()
        if suffix in {".tif", ".tiff"} or "TIFF" in selected_filter:
            export_format = "tiff"

        worker = ExportWorker(
            jobs=[copy.deepcopy(job)],
            export_dir=export_dir,
            export_format=export_format,
            jpeg_quality=int(self.settings.get("jpeg_quality", 95))
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self.thread_pool.start(worker)
        self.status_label.setText("Exporting current image...")

    def export_all(self) -> None:
        if not self.state.queue:
            return

        start_dir = self.settings.get("last_export_dir", str(Path(self.state.queue[0].source_path).parent))
        export_dir = QFileDialog.getExistingDirectory(self, "Export All To Folder", start_dir)
        if not export_dir:
            return

        self.settings.set("last_export_dir", export_dir)
        self.settings.save()

        jobs = [copy.deepcopy(job) for job in self.state.queue]
        worker = ExportWorker(
            jobs=jobs,
            export_dir=export_dir,
            export_format=self.batch_export_format.currentText(),
            jpeg_quality=int(self.settings.get("jpeg_quality", 95))
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self.thread_pool.start(worker)
        self.status_label.setText("Exporting all images...")

    def _on_export_progress(self, idx: int, total: int, name: str) -> None:
        self.status_label.setText(f"Export {idx}/{total}: {name}")

    def _on_export_finished(self, exported: int, failed: int) -> None:
        if failed:
            QMessageBox.warning(self, "Export Complete", f"Exported {exported} image(s), failed {failed}.")
        else:
            QMessageBox.information(self, "Export Complete", f"Exported {exported} image(s).")
        self.status_label.setText(f"Export complete: {exported} exported, {failed} failed")

    def _on_export_error(self, trace: str) -> None:
        QMessageBox.critical(self, "Export Error", trace)