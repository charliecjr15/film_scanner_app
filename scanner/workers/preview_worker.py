from __future__ import annotations

import traceback
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from scanner.models.image_job import ImageJob
from scanner.core.pipeline import process_image_and_histogram

class PreviewWorkerSignals(QObject):
    finished = Signal(object, object, int)   # image, histogram, request_id
    error = Signal(str, int)

class PreviewWorker(QRunnable):
    def __init__(self, job: ImageJob, request_id: int) -> None:
        super().__init__()
        self.job = job
        self.request_id = request_id
        self.signals = PreviewWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            image, hist = process_image_and_histogram(self.job, preview=True)
            self.signals.finished.emit(image, hist, self.request_id)
        except Exception:
            self.signals.error.emit(traceback.format_exc(), self.request_id)