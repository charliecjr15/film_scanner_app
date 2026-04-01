from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from scanner.models.image_job import ImageJob
from scanner.core.pipeline import process_image
from scanner.core.image_io import save_image_jpeg, save_image_tiff


class ExportWorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int)
    error = Signal(str)


class ExportWorker(QRunnable):
    def __init__(
        self,
        jobs: list[ImageJob],
        export_dir: str,
        export_format: str,
        jpeg_quality: int
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.export_dir = export_dir
        self.export_format = export_format.lower()
        self.jpeg_quality = jpeg_quality
        self.signals = ExportWorkerSignals()

    @Slot()
    def run(self) -> None:
        exported = 0
        failed = 0

        try:
            for idx, job in enumerate(self.jobs, start=1):
                try:
                    image = process_image(job, preview=False)
                    stem = Path(job.source_path).stem + "_scan"

                    if self.export_format == "tiff":
                        out_path = str(Path(self.export_dir) / f"{stem}.tiff")
                        save_image_tiff(
                            out_path,
                            image,
                            output_profile_name=job.output_profile_name,
                            custom_output_icc_path=job.custom_output_icc_path or None,
                        )
                    else:
                        out_path = str(Path(self.export_dir) / f"{stem}.jpg")
                        save_image_jpeg(
                            out_path,
                            image,
                            quality=self.jpeg_quality,
                            output_profile_name=job.output_profile_name,
                            custom_output_icc_path=job.custom_output_icc_path or None,
                        )

                    exported += 1
                    self.signals.progress.emit(idx, len(self.jobs), Path(out_path).name)
                except Exception:
                    failed += 1
                    self.signals.progress.emit(idx, len(self.jobs), f"FAILED: {job.display_name()}")

            self.signals.finished.emit(exported, failed)
        except Exception:
            self.signals.error.emit(traceback.format_exc())