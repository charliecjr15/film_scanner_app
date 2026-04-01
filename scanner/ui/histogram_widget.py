from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget

class HistogramWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(180)
        self._hist = None

    def set_histogram(self, hist: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        self._hist = hist
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191c"))

        if self._hist is None:
            return

        hist_r, hist_g, hist_b = self._hist
        max_val = max(float(hist_r.max()), float(hist_g.max()), float(hist_b.max()), 1.0)

        width = self.width()
        height = self.height()
        bins = len(hist_r)

        def draw_hist(data: np.ndarray, color: QColor) -> None:
            pen = QPen(color, 1)
            painter.setPen(pen)
            prev_x = 0
            prev_y = height
            for i, v in enumerate(data):
                x = int(i * (width - 1) / max(1, bins - 1))
                y = height - int((float(v) / max_val) * (height - 4))
                painter.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y

        draw_hist(hist_r, QColor(255, 100, 100))
        draw_hist(hist_g, QColor(100, 255, 120))
        draw_hist(hist_b, QColor(120, 170, 255))