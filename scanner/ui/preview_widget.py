from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel, QSizePolicy

def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    h, w, ch = rgb8.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb8.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

class PreviewWidget(QLabel):
    crop_changed = Signal(tuple)          # normalized rect
    gray_point_chosen = Signal(tuple)     # normalized point

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)

        # Reduced minimum size so the app fits smaller screens better
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("Import images to begin")

        self._pixmap: QPixmap | None = None
        self._scaled_pixmap: QPixmap | None = None
        self._display_rect = QRect()
        self._image_shape: tuple[int, int] | None = None

        self.manual_crop_enabled = False
        self.gray_picker_enabled = False

        self._dragging = False
        self._drag_start = QPoint()
        self._crop_rect_widget = QRect()
        self._gray_point_widget: QPoint | None = None

    def set_image(self, image: np.ndarray) -> None:
        self._image_shape = (image.shape[1], image.shape[0])
        self._pixmap = numpy_to_qpixmap(image)
        self._update_scaled()
        self.update()

    def set_manual_crop_enabled(self, enabled: bool) -> None:
        self.manual_crop_enabled = enabled
        if enabled:
            self.gray_picker_enabled = False
        self.update()

    def set_gray_picker_enabled(self, enabled: bool) -> None:
        self.gray_picker_enabled = enabled
        if enabled:
            self.manual_crop_enabled = False
        self.update()

    def set_crop_rect_normalized(self, norm_rect: tuple[float, float, float, float] | None) -> None:
        if self._display_rect.width() <= 0 or self._display_rect.height() <= 0 or norm_rect is None:
            self._crop_rect_widget = QRect()
            self.update()
            return

        x = self._display_rect.x() + int(norm_rect[0] * self._display_rect.width())
        y = self._display_rect.y() + int(norm_rect[1] * self._display_rect.height())
        w = int(norm_rect[2] * self._display_rect.width())
        h = int(norm_rect[3] * self._display_rect.height())
        self._crop_rect_widget = QRect(x, y, w, h)
        self.update()

    def set_gray_point_normalized(self, norm_pt: tuple[float, float] | None) -> None:
        if norm_pt is None or not self._display_rect.isValid():
            self._gray_point_widget = None
            self.update()
            return

        x = self._display_rect.x() + int(norm_pt[0] * self._display_rect.width())
        y = self._display_rect.y() + int(norm_pt[1] * self._display_rect.height())
        self._gray_point_widget = QPoint(x, y)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None:
            return

        self._scaled_pixmap = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        x = (self.width() - self._scaled_pixmap.width()) // 2
        y = (self.height() - self._scaled_pixmap.height()) // 2
        self._display_rect = QRect(x, y, self._scaled_pixmap.width(), self._scaled_pixmap.height())
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        if self._scaled_pixmap is None:
            return

        painter = QPainter(self)
        painter.drawPixmap(self._display_rect.topLeft(), self._scaled_pixmap)

        if self._crop_rect_widget.isValid():
            pen = QPen(QColor(0, 220, 180), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._crop_rect_widget)

        if self._gray_point_widget is not None:
            pen = QPen(QColor(255, 220, 90), 2)
            painter.setPen(pen)
            x = self._gray_point_widget.x()
            y = self._gray_point_widget.y()
            painter.drawEllipse(QPoint(x, y), 8, 8)
            painter.drawLine(x - 12, y, x + 12, y)
            painter.drawLine(x, y - 12, x, y + 12)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()

        if self.gray_picker_enabled and self._display_rect.contains(pos):
            nx = (pos.x() - self._display_rect.x()) / max(1, self._display_rect.width())
            ny = (pos.y() - self._display_rect.y()) / max(1, self._display_rect.height())
            self.gray_point_chosen.emit((float(nx), float(ny)))
            return

        if self.manual_crop_enabled and self._display_rect.contains(pos):
            self._dragging = True
            self._drag_start = pos
            self._crop_rect_widget = QRect(self._drag_start, self._drag_start)
            self.update()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.manual_crop_enabled and self._dragging:
            current = event.position().toPoint()
            rect = QRect(self._drag_start, current).normalized()
            rect = rect.intersected(self._display_rect)
            self._crop_rect_widget = rect
            self.update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.manual_crop_enabled and self._dragging:
            self._dragging = False
            rect = self._crop_rect_widget.intersected(self._display_rect)
            if rect.width() >= 5 and rect.height() >= 5:
                nx = (rect.x() - self._display_rect.x()) / max(1, self._display_rect.width())
                ny = (rect.y() - self._display_rect.y()) / max(1, self._display_rect.height())
                nw = rect.width() / max(1, self._display_rect.width())
                nh = rect.height() / max(1, self._display_rect.height())
                self.crop_changed.emit((float(nx), float(ny), float(nw), float(nh)))
                return

        super().mouseReleaseEvent(event)