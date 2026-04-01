from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
    QCheckBox, QGroupBox, QHBoxLayout, QLineEdit
)

from scanner.core.negative import list_negative_presets
from scanner.core.color_management import list_output_profile_names


class FilmControls(QWidget):
    changed = Signal()
    export_clicked = Signal()
    export_all_clicked = Signal()
    reset_clicked = Signal()
    rotate_left_clicked = Signal()
    rotate_right_clicked = Signal()
    manual_crop_toggled = Signal(bool)
    gray_picker_toggled = Signal(bool)
    browse_custom_icc_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()

        root = QVBoxLayout(self)

        basic_box = QGroupBox("Adjustments")
        basic_layout = QVBoxLayout(basic_box)

        basic_layout.addWidget(QLabel("Film Type"))
        self.film_type = QComboBox()
        self.film_type.addItems(["color_negative", "bw_negative", "slide_positive"])
        self.film_type.currentIndexChanged.connect(lambda _=None: self.changed.emit())
        basic_layout.addWidget(self.film_type)

        basic_layout.addWidget(QLabel("Stock / Preset"))
        self.preset_name = QComboBox()
        self.preset_name.addItems(list_negative_presets())
        self.preset_name.currentIndexChanged.connect(lambda _=None: self.changed.emit())
        basic_layout.addWidget(self.preset_name)

        basic_layout.addWidget(QLabel("Output Profile"))
        self.output_profile = QComboBox()
        self.output_profile.addItems(list_output_profile_names())
        self.output_profile.currentIndexChanged.connect(lambda _=None: self.changed.emit())
        basic_layout.addWidget(self.output_profile)

        basic_layout.addWidget(QLabel("Custom Output ICC (optional)"))
        row = QHBoxLayout()
        self.custom_icc_path = QLineEdit()
        self.custom_icc_path.textChanged.connect(lambda _=None: self.changed.emit())
        self.browse_custom_icc_btn = QPushButton("Browse…")
        self.browse_custom_icc_btn.clicked.connect(self.browse_custom_icc_clicked.emit)
        row.addWidget(self.custom_icc_path, 1)
        row.addWidget(self.browse_custom_icc_btn)
        basic_layout.addLayout(row)

        self.auto_crop = QCheckBox("Auto Crop")
        self.auto_crop.setChecked(True)
        self.auto_crop.stateChanged.connect(lambda _=None: self.changed.emit())
        basic_layout.addWidget(self.auto_crop)

        self.include_border = QCheckBox("Include Border")
        self.include_border.stateChanged.connect(lambda _=None: self.changed.emit())
        basic_layout.addWidget(self.include_border)

        self.manual_crop = QCheckBox("Manual Crop")
        self.manual_crop.stateChanged.connect(self._emit_manual_crop_toggled)
        basic_layout.addWidget(self.manual_crop)

        self.gray_picker = QCheckBox("Gray Picker")
        self.gray_picker.stateChanged.connect(self._emit_gray_picker_toggled)
        basic_layout.addWidget(self.gray_picker)

        for label, attr, low, high, value in [
            ("Exposure", "exposure", -30, 30, 0),
            ("Temperature", "temp", -30, 30, 0),
            ("Tint", "tint", -30, 30, 0),
            ("Contrast", "contrast", -25, 25, 0),
            ("Saturation", "saturation", -25, 25, 0),
            ("Black Point", "black_point", 0, 30, 0),
            ("White Point", "white_point", 70, 100, 100),
            ("Sharpness", "sharpness", 0, 50, 25),
        ]:
            basic_layout.addWidget(QLabel(label))
            setattr(self, attr, self._make_slider(low, high, value, basic_layout))

        root.addWidget(basic_box)

        transform_box = QGroupBox("Transform")
        t_layout = QHBoxLayout(transform_box)
        self.rotate_left_btn = QPushButton("Rotate -90°")
        self.rotate_right_btn = QPushButton("Rotate +90°")
        self.rotate_left_btn.clicked.connect(self.rotate_left_clicked.emit)
        self.rotate_right_btn.clicked.connect(self.rotate_right_clicked.emit)
        t_layout.addWidget(self.rotate_left_btn)
        t_layout.addWidget(self.rotate_right_btn)
        root.addWidget(transform_box)

        actions_box = QGroupBox("Actions")
        a_layout = QVBoxLayout(actions_box)
        self.reset_btn = QPushButton("Reset Current")
        self.export_btn = QPushButton("Export Current")
        self.export_all_btn = QPushButton("Export All")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        self.export_btn.clicked.connect(self.export_clicked.emit)
        self.export_all_btn.clicked.connect(self.export_all_clicked.emit)
        a_layout.addWidget(self.reset_btn)
        a_layout.addWidget(self.export_btn)
        a_layout.addWidget(self.export_all_btn)
        root.addWidget(actions_box)
        root.addStretch(1)

    def _make_slider(self, low: int, high: int, value: int, layout: QVBoxLayout) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(low)
        slider.setMaximum(high)
        slider.setValue(value)
        slider.valueChanged.connect(lambda _=None: self.changed.emit())
        layout.addWidget(slider)
        return slider

    def _emit_manual_crop_toggled(self, state: int) -> None:
        self.manual_crop_toggled.emit(bool(state))
        self.changed.emit()

    def _emit_gray_picker_toggled(self, state: int) -> None:
        self.gray_picker_toggled.emit(bool(state))
        self.changed.emit()