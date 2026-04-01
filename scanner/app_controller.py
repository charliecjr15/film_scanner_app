import sys
from PySide6.QtWidgets import QApplication
from scanner.settings_manager import SettingsManager
from scanner.ui.main_window import MainWindow

DARK_QSS = """
QWidget {
    background-color: #1e1f22;
    color: #e8e8e8;
    font-size: 10pt;
}
QMainWindow, QDialog {
    background-color: #1e1f22;
}
QLabel {
    color: #e8e8e8;
}
QPushButton {
    background-color: #2d3138;
    border: 1px solid #3b4048;
    border-radius: 6px;
    padding: 8px 10px;
}
QPushButton:hover {
    background-color: #373c45;
}
QPushButton:pressed {
    background-color: #272b31;
}
QComboBox, QListWidget, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #25282d;
    border: 1px solid #3b4048;
    border-radius: 6px;
    padding: 5px;
}
QSlider::groove:horizontal {
    border: 1px solid #3b4048;
    height: 6px;
    background: #2b2f36;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #9fb7ff;
    border: 1px solid #6f87c8;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QListWidget::item:selected {
    background-color: #384050;
    color: #ffffff;
}
QGroupBox {
    border: 1px solid #353942;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QStatusBar {
    background-color: #191b1e;
}
QCheckBox {
    spacing: 8px;
}
"""

def run_app() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Film Scanner")
    app.setOrganizationName("Film Scanner Co")

    settings = SettingsManager()
    if settings.get("theme", "dark") == "dark":
        app.setStyleSheet(DARK_QSS)

    window = MainWindow(settings)
    window.show()

    sys.exit(app.exec())