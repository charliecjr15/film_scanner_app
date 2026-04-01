from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton, QHBoxLayout

class QueuePanel(QWidget):
    selection_changed = Signal(int)
    add_files_clicked = Signal()
    add_folder_clicked = Signal()
    remove_clicked = Signal()
    clear_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Queue"))

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.selection_changed.emit)
        layout.addWidget(self.list_widget, 1)

        row1 = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.add_folder_btn = QPushButton("Add Folder")
        row1.addWidget(self.add_files_btn)
        row1.addWidget(self.add_folder_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear")
        row2.addWidget(self.remove_btn)
        row2.addWidget(self.clear_btn)
        layout.addLayout(row2)

        self.add_files_btn.clicked.connect(self.add_files_clicked.emit)
        self.add_folder_btn.clicked.connect(self.add_folder_clicked.emit)
        self.remove_btn.clicked.connect(self.remove_clicked.emit)
        self.clear_btn.clicked.connect(self.clear_clicked.emit)

    def set_items(self, names: list[str]) -> None:
        self.list_widget.clear()
        self.list_widget.addItems(names)

    def set_current_index(self, index: int) -> None:
        self.list_widget.setCurrentRow(index)