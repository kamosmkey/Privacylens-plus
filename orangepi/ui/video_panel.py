"""Reusable video display panel for the thermal pose UI."""

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout


class VideoPanel(QFrame):
    def __init__(self, title: str, empty_text: str, ui_scale=1.0):
        super().__init__()
        px = lambda value: max(1, round(value * ui_scale))
        self.setObjectName("videoPanel")
        self.title = QLabel(title)
        self.title.setObjectName("panelTitle")
        self.image = QLabel(empty_text)
        self.image.setObjectName("videoImage")
        self.image.setAlignment(Qt.AlignCenter)
        # Let the grid determine panel proportions at every window size. Using
        # Ignored prevents a pixmap's native size from expanding the layout.
        self.image.setMinimumSize(1, 1)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._last_image = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(px(12), px(12), px(12), px(12))
        layout.setSpacing(px(10))
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)

    def set_ui_scale(self, scale):
        px = lambda value: max(1, round(value * scale))
        self.layout().setContentsMargins(px(12), px(12), px(12), px(12))
        self.layout().setSpacing(px(10))

    def set_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        self._last_image = QImage(
            rgb.data, w, h, channels * w, QImage.Format_RGB888
        ).copy()
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image).scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image.setPixmap(pixmap)

