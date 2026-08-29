#!/usr/bin/env python3
"""Native PyQt5 UI for the thermal/RGB stick-figure pipeline."""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QAction,
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pipeline_worker import PipelineConfig, PipelineWorker
from temperature import THERMAL_ROOT, read_temperatures
from video_panel import VideoPanel
from video_recorder import RESEARCH_RECORDING_MODES

UI_DIR = Path(__file__).resolve().parent
BOARD_TEMP_LIMIT_C = 70.0
BOARD_TEMP_LIMIT_SECONDS = 5 * 60.0
BOARD_TEMP_POLL_MS = 2000


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = PipelineConfig()
        self.worker = None
        self._recording = False
        self._board_hot_since = None
        self._thermal_shutdown = False
        self._temperature_lock = threading.Lock()
        self._temperature_range = (26.0, 36.0)
        self._alignment_lock = threading.Lock()
        self._alignment_xy = (self.config.thermal_x, self.config.thermal_y)
        self._ui_ready = False
        self._metric_layouts = []
        screen_size = QApplication.primaryScreen().availableGeometry().size()
        self._ui_scale = min(
            screen_size.width() / 1380.0,
            screen_size.height() / 860.0,
        )

        self.setWindowTitle("Thermal Pose Monitor")
        self.setMinimumSize(1, 1)
        self.resize(screen_size)
        self._build_ui()
        self._apply_style()
        self._ui_ready = True

        self.board_temp_timer = QTimer(self)
        self.board_temp_timer.setInterval(BOARD_TEMP_POLL_MS)
        self.board_temp_timer.timeout.connect(self._update_board_temperature)
        self.board_temp_timer.start()
        self._update_board_temperature()

        self.exit_fullscreen_action = QAction(self)
        self.exit_fullscreen_action.setShortcut("Esc")
        self.exit_fullscreen_action.triggered.connect(self.exit_fullscreen)
        self.addAction(self.exit_fullscreen_action)

    def _px(self, value):
        return max(1, round(value * self._ui_scale))

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.outer_layout = outer
        outer.setContentsMargins(
            self._px(22), self._px(18), self._px(22), self._px(20)
        )
        outer.setSpacing(self._px(16))

        header = QHBoxLayout()
        self.header_layout = header
        header.setSpacing(self._px(8))
        heading_box = QVBoxLayout()
        self.heading_layout = heading_box
        heading_box.setSpacing(self._px(4))
        title = QLabel("Thermal Pose Monitor")
        title.setObjectName("appTitle")
        subtitle = QLabel("Thermal privacy masking and human pose monitoring")
        subtitle.setObjectName("subtitle")
        heading_box.addWidget(title)
        heading_box.addWidget(subtitle)
        header.addLayout(heading_box)
        header.addStretch()

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_label = QLabel("Stopped")
        self.status_label.setObjectName("statusText")
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        outer.addLayout(header)

        controls = QFrame()
        controls.setObjectName("controlBar")
        control_layout = QGridLayout(controls)
        self.control_layout = control_layout
        control_layout.setContentsMargins(
            self._px(16), self._px(13), self._px(16), self._px(13)
        )
        control_layout.setHorizontalSpacing(self._px(10))
        control_layout.setVerticalSpacing(self._px(8))

        self.start_button = QPushButton("▶  Start")
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("■  Close")
        self.stop_button.setObjectName("stopButton")
        self.background_button = QPushButton("Capture Background")
        self.background_button.setEnabled(False)
        self.record_button = QToolButton()
        self.record_button.setText("●  Record")
        self.record_button.setObjectName("recordButton")
        self.record_button.setEnabled(False)
        self.record_button.setToolTip(
            "Records Stick Figure, Raw RGB, Thermal video, and pose metadata"
        )
        self.stick_button = QPushButton("Stick Figure")
        self.stick_button.setCheckable(True)
        self.thermal_button = QPushButton("Raw Thermal")
        self.thermal_button.setCheckable(True)
        self.color_button = QPushButton("Raw RGB")
        self.color_button.setCheckable(True)

        control_layout.addWidget(self.start_button, 0, 0)
        control_layout.addWidget(self.stop_button, 0, 1)
        control_layout.addWidget(self.background_button, 0, 2)
        control_layout.addWidget(self.record_button, 0, 3)
        control_layout.addWidget(self.stick_button, 0, 4)
        control_layout.addWidget(self.thermal_button, 0, 5)
        control_layout.addWidget(self.color_button, 0, 6)
        control_layout.setColumnStretch(7, 1)

        self.min_temp = self._spin_box(26.0)
        self.max_temp = self._spin_box(36.0)
        control_layout.addWidget(QLabel("Temperature range"), 1, 0, 1, 2)
        control_layout.addWidget(self.min_temp, 1, 2)
        control_layout.addWidget(QLabel("°C  —"), 1, 3)
        control_layout.addWidget(self.max_temp, 1, 4)
        control_layout.addWidget(QLabel("°C"), 1, 5)
        outer.addWidget(controls)

        metrics = QHBoxLayout()
        self.metrics_layout = metrics
        metrics.setSpacing(self._px(10))
        self.fps_value = self._metric_card(metrics, "REAL-TIME FPS", "0.0")
        self.board_temp_value = self._metric_card(
            metrics, "BOARD TEMPERATURE", "— °C"
        )
        self.thermal_x_input = self._metric_input(
            metrics, "THERMAL X", self.config.thermal_x
        )
        self.thermal_y_input = self._metric_input(
            metrics, "THERMAL Y", self.config.thermal_y
        )
        outer.addLayout(metrics)

        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(self._px(14))
        self.video_grid.setColumnStretch(0, 1)
        self.video_grid.setColumnStretch(1, 1)
        self.video_grid.setColumnStretch(2, 1)
        self.video_grid.setRowStretch(0, 1)
        self.stick_panel = VideoPanel(
            "Stick Figure Mode", "Click Start to begin", self._ui_scale
        )
        self.thermal_panel = VideoPanel(
            "Raw Thermal",
            "Waiting for thermal video",
            self._ui_scale,
        )
        self.color_panel = VideoPanel(
            "Raw RGB", "Waiting for RGB video", self._ui_scale
        )
        self.video_grid.addWidget(self.stick_panel, 0, 0, 1, 3)
        self.video_grid.addWidget(self.thermal_panel, 0, 1)
        self.video_grid.addWidget(self.color_panel, 0, 2)
        self.stick_panel.hide()
        self.thermal_panel.hide()
        self.color_panel.hide()
        outer.addLayout(self.video_grid, 1)

        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.close)
        self.background_button.clicked.connect(self.capture_background)
        self.record_button.clicked.connect(self.toggle_recording)
        self.stick_button.toggled.connect(self._toggle_stick)
        self.thermal_button.toggled.connect(self._toggle_thermal)
        self.color_button.toggled.connect(self._toggle_color)
        self.min_temp.editingFinished.connect(self._update_temperature)
        self.max_temp.editingFinished.connect(self._update_temperature)
        self.thermal_x_input.editingFinished.connect(self._update_alignment)
        self.thermal_y_input.editingFinished.connect(self._update_alignment)

    def _metric_card(self, parent, label, value):
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            self._px(16), self._px(10), self._px(20), self._px(10)
        )
        name = QLabel(label)
        name.setObjectName("metricName")
        number = QLabel(value)
        number.setObjectName("metricValue")
        layout.addWidget(name)
        layout.addWidget(number)
        self._metric_layouts.append(layout)
        parent.addWidget(card, 1)
        return number

    def _metric_input(self, parent, label, value):
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            self._px(16), self._px(10), self._px(20), self._px(10)
        )
        name = QLabel(label)
        name.setObjectName("metricName")
        field = self._spin_box(value)
        field.setObjectName("metricInput")
        layout.addWidget(name)
        layout.addWidget(field)
        self._metric_layouts.append(layout)
        parent.addWidget(card, 1)
        return field

    def _spin_box(self, default):
        field = QDoubleSpinBox()
        field.setRange(default - 10.0, default + 10.0)
        field.setSingleStep(0.5)
        field.setDecimals(1)
        field.setValue(default)
        field.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        field.setAlignment(Qt.AlignCenter)
        field.setFixedHeight(self._px(58))
        field.setMinimumWidth(self._px(130))
        return field

    def _apply_style(self):
        stylesheet = (
            """
            QMainWindow, QWidget { background: #0b1118; color: #e8edf3;
                font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
                font-size: 14px; }
            #appTitle { font-size: 26px; font-weight: 700; color: #f7fafc; }
            #subtitle { font-size: 13px; color: #8090a3; }
            #statusDot { color: #657386; font-size: 18px; }
            #statusText { color: #aab6c5; font-weight: 600; }
            #controlBar, #metricCard, #videoPanel {
                background: #121b25; border: 1px solid #223040; border-radius: 10px; }
            QPushButton, QToolButton { background: #1b2836; border: 1px solid #314255;
                border-radius: 7px; padding: 9px 15px; font-weight: 600; }
            QPushButton:hover, QToolButton:hover { background: #26384a; }
            QPushButton:checked, QToolButton:checked {
                color: #72e0c1; border-color: #35aa8b;
                background: #153b36; }
            QPushButton:disabled, QToolButton:disabled {
                color: #586575; background: #151d26; }
            QMenu { background: #121b25; color: #f1f5f9;
                border: 1px solid #33465a; selection-background-color: #26384a;
                padding: 5px; }
            QMenu::item { padding: 9px 24px; }
            #startButton { color: #09201a; background: #55d6ae; border: none; }
            #startButton:hover { background: #73e3c1; }
            #stopButton { color: #ff9c9c; }
            #recordButton { color: #ff9c9c; padding-right: 34px; }
            #recordButton::menu-button { subcontrol-origin: border;
                subcontrol-position: top right; width: 30px;
                border-left: 1px solid #314255; }
            #recordButton::menu-arrow { image: url(__SPIN_DOWN__);
                width: 14px; height: 9px; }
            QDoubleSpinBox { background: #091018; color: #f1f5f9;
                border: 1px solid #33465a; border-radius: 6px;
                padding: 6px 38px 6px 10px; font-size: 24px;
                font-weight: 700;
                selection-background-color: #35aa8b; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; width: 36px; background: #1b2836;
                border-left: 1px solid #33465a; }
            QDoubleSpinBox::up-button { subcontrol-position: top right; }
            QDoubleSpinBox::down-button { subcontrol-position: bottom right; }
            QDoubleSpinBox::up-button { border-top-right-radius: 5px; }
            QDoubleSpinBox::down-button { border-bottom-right-radius: 5px; }
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
                width: 18px; height: 12px; }
            QDoubleSpinBox::up-arrow { image: url(__SPIN_UP__); }
            QDoubleSpinBox::down-arrow { image: url(__SPIN_DOWN__); }
            #metricName { color: #7f90a4; font-size: 14px; }
            #metricValue { color: #f2f6fa; font-size: 20px; font-weight: 700; }
            #metricInput { font-size: 24px; font-weight: 700; }
            #panelTitle { color: #aab8c8; font-size: 13px; font-weight: 650; }
            #videoImage { background: #070b10; border-radius: 6px; color: #526172; }
            """
        )
        stylesheet = stylesheet.replace(
            "__SPIN_UP__", (UI_DIR / "assets/spin_up.svg").as_posix()
        ).replace(
            "__SPIN_DOWN__", (UI_DIR / "assets/spin_down.svg").as_posix()
        )
        stylesheet = re.sub(
            r"(\d+)px",
            lambda match: f"{self._px(int(match.group(1)))}px",
            stylesheet,
        )
        self.setStyleSheet(stylesheet)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._ui_ready:
            return
        size = event.size()
        scale = max(0.55, min(size.width() / 1380.0, size.height() / 860.0))
        if abs(scale - self._ui_scale) < 0.02:
            return
        self._ui_scale = scale
        self._apply_responsive_scale()

    def _apply_responsive_scale(self):
        px = self._px
        self.outer_layout.setContentsMargins(px(22), px(18), px(22), px(20))
        self.outer_layout.setSpacing(px(16))
        self.header_layout.setSpacing(px(8))
        self.heading_layout.setSpacing(px(4))
        self.control_layout.setContentsMargins(px(16), px(13), px(16), px(13))
        self.control_layout.setHorizontalSpacing(px(10))
        self.control_layout.setVerticalSpacing(px(8))
        self.metrics_layout.setSpacing(px(10))
        self.video_grid.setSpacing(px(14))
        for layout in self._metric_layouts:
            layout.setContentsMargins(px(16), px(10), px(20), px(10))
        for field in (
            self.min_temp,
            self.max_temp,
            self.thermal_x_input,
            self.thermal_y_input,
        ):
            field.setFixedHeight(px(58))
            field.setMinimumWidth(px(130))
        for panel in (self.stick_panel, self.thermal_panel, self.color_panel):
            panel.set_ui_scale(self._ui_scale)
        self._apply_style()

    def temperature_range(self):
        with self._temperature_lock:
            return self._temperature_range

    def _update_temperature(self):
        low = self.min_temp.value()
        high = self.max_temp.value()
        if low >= high:
            low, high = self.temperature_range()
            self.min_temp.setValue(low)
            self.max_temp.setValue(high)
            QMessageBox.warning(
                self,
                "Invalid temperature range",
                "The minimum temperature must be lower than the maximum.",
            )
            return
        with self._temperature_lock:
            self._temperature_range = (low, high)

    def alignment_xy(self):
        with self._alignment_lock:
            return self._alignment_xy

    def _update_alignment(self):
        x = self.thermal_x_input.value()
        y = self.thermal_y_input.value()
        with self._alignment_lock:
            self._alignment_xy = (x, y)

    def start_pipeline(self):
        if self.worker and self.worker.isRunning():
            return
        self._update_temperature()
        self._update_alignment()
        self.stick_button.setChecked(False)
        self.thermal_button.setChecked(False)
        self.color_button.setChecked(False)
        self.background_button.setText("Capture Background")
        self.background_button.setEnabled(False)
        self.record_button.setText("●  Record")
        self.record_button.setEnabled(False)
        self._recording = False
        self.worker = PipelineWorker(
            self.config, self.temperature_range, self.alignment_xy
        )
        self.worker.frames_ready.connect(self._show_frames)
        self.worker.stats_ready.connect(self._show_stats)
        self.worker.state_changed.connect(self._show_state)
        self.worker.background_captured.connect(self._background_captured)
        self.worker.recording_changed.connect(self._recording_changed)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.start_button.setEnabled(False)
        self.worker.start()

    def _worker_finished(self):
        self.start_button.setEnabled(True)
        self.background_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.record_button.setText("●  Record")
        self._recording = False
        self.status_dot.setStyleSheet("color: #657386")

    def _show_state(self, state):
        self.status_label.setText(state)
        active = state == "Running"
        self.background_button.setEnabled(active)
        self.record_button.setEnabled(active)
        self.status_dot.setStyleSheet(
            "color: #55d6ae" if active else "color: #657386"
        )

    def capture_background(self):
        if self.worker and self.worker.isRunning():
            self.background_button.setEnabled(False)
            self.status_label.setText("Capturing empty background…")
            self.worker.request_background_capture()

    def _background_captured(self):
        self.status_label.setText("Running · Background captured")
        self.background_button.setText("Recapture Background")
        self.background_button.setEnabled(True)

    def toggle_recording(self):
        if not self.worker or not self.worker.isRunning():
            return
        self.record_button.setEnabled(False)
        if self._recording:
            self.worker.request_recording(False)
            return
        self.worker.request_recording(True, RESEARCH_RECORDING_MODES)

    def _recording_changed(self, recording, paths):
        self._recording = recording
        paths = tuple(Path(path) for path in paths)
        if recording:
            self.record_button.setText("■  Stop Recording")
            self.status_label.setText(
                f"Recording {len(paths)} mode(s)"
            )
        else:
            self.record_button.setText("●  Record")
            self.status_label.setText(
                f"Saved {len(paths)} recording(s)"
            )
        self.record_button.setEnabled(
            bool(self.worker and self.worker.isRunning())
        )

    def _show_error(self, message):
        QMessageBox.critical(self, "Pipeline error", message)

    def _show_frames(self, stick, thermal, color):
        if self.stick_panel.isVisible():
            self.stick_panel.set_frame(stick)
        # Avoid unnecessary RGB conversion/painting work for hidden diagnostics.
        # A newly shown panel receives the next camera frame immediately.
        if self.thermal_panel.isVisible() and thermal is not None:
            self.thermal_panel.set_frame(thermal)
        if self.color_panel.isVisible() and color is not None:
            self.color_panel.set_frame(color)

    def _show_stats(self, stats):
        self.fps_value.setText(f"{stats['fps']:.1f}")

    def _update_board_temperature(self):
        readings = read_temperatures(THERMAL_ROOT)
        if not readings:
            self.board_temp_value.setText("Unavailable")
            self.board_temp_value.setStyleSheet("color: #ff9c9c")
            self._board_hot_since = None
            return

        zone_name, hottest = max(readings, key=lambda item: item[1])
        self.board_temp_value.setText(f"{hottest:.1f} °C")
        self.board_temp_value.setToolTip(f"Hottest thermal zone: {zone_name}")
        now = time.monotonic()

        if hottest > BOARD_TEMP_LIMIT_C:
            self.board_temp_value.setStyleSheet("color: #ff6b6b")
            if self._board_hot_since is None:
                self._board_hot_since = now
            if now - self._board_hot_since >= BOARD_TEMP_LIMIT_SECONDS:
                self._begin_thermal_shutdown(hottest, zone_name)
        else:
            self.board_temp_value.setStyleSheet("")
            self._board_hot_since = None

    def _begin_thermal_shutdown(self, temperature, zone_name):
        if self._thermal_shutdown:
            return
        self._thermal_shutdown = True
        self.board_temp_timer.stop()
        self.status_label.setText(
            f"Overheat: {zone_name} {temperature:.1f} °C · Shutting down"
        )
        self.status_dot.setStyleSheet("color: #ff6b6b")
        self.start_button.setEnabled(False)

        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.finished.connect(QApplication.instance().quit)
        else:
            QApplication.instance().quit()

    def _toggle_stick(self, visible):
        self.stick_panel.setVisible(visible)
        self._update_video_layout()

    def _toggle_thermal(self, visible):
        self.thermal_panel.setVisible(visible)
        if self.worker:
            self.worker.request_view("raw_thermal", visible)
        self._update_video_layout()

    def _toggle_color(self, visible):
        self.color_panel.setVisible(visible)
        if self.worker:
            self.worker.request_view("raw_rgb", visible)
        self._update_video_layout()

    def _update_video_layout(self):
        # A horizontal layout makes better use of widescreen space for 4:3 video.
        visible_panels = [
            panel
            for panel in (self.stick_panel, self.thermal_panel, self.color_panel)
            if panel.isVisible()
        ]
        self.video_grid.removeWidget(self.stick_panel)
        self.video_grid.removeWidget(self.thermal_panel)
        self.video_grid.removeWidget(self.color_panel)
        panel_count = len(visible_panels)
        for column in range(3):
            self.video_grid.setColumnStretch(
                column, 1 if column < panel_count else 0
            )
        for column, panel in enumerate(visible_panels):
            self.video_grid.addWidget(
                panel, 0, column, 1, 3 if panel_count == 1 else 1
            )

    def exit_fullscreen(self):
        if not self.isFullScreen():
            return
        available = QApplication.primaryScreen().availableGeometry()
        width = round(available.width() * 0.92)
        height = round(available.height() * 0.92)
        self.showNormal()
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            if not self.worker.wait(3500):
                event.ignore()
                QMessageBox.information(
                    self,
                    "Shutting down",
                    "Camera resources are still being released. Please try again shortly.",
                )
                return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Thermal Pose Monitor")
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec_())
