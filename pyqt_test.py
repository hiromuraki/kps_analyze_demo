"""PyQt 实时管线测试 —— 摄像头 + default 分析器。"""

from __future__ import annotations

import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from core.video_source import CameraRgbVideoSource
from core import (
    FrameAnalyzer,
    RTMPose2dPoseExtractor,
    MHFormer3dPoseReconstructor,
    get_rule_names,
    load_rule,
)

CAMERA_ID = 0
WIDTH, HEIGHT, FPS = 640, 480, 30


class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PyQt Pipeline — camera {CAMERA_ID}")

        # ── 视频源 ──
        self._camera = CameraRgbVideoSource(camera_id=CAMERA_ID, width=WIDTH, height=HEIGHT, fps=FPS)
        self._camera.flip_x = True
        if not self._camera.open():
            raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

        # ── 分析器 ──
        rule_names = get_rule_names()
        pose_type = rule_names[0] if rule_names else ""
        rule = load_rule(pose_type) if pose_type else {}
        self._analyzer = FrameAnalyzer(
            kp2d_extractor=RTMPose2dPoseExtractor(),
            kp3d_reconstructor=MHFormer3dPoseReconstructor(),
            pose_name=pose_type,
            pose_rule=rule,
        )

        # ── UI ──
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background: #000;")
        self._video_label.setMinimumSize(640, 360)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet("background: #1a1a1a; color: #ccc; font: 11px monospace;")

        self._stats = QLabel("等待数据 …")
        self._stats.setStyleSheet("color: #888; font: 11px monospace;")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self._log)
        rl.addWidget(self._stats)

        central = QWidget()
        cl = QHBoxLayout(central)
        cl.addWidget(self._video_label, 3)
        cl.addWidget(right, 1)
        self.setCentralWidget(central)
        self.resize(960, 540)

        # ── 计时 ──
        self._frame_count = 0
        self._wall_start = time.monotonic()
        self._total_capture_ms = 0.0
        self._total_analyze_ms = 0.0
        self._total_display_ms = 0.0

        # ── 主循环（用空闲定时器尽可能快） ──
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(0)

    # ------------------------------------------------------------------
    def _tick(self):
        # (1) 采集
        t = time.monotonic()
        frame = self._camera.get_frame()
        self._total_capture_ms += (time.monotonic() - t) * 1000
        h, w, _ = frame.shape

        # (2) 分析
        t = time.monotonic()
        rendered, _, violations = self._analyzer.analyze_frame(frame)
        self._total_analyze_ms += (time.monotonic() - t) * 1000

        if self._frame_count == 0:
            self._append_log(f"第一帧: {w}x{h}")
        if violations:
            self._append_log(";".join(violations))
        self._frame_count += 1

        # (3) 显示
        t = time.monotonic()
        image = QImage(rendered.data, w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        self._video_label.setPixmap(pixmap.scaled(
            self._video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self._total_display_ms += (time.monotonic() - t) * 1000

        # 实时 FPS
        if self._frame_count % 30 == 0:
            elapsed = time.monotonic() - self._wall_start
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            ana = self._total_analyze_ms / self._frame_count
            self._stats.setText(
                f"帧: {self._frame_count}  |  FPS: {fps:.1f}  |  分析: {ana:.0f} ms"
            )

    # ------------------------------------------------------------------
    def _append_log(self, text: str):
        self._log.append(text)
        c = self._log.textCursor()
        while self._log.document().blockCount() > 200:
            c.movePosition(c.MoveOperation.Start)
            c.select(c.SelectionType.BlockUnderCursor)
            c.removeSelectedText()
            c.deleteChar()

    # ------------------------------------------------------------------
    def _print_stats(self):
        elapsed = time.monotonic() - self._wall_start
        if self._frame_count == 0:
            return
        n = self._frame_count
        fps = n / elapsed
        lines = [
            "=" * 50,
            "  PyQt — Camera + Default Analyzer",
            "=" * 50,
            f"  wall time        {elapsed:8.2f} s",
            f"  frames           {n:8d}",
            f"  effective FPS    {fps:8.1f}",
            "-" * 50,
            f"  avg capture      {self._total_capture_ms / n:8.2f} ms",
            f"  avg analyze      {self._total_analyze_ms / n:8.2f} ms",
            f"  avg display      {self._total_display_ms / n:8.2f} ms",
            "-" * 50,
            f"  avg per frame    {elapsed / n * 1000:8.2f} ms",
            "=" * 50,
        ]
        for line in lines:
            print(line)
            self._append_log(line)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._timer.stop()
        self._print_stats()
        self._camera.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = AnalyzerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
