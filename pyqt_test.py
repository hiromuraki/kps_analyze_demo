import sys
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

VIDEO_PATH = Path("./sample_data/small/example.mp4")


class VideoPlayer(QMainWindow):
    def __init__(self, video_path: Path):
        super().__init__()
        self.setWindowTitle(f"KPS Analyze Demo — {video_path}")
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._interval = int(1000 / self._fps)

        # 居中显示视频帧的 QLabel
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #000;")

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self._label)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(self._cap_width(), self._cap_height())

        # 定时器驱动帧刷新
        self._timer = QTimer()
        self._timer.timeout.connect(self._show_next_frame)
        self._timer.start(self._interval)

    def _cap_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def _cap_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _show_next_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret:
                self._timer.stop()
                return

        # BGR → RGB → QImage → QPixmap
        h, w, _ = frame.shape
        image = QImage(frame.data, w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        self._label.setPixmap(pixmap.scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def closeEvent(self, event):
        self._timer.stop()
        self._cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    player = VideoPlayer(VIDEO_PATH)
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
