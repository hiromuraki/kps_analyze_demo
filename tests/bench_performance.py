"""性能测试：在 QCS8550 实机上测量各环节耗时与端到端帧率。

三个层次：
  1. 单模型稳态耗时 —— 2D 提取、3D 重建（TTA 开/关）
  2. 冷启动耗时       —— 模型从加载到可用的时间
  3. 端到端每帧耗时   —— FrameAnalyzer 全链路（提取 + 重建 + 判定 + 计次 + 渲染 + JPEG 编码）

运行：
    uv run tests/bench_performance.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import FrameAnalyzer
from core.converter import DataConverter
from core.kp2d_extractor import RTMPose2dPoseExtractor
from core.kp3d_reconstructor import MHFormer3dPoseReconstructor
from core.rules_loader import load_rule

VIDEO = "sample_data/example-1/video.mp4"
BENCH_FRAMES = 120


def _stats(samples: list[float]) -> str:
    return (
        f"均值 {statistics.mean(samples):6.1f} ms  "
        f"中位 {statistics.median(samples):6.1f} ms  "
        f"P95 {sorted(samples)[int(len(samples) * 0.95) - 1]:6.1f} ms  "
        f"最小 {min(samples):6.1f} ms"
    )


def _read_frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def bench_2d(frames: list[np.ndarray]) -> None:
    print("\n=== 1. 2D 提取 RTMDet + RTMPose ===")
    t0 = time.perf_counter()
    ext = RTMPose2dPoseExtractor()
    cold = time.perf_counter() - t0
    print(f"  构造（惰性，不含模型加载）  {cold * 1000:6.1f} ms")

    t0 = time.perf_counter()
    kp_first = ext.extract(frames[0])
    print(f"  首帧（含 DSP 模型加载）      {(time.perf_counter() - t0) * 1000:6.1f} ms")

    samples = []
    for f in frames[:BENCH_FRAMES]:
        t = time.perf_counter()
        kp = ext.extract(f)
        samples.append((time.perf_counter() - t) * 1000)
    print(f"  稳态（{BENCH_FRAMES} 帧）           {_stats(samples)}")
    print(f"  平均检出的关键点置信度      {kp[:, 2].mean():.3f}")
    ext.close()


def bench_3d(frames: list[np.ndarray]) -> None:
    print("\n=== 2. 3D 重建 MHFormer（351 帧窗口）===")
    ext = RTMPose2dPoseExtractor()
    kp2d = DataConverter.coco17_to_h36m(np.stack([ext.extract(f) for f in frames[:60]]))
    ext.close()
    seq = np.tile(kp2d, (6, 1, 1))[:351]  # 填满 351 帧窗口
    size = (frames[0].shape[1], frames[0].shape[0])

    for disable_flip, label in ((False, "TTA 开"), (True, "TTA 关")):
        t0 = time.perf_counter()
        rec = MHFormer3dPoseReconstructor(disable_flip=disable_flip)
        t0 = time.perf_counter()
        rec.reconstruct(seq, [-1], size)  # 触发加载
        cold = (time.perf_counter() - t0) * 1000

        samples = []
        for _ in range(BENCH_FRAMES):
            t = time.perf_counter()
            rec.reconstruct(seq, [-1], size)
            samples.append((time.perf_counter() - t) * 1000)
        print(f"  {label}：冷启动 {cold:6.1f} ms")
        print(f"  {label}：稳态      {_stats(samples)}")
        rec.close()


def bench_end_to_end() -> None:
    print("\n=== 3. 端到端每帧耗时（FrameAnalyzer 全链路）===")
    frames = _read_frames(BENCH_FRAMES)
    analyzer = FrameAnalyzer(
        kp2d_extractor=RTMPose2dPoseExtractor(),
        kp3d_reconstructor=MHFormer3dPoseReconstructor(),
        pose_name="哈克深蹲-new",
        pose_rule=load_rule("哈克深蹲-new"),
    )

    # 先跑一段让滑动窗口填满，再进入稳态测量
    for f in frames[:30]:
        analyzer.analyze_frame(f)

    total, encode_ms = [], []
    for f in frames[30:]:
        t = time.perf_counter()
        result = analyzer.analyze_frame(f)
        total.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        cv2.imencode(".jpg", result.rendered, [cv2.IMWRITE_JPEG_QUALITY, 50])
        encode_ms.append((time.perf_counter() - t) * 1000)

    n = len(total)
    mean_total = statistics.mean(total)
    mean_enc = statistics.mean(encode_ms)
    print(f"  分析（提取+重建+判定+计次+渲染）  {_stats(total)}")
    print(f"  JPEG 编码（质量 50）              {_stats(encode_ms)}")
    print(f"  服务端每帧合计                    {mean_total + mean_enc:6.1f} ms")
    print(f"  理论服务端帧率上限                {1000 / (mean_total + mean_enc):6.1f} fps")
    print(f"  实测累计动作计数                  {analyzer.rep_count}")
    print(f"  违规帧数 / 有效帧数               {analyzer.violated_frame_count} / {analyzer.total_frames}")


def main() -> int:
    print("=" * 68)
    print("  性能测试 — QCS8550 (aarch64, AidLux, QNN 2.36, Hexagon DSP)")
    print("=" * 68)
    frames = _read_frames(BENCH_FRAMES)
    print(f"\n测试素材：{VIDEO}，{frames[0].shape[1]}x{frames[0].shape[0]}，取 {len(frames)} 帧")

    bench_2d(frames)
    bench_3d(frames)
    bench_end_to_end()
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
