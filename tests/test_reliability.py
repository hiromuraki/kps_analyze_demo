"""可靠性测试：异常输入、边界条件、状态机语义、长时运行、端到端服务。

运行：
    uv run tests/test_reliability.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import FrameAnalyzer
from core.kp2d_extractor import Mock2dExtractor
from core.kp3d_reconstructor import Mock3dReconstructor
from core.pose_judger import judge_pose
from core.rules_loader import load_rule

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, str, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if passed else "FAIL", detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


def _analyzer(rule_name: str = "哈克深蹲-new") -> FrameAnalyzer:
    return FrameAnalyzer(
        kp2d_extractor=Mock2dExtractor(str(ROOT / "sample_data/example-1/2d_coco17_kps.npz")),
        kp3d_reconstructor=Mock3dReconstructor(str(ROOT / "sample_data/example-1/3d_kps.npz")),
        pose_name=rule_name,
        pose_rule=load_rule(rule_name),
    )


# ---------------------------------------------------------------------------
def test_invalid_input() -> None:
    print("\n=== 1. 异常输入 ===")

    check("不存在的规则文件", load_rule("不存在的动作") == {}, "返回空 dict，不抛异常")
    check("空规则名", load_rule("") == {}, "返回空 dict，不抛异常")

    from core.rules_loader import get_rule_names

    names = get_rule_names()
    check("规则目录可枚举", len(names) >= 5, f"{names}")

    # 损坏的 npz
    broken = ROOT / "tests" / "_broken.npz"
    broken.write_bytes(b"this is not an npz")
    try:
        Mock2dExtractor(str(broken))
        check("损坏的 npz 被拒绝", False, "未抛异常")
    except Exception as e:
        check("损坏的 npz 被拒绝", True, f"{type(e).__name__}")
    finally:
        broken.unlink(missing_ok=True)

    # 全零关键点（模拟未检出人体）
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    an = _analyzer()
    an._kp2d_extractor._kps_frames = np.zeros((5, 17, 3), dtype=np.float32)
    an._kp3d_reconstructor._kps_frames = np.zeros((5, 17, 3), dtype=np.float32)
    try:
        r = an.analyze_frame(frame)
        ok = r.rendered.shape == frame.shape and r.kps_3d.shape == (17, 3)
        check("全零骨骼不中断管线", ok, f"渲染帧 {r.rendered.shape}，计次 {an.rep_count}")
    except Exception as e:
        check("全零骨骼不中断管线", False, f"{type(e).__name__}: {e}")

    # 空规则
    try:
        v, k = judge_pose(np.zeros((17, 2)), np.zeros((17, 3)), {})
        check("空规则不抛异常", v == [] and k == [], f"violations={v}")
    except Exception as e:
        check("空规则不抛异常", False, f"{type(e).__name__}: {e}")

    # 无 rep_counting 的规则不创建计次状态机
    an2 = _analyzer("高位下拉-new")
    check(
        "无 rep_counting 时不创建状态机",
        an2._rep_counter is None and an2.rep_feature_value == 0.0,
        "motion 与特征值退化为空，判定链路不受影响",
    )


# ---------------------------------------------------------------------------
def test_state_machine() -> None:
    print("\n=== 2. 状态机语义 ===")
    an = _analyzer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        an.analyze_frame(frame)
    running_reps, running_frames = an.rep_count, an.total_frames

    an.pause()
    check("pause 后 state", an.state == "paused", an.state)
    for _ in range(10):
        an.analyze_frame(frame)
    check(
        "paused 期间统计冻结",
        an.rep_count == running_reps and an.total_frames == running_frames + 10,
        f"计次 {an.rep_count} 不变，帧计数继续（供渲染推流）",
    )

    an.resume()
    check("resume 后 state", an.state == "running", an.state)

    an.stop()
    snap_reps = an.rep_count
    frozen = an.stats_history[-1]
    check("stop 后进入 stopped", an.state == "stopped", an.state)
    check("stop 保存历史快照", len(an.stats_history) == 1, f"training_id={frozen['training_id']}")
    check("stopped 后属性返回冻结值", an.rep_count == snap_reps, f"rep_count={an.rep_count}")

    old_tid = an.training_id
    an.resume()
    for _ in range(5):
        an.analyze_frame(frame)
    check(
        "重新开始清零并换新 ID",
        an.training_id != old_tid and an.total_frames == 5,
        f"{old_tid} → {an.training_id}",
    )


# ---------------------------------------------------------------------------
def test_long_run() -> None:
    print("\n=== 3. 长时运行稳定性 ===")
    an = _analyzer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rounds = 12  # 12 × 265 帧
    per_frame: list[float] = []
    counts: list[int] = []
    try:
        for _ in range(rounds * 265):
            t = time.perf_counter()
            an.analyze_frame(frame)
            per_frame.append((time.perf_counter() - t) * 1000)
            counts.append(an.rep_count)
        ok = True
        detail = ""
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"

    n = rounds * 265
    check("长时运行无异常", ok, detail or f"{n} 帧（{rounds} 轮）")
    check("计次单调不减", all(b >= a for a, b in zip(counts, counts[1:])), f"最终 {counts[-1]} 次")

    head = statistics.mean(per_frame[:1000])
    tail = statistics.mean(per_frame[-1000:])
    drift = abs(tail - head) / head
    check(
        "耗时不随时间劣化",
        drift < 0.5,
        f"前 1000 帧 {head:.3f} ms → 后 1000 帧 {tail:.3f} ms（漂移 {drift * 100:.1f}%）",
    )


# ---------------------------------------------------------------------------
async def _ws_probe(timeout: float = 25.0) -> tuple[bool, str]:
    """启动真实服务，经 WebSocket 与 REST 验证端到端链路。"""
    import websockets

    proc = subprocess.Popen(
        [
            "uv", "run", "main.py",
            "--analyzer-2d", "mock", "--analyzer-3d", "mock", "--camera", "-1",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    detail = ""
    try:
        # 等待 HTTP 就绪
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                import urllib.request
                urllib.request.urlopen("http://127.0.0.1:2800/poses", timeout=1)
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            return False, "服务在 40 s 内未就绪"

        import urllib.request
        poses = json.loads(urllib.request.urlopen("http://127.0.0.1:2800/poses", timeout=2).read())

        got = {"binary": 0, "kps3d": 0, "stats": 0}
        async with websockets.connect("ws://127.0.0.1:2800/ws", max_size=None) as ws:
            end = time.time() + timeout
            while time.time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    got["binary"] += 1
                else:
                    m = json.loads(msg)
                    if m.get("type") in got:
                        got[m["type"]] += 1

        ok = (
            got["binary"] > 0
            and got["kps3d"] > 0
            and got["stats"] > 0
            and poses.get("poses")
        )
        detail = (
            f"JPEG 帧 {got['binary']} 条，kps3d {got['kps3d']} 条，"
            f"stats {got['stats']} 条，动作列表 {len(poses.get('poses', []))} 项"
        )
        return ok, detail
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_end_to_end() -> None:
    print("\n=== 4. 端到端服务（WebSocket + REST）===")
    ok, detail = asyncio.run(_ws_probe())
    check("服务启动并推送三类消息", ok, detail)


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 68)
    print("  可靠性测试 — kps_analyze_demo")
    print("=" * 68)
    test_invalid_input()
    test_state_machine()
    test_long_run()
    test_end_to_end()

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print("\n" + "=" * 68)
    print(f"  用例 {len(RESULTS)} 项：通过 {n_pass}，失败 {len(RESULTS) - n_pass}")
    print("=" * 68)
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"  FAILED: {name} — {detail}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
