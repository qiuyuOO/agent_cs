"""完整自检 —— 单元级 + 集成级 + 端到端.

用法:
    # 单元 + 集成 (约 1 分钟, 不需要 API key)
    .venv\\Scripts\\python.exe -m cs2clipper.selftest

    # 加上端到端出片 (较慢, 会真的渲染视频)
    .venv\\Scripts\\python.exe -m cs2clipper.selftest --e2e

    # 端到端并额外验证 LLM 编排
    .venv\\Scripts\\python.exe -m cs2clipper.selftest --e2e --llm

    # 指定真实音乐
    .venv\\Scripts\\python.exe -m cs2clipper.selftest --e2e --music "D:\\CloudMusic\\xxx.mp3"

检查项按模块分组, 每项独立捕获异常, 最后打印汇总。这样一次运行就能看出
"哪个模块坏了", 而不是第一个报错就中断。
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

# 必须在其他重库之前导入 (环境重定向)
from . import config  # noqa: F401

import numpy as np  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []   # (分组, 名称, 状态/说明)
_CURRENT_GROUP = "?"


class Check:
    """把一组断言收集起来, 单项失败不影响后续检查."""

    def __init__(self, group: str) -> None:
        self.group = group
        global _CURRENT_GROUP
        _CURRENT_GROUP = group

    def __call__(self, name: str, fn) -> object | None:
        try:
            t0 = time.time()
            detail = fn()
            dt = time.time() - t0
            msg = f"{detail}" if detail else "ok"
            RESULTS.append((self.group, name, f"PASS  ({dt:.2f}s) {msg}"))
            print(f"  [PASS] {name}  ({dt:.2f}s) {msg}")
            return detail
        except Exception as e:
            RESULTS.append((self.group, name, f"FAIL  {type(e).__name__}: {e}"))
            print(f"  [FAIL] {name}")
            print(f"         {type(e).__name__}: {e}")
            tb = traceback.format_exc().strip().splitlines()[-4:]
            for line in tb:
                print(f"         {line.strip()}")
            return None


def approx(a: float, b: float, tol: float) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (容差 {tol})")


def is_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ==================================================================
# 1. 环境
# ==================================================================
def test_environment(c: Check) -> None:
    def ffmpeg_present():
        p = config.require_ffmpeg()
        probe = config.find_ffprobe()
        is_true(probe is not None, "找不到 ffprobe")
        return f"{p.name} + {probe.name}"

    def dirs_writable():
        for d in (config.OUT_DIR, config.WORK_DIR, config.CACHE_DIR, config.HOME_DIR):
            is_true(d.is_dir(), f"目录缺失: {d}")
        probe = config.WORK_DIR / "_selftest_write.tmp"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return "工作区可写"

    def aspect_presets():
        for name, (w, h) in config.ASPECT_PRESETS.items():
            is_true(w > 0 and h > 0, f"画幅 {name} 尺寸非法")
        return str(list(config.ASPECT_PRESETS))

    def imports_ok():
        import cs2clipper.compose, cs2clipper.demo, cs2clipper.mapview  # noqa
        import cs2clipper.music, cs2clipper.planner, cs2clipper.pipeline  # noqa
        import cs2clipper.radar  # noqa
        return "全部模块可导入"

    def demo_file_exists():
        is_true(config.DEFAULT_DEMO.is_file(), f"示例 demo 不存在: {config.DEFAULT_DEMO}")
        mb = config.DEFAULT_DEMO.stat().st_size / 1024 / 1024
        return f"{mb:.0f} MB"

    c("ffmpeg / ffprobe 可用", ffmpeg_present)
    c("工程目录可写", dirs_writable)
    c("画幅预设合法", aspect_presets)
    c("全部模块可导入", imports_ok)
    c("示例 demo 存在", demo_file_exists)


# ==================================================================
# 2. 音乐分析 (纯函数 + 合成曲 + 真实曲)
# ==================================================================
def test_music(c: Check, real_music: Path | None) -> None:
    from . import music as M
    def synth_track():
        p = config.WORK_DIR / "selftest_track.wav"
        if not p.is_file():
            M.make_test_track(p)
        is_true(p.stat().st_size > 100_000, "合成音轨太小")
        return f"{p.stat().st_size / 1024:.0f} KB"

    c("生成合成测试音轨", synth_track)

    holder: dict = {}

    def synth_analysis():
        p = config.WORK_DIR / "selftest_track.wav"
        a = M.analyze_music(p)
        holder["synth"] = a
        is_true(len(a.segments) >= 3, f"段数过少: {len(a.segments)}")
        is_true(len(a.beat_times) > 50, f"拍点过少: {len(a.beat_times)}")
        is_true(100 < a.bpm < 160, f"BPM 异常: {a.bpm}")
        # 合成曲的 drop 段应该是能量最高的
        top = max(a.segments, key=lambda s: s.arousal)
        is_true(top.arousal > 0.8, f"最高激烈度仅 {top.arousal}")
        return f"BPM {a.bpm:.1f}, {len(a.segments)} 段, 最高段 {top.label}"

    c("合成曲分析 + 情绪排序", synth_analysis)

    def beat_grid():
        a = holder["synth"]
        grid = M.build_beat_grid(a.beat_times, a.duration)
        is_true(len(grid) > 10, "节拍网格过稀")
        # 网格必须单调递增且间隔不小于 250ms
        for i in range(1, len(grid)):
            is_true(grid[i] > grid[i - 1], "节拍网格非单调")
            is_true(grid[i] - grid[i - 1] >= 0.24, "节拍网格过密")
        return f"{len(grid)} 个卡点"

    c("卡点网格单调且不重叠", beat_grid)

    def snap_works():
        """snap_to_beat 的合约验证.

        这个用例先后写错过三次, 根因都是**用挑出来的两个数去验证集合属性**:
          * 假设"容差内必吸到我挑的那条" —— 拍点有重复值/密集处, min() 取哪条都合理
          * 假设"bt[2]+5.0 的最近拍是 bt[2]" —— 5 秒后早就换成别的拍了
        所以这里改成: 只对**确定可控**的输入断言, 对真实拍点只验证不变量。
        """
        a = holder["synth"]
        bt = a.beat_times
        is_true(len(bt) > 5, "拍点不足")

        # --- 1. 确定输入: 用手工构造的拍点表 ---
        manual = [1.0, 2.0, 3.0, 10.0]
        approx(M.snap_to_beat(1.2, manual, tol=0.35), 1.0, 1e-9)     # 容差内左吸
        approx(M.snap_to_beat(1.9, manual, tol=0.35), 2.0, 1e-9)     # 容差内右吸
        approx(M.snap_to_beat(1.5, manual, tol=0.35), 1.5, 1e-9)     # 等距且超容差 -> 原样
        approx(M.snap_to_beat(6.0, manual, tol=0.35), 6.0, 1e-9)     # 容差外 -> 原样
        approx(M.snap_to_beat(10.2, manual, tol=0.35), 10.0, 1e-9)   # 末条也能吸
        approx(M.snap_to_beat(3.0, manual, tol=0.0), 3.0, 1e-9)      # 正好落在拍上
        approx(M.snap_to_beat(3.3, [], tol=0.35), 3.3, 1e-9)         # 空表不崩

        # --- 2. 真实拍点: 只验证不变量, 不预设吸到哪一条 ---
        bt_set = set(bt)
        checked = 0
        for i in range(0, len(bt) - 1, max(1, len(bt) // 10)):
            for off in (0.0, 0.03, 0.12, 0.28):
                probe = bt[i] + off
                got = M.snap_to_beat(probe, bt, tol=0.35)
                nearest = min(abs(b - probe) for b in bt)
                if nearest <= 0.35:
                    is_true(got in bt_set, f"{probe:.4f} 结果不在拍点集合内")
                    approx(abs(got - probe), nearest, 1e-9)
                else:
                    approx(got, probe, 1e-9)
                checked += 1

        ivs = [bt[i + 1] - bt[i] for i in range(len(bt) - 1)]
        return (f"确定用例 7 项 + 真实拍点 {checked} 组, "
                f"拍距 {min(ivs):.3f}~{max(ivs):.3f}s")

    c("节拍吸附逻辑", snap_works)

    c("节拍吸附逻辑", snap_works)

    def label_no_fake_drop():
        """全曲能量平稳时不得凭空造 drop —— 这是真实音乐暴露过的缺陷.

        这里刻意用**实测数字**(审查时从真歌曲逐段量出来的原始 RMS): 峰值段
        0.1319 只比次高 0.1210 高 9%、比中位 0.1085 高 22%, 但末尾淡出段
        0.0084 把 (max-min)/median 撑到 1.138。旧判据拿 1.138 跟 0.35 比
        -> has_climax=True -> 平稳歌被判出 drop, 保护完全失效。
        用真实数值测, 才不会出现"注入一个理想常量所以永远通过"的假绿。
        """
        real_rms = np.array([0.1210, 0.1150, 0.0873, 0.1085, 0.1085, 0.1319,
                             0.0975, 0.0701, 0.1044, 0.1193, 0.0084])
        st = M.energy_stats(real_rms)
        # 旧公式算出来的值必须确实 >= 0.35, 否则这个用例没有意义
        old_range = (real_rms.max() - real_rms.min()) / abs(np.median(real_rms))
        is_true(old_range >= 0.35,
                f"前提不成立: 旧公式只得到 {old_range:.3f}")
        is_true(not st["has_climax"],
                f"平稳曲被判有高潮: {st}")
        is_true(st["peak_vs_second"] < M.PEAK_VS_SECOND or st["peak_vs_median"] < M.PEAK_VS_MEDIAN,
                f"峰值显著度判据异常: {st}")

        stats = [
            {"energy": 0.20, "arousal": 0.2, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.9, "start": 0, "end": 10},
            {"energy": 0.55, "arousal": 0.5, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.1, "start": 10, "end": 20},
            {"energy": 1.00, "arousal": 1.0, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": 1.0, "start": 20, "end": 30},
            {"energy": 0.45, "arousal": 0.4, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.2, "start": 30, "end": 40},
        ]
        # 先确认"没有高潮保护"时确实会选出峰值 (说明这个用例有意义)
        raw = M._label_segments(stats)
        is_true("drop" in raw, f"前提不成立: 无保护时也没选出 drop: {raw}")
        labels = M._label_segments(stats, has_climax=False)
        is_true("drop" not in labels, f"平稳曲被造出 drop: {labels}")
        return (f"实测峰值/次高 {st['peak_vs_second']:.3f} "
                f"-> has_climax={st['has_climax']}; 无保护={raw} -> 有保护={labels}")

    c("平稳曲不凭空造 drop", label_no_fake_drop)

    def energy_stats_sees_real_climax():
        """真有大落差的曲子必须仍被判为有高潮 (避免把保护做成"永不高潮")."""
        st = M.energy_stats(np.array([0.12, 0.20, 0.28, 0.60, 0.90, 0.80, 0.30, 0.18]))
        is_true(st["has_climax"], f"明显有落差的曲子被判无高潮: {st}")
        is_true(st["peak_vs_median"] >= M.PEAK_VS_MEDIAN,
                f"峰值/中位应达标: {st}")
        # 中位附近是"平台 + 一个尖峰"时也要认高潮 (此时 peak_vs_second 不必达标)
        plateau = M.energy_stats(np.array([0.50, 0.52, 0.50, 0.53, 0.50, 1.60, 0.51, 0.50]))
        is_true(plateau["has_climax"], f"尖峰被平台淹没: {plateau}")
        flat = M.energy_stats(np.full(8, 0.5))
        is_true(not flat["has_climax"], "恒定能量被判有高潮")
        return (f"有落差 {st['peak_vs_second']:.2f} -> True; "
                f"平台尖峰 {plateau['peak_prominence']:.2f} -> True; 恒定 -> False")

    c("有真实落差的曲子仍判有高潮", energy_stats_sees_real_climax)

    def label_real_drop():
        """有明显落差的曲子应识别出 drop."""
        stats = [
            {"energy": 0.20, "arousal": 0.2, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.9, "start": 0, "end": 10},
            {"energy": 0.55, "arousal": 0.5, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.1, "start": 10, "end": 20},
            {"energy": 1.00, "arousal": 1.0, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": 1.0, "start": 20, "end": 30},
            {"energy": 0.45, "arousal": 0.4, "valence": 0.5, "onset_rate": 1.0,
             "rel_energy": -0.2, "start": 30, "end": 40},
        ]
        labels = M._label_segments(stats)
        is_true("drop" in labels, f"有明显高潮却没识别出 drop: {labels}")
        is_true(labels[2] == "drop", f"drop 位置不对: {labels}")
        return f"{labels}"

    c("有明显落差的曲子识别出 drop", label_real_drop)

    def emotion_consistency():
        """高激烈度 + 暗 不得输出"压迫"(自相矛盾的旧缺陷).

        注意断言要检查完整词"压迫"而不是"压": 正确输出"压制感强"里也含"压",
        用单字断言会把正确结果误判为失败。
        """
        loud_dark = M._emotion_text(0.95, 0.1, "verse", rel_energy=0.5)
        quiet_dark = M._emotion_text(0.95, 0.1, "verse", rel_energy=-0.5)
        is_true("压迫" not in loud_dark, f"偏响的暗色段仍被称压迫: {loud_dark}")
        is_true("压制感" in quiet_dark, f"偏轻的暗色段应称压制感: {quiet_dark}")
        is_true("压迫" not in quiet_dark, f"用词应统一为'压制感': {quiet_dark}")
        return "偏响→厚重有力 / 偏轻→压制感强"

    c("情绪标签不再自相矛盾", emotion_consistency)

    def edge_cases():
        """极短音频不应崩溃."""
        import soundfile as sf

        p = config.WORK_DIR / "selftest_short.wav"
        sf.write(str(p), np.zeros(4000, dtype="float32"), 22050)   # 0.18s
        try:
            a = M.analyze_music(p)
            return f"0.18s 音频 -> {len(a.segments)} 段, 未崩溃"
        except ValueError:
            return "0.18s 音频被正确拒绝 (ValueError)"
        finally:
            p.unlink(missing_ok=True)

    c("极短音频边界处理", edge_cases)

    def save_load_analysis():
        a = holder["synth"]
        out = config.WORK_DIR / "selftest_music.json"
        M.save_analysis(a, out)
        d = json.loads(out.read_text(encoding="utf-8"))
        is_true(d["bpm"] > 0, "序列化后 BPM 丢失")
        is_true(len(d["segments"]) == len(a.segments), "段落数不一致")
        is_true("dynamic_range" in d, "缺少 dynamic_range")
        out.unlink(missing_ok=True)
        return f"{len(d['segments'])} 段已序列化"

    c("音乐分析 JSON 往返", save_load_analysis)

    def curve_endpoints_honest():
        """平滑曲线不得被零填充压出假淡入/假淡出 (确定性验证).

        真实缺陷: `np.convolve(arousal, kern, mode="same")` 把窗口外的部分当 0,
        于是曲线首尾各约 1 秒被压低 (实测真歌曲首点低 48%、末点低 98%), 而
        start/end 附近的 |diff| 被抬高 —— 调用方若传小 min_segment 就会在
        这些位置选出**假边界**。

        验证不依赖真实音频的归一化数值, 而是按**规格**逐点复算首尾两点:
        窗口越界的那部分必须用**边界值本身**补齐 (edge padding),
        即 out[0] = mean(edge 份数×x[0] + 真实样本) —— 这样常数序列保持常数,
        首尾不会被凭空压低。再对合成曲的导出曲线做量级检查。
        """
        win = max(int(2.0 * M.FRAMES_PER_SEC), 8)
        half = win // 2
        x = np.linspace(0.2, 0.8, 200)
        got = M.smooth_frame(x, win)

        def spec(sig: np.ndarray, i: int) -> float:
            """逐点复算: 窗口 = [i-half, i+half], 越界处用边界值补齐."""
            n = len(sig)
            l, r = i - half, i + half
            part = sig[max(l, 0):min(r + 1, n)]
            pad = (r - l + 1) - part.size
            edge = sig[0] if l < 0 else sig[-1]
            return float((np.sum(part) + pad * edge) / (2 * half + 1))

        approx(float(got[0]), spec(x, 0), 1e-12)
        approx(float(got[-1]), spec(x, len(x) - 1), 1e-12)
        for i in (0, 1, half - 1, half, len(x) - half, len(x) - 2, len(x) - 1):
            approx(float(got[i]), spec(x, i), 1e-12)
        # 常数序列必须逐点不变 —— 零填充会把首尾压低约 half/win
        flat = np.full(200, 0.5)
        approx(float(np.max(np.abs(M.smooth_frame(flat, win) - flat))), 0.0, 1e-12)
        # 旧实现 (零填充) 的首点偏差, 用来证明用例有区分度
        old_dev = abs(float(np.convolve(x, np.ones(win) / win, mode="same")[0]) - x[0])
        is_true(old_dev > 0.02, f"旧实现偏差过小, 用例无区分度: {old_dev:.4f}")
        # 极端信号 (首点最响) 下, 新实现必须保住首点的量级, 旧实现会掉到 1/win
        spike = np.zeros(400)
        spike[:20] = 1.0
        new_spike = M.smooth_frame(spike, win)
        old_spike = np.convolve(spike, np.ones(win) / win, mode="same")
        is_true(new_spike[0] > 0.5 * spike[0],
                f"首点最响时新实现仍压低首点: {new_spike[0]:.3f}")
        is_true(old_spike[0] < 0.3 * spike[0],
                f"旧实现没有明显压低首点, 用例无区分度: {old_spike[0]:.3f}")
        return (f"首尾与规格一致 (7 点), 常值不变; 旧实现首点偏差 {old_dev:.4f}; "
                f"首点最响 {spike[0]:.1f} -> 新 {new_spike[0]:.3f} / 旧 {old_spike[0]:.3f}")

    c("能量曲线首尾无人工淡入淡出", curve_endpoints_honest)

    def curve_hz_is_real():
        """energy_curve_hz 必须等于**实现**的采样率, 而不是期望值.

        真实缺陷: `step = int(43.0664/2)` = 21 -> 真实 2.0508Hz, 旧代码却硬写
        2.0, 任何用 i/hz 反推时间的下游都会累积漂移 (300s 曲子约 3.7s)。
        """
        a = holder["synth"]
        step = max(int(M.FRAMES_PER_SEC / 2.0), 1)
        approx(a.energy_curve_hz, M.FRAMES_PER_SEC / step, 1e-9)
        is_true(abs(a.energy_curve_hz - 2.0) > 1e-6,
                f"合成参数下真实采样率就是 2.0 的话这个用例没有意义: {a.energy_curve_hz}")
        # 用声明的 hz 反推曲线覆盖的时长, 必须与音频时长一致
        derived = len(a.energy_curve) / a.energy_curve_hz
        is_true(abs(derived - a.duration) <= 2.0,
                f"按 curve_hz 反推时长 {derived:.1f}s 与音频 {a.duration:.1f}s 不符")
        return (f"真实 {a.energy_curve_hz:.4f}Hz, {len(a.energy_curve)} 点 -> "
                f"反推 {derived:.1f}s vs 音频 {a.duration:.1f}s")

    c("能量曲线采样率与实现一致", curve_hz_is_real)

    def beat_and_segment_consistency():
        """拍率与段内 BPM 不得自相矛盾."""
        a = holder["synth"]
        bt = a.beat_times
        # beats_per_second 必须与 bpm 同量纲 (旧实现拿整曲时长当分母, 无拍的
        # 前奏/尾奏会把它压低: 实测 1.825 vs bpm 117.5, 隐含 109.5 BPM)
        is_true(abs(a.beats_per_second * 60.0 - a.bpm) < 12.0,
                f"拍率 {a.beats_per_second:.3f} 与 bpm {a.bpm:.1f} 不一致")
        span = bt[-1] - bt[0]
        approx(a.beats_per_second, (len(bt) - 1) / span, 1e-6)
        # 段内 BPM: 允许为 0 (整段无拍), 但不允许出现"塌陷"的低值 ——
        # 旧实现 `段内拍数/段时长` 在淡出段实测掉到 32 (全局 117.5)
        low = [(s.index, round(s.bpm_local, 1)) for s in a.segments
               if 0 < s.bpm_local < a.bpm * 0.6]
        is_true(not low, f"段内 BPM 塌陷: {low} (全局 {a.bpm:.1f})")
        # 段末不得超出音频时长 (帧数/FPS 会多出约一个 hop)
        for s in a.segments:
            is_true(s.end <= a.duration + 1e-6,
                    f"段落 {s.index} 末尾 {s.end} 超出音频时长 {a.duration:.3f}")
        return (f"bps {a.beats_per_second:.3f} ~ {a.beats_per_second * 60:.1f}BPM "
                f"(全局 {a.bpm:.1f}); 段内 BPM "
                f"{min(s.bpm_local for s in a.segments):.1f}~"
                f"{max(s.bpm_local for s in a.segments):.1f}")

    c("拍率与段内 BPM 不自相矛盾", beat_and_segment_consistency)

    def segment_floor_respected():
        """吸附到拍点后, 每段仍需满足 min_segment.

        真实缺陷: 吸附被拒时只回退**当前**边界, 却保留了上一个已吸附的边界,
        于是最小值约束被绕过 (实测 min_segment=6s 切出 5.5s 的段)。
        """
        raw = np.concatenate([
            np.full(int(5 * M.SAMPLE_RATE), 0.05, dtype="float32"),
            np.full(int(9 * M.SAMPLE_RATE), 0.45, dtype="float32"),
            np.full(int(5 * M.SAMPLE_RATE), 0.10, dtype="float32"),
            np.full(int(9 * M.SAMPLE_RATE), 0.50, dtype="float32"),
            np.full(int(5 * M.SAMPLE_RATE), 0.05, dtype="float32"),
        ])
        p = config.WORK_DIR / "selftest_minseg.wav"
        import soundfile as sf
        sf.write(str(p), raw, M.SAMPLE_RATE)
        try:
            a = M.analyze_music(p, min_segment=6.0)
        finally:
            p.unlink(missing_ok=True)
        shortest = min(s.duration for s in a.segments)
        is_true(shortest >= 6.0 - 0.05,
                f"出现短于 min_segment 的段: {shortest:.2f}s "
                f"({[(s.start, s.end) for s in a.segments]})")
        return f"{len(a.segments)} 段, 最短 {shortest:.2f}s (下限 6.0s)"

    c("吸附后仍满足最短段长", segment_floor_respected)

    if real_music and real_music.is_file():
        def real_analysis():
            a = M.analyze_music(real_music)
            is_true(a.duration > 20, "真实曲时长异常")
            is_true(3 <= len(a.segments) <= 20, f"段数异常: {len(a.segments)}")
            # 段边界必须递增且覆盖全曲
            prev = 0.0
            for s in a.segments:
                is_true(s.start >= prev - 1e-6, "段落边界非递增")
                is_true(s.end > s.start, "段落时长为非正")
                prev = s.end
            approx(a.segments[-1].end, a.duration, 0.5)
            # 边界应吸附到拍点附近 (允许未吸附的退化情况)
            bt = np.asarray(a.beat_times)
            near = 0
            for s in a.segments[1:]:
                if bt.size and np.min(np.abs(bt - s.start)) < 0.05:
                    near += 1
            return (f"{real_music.name}: {a.duration:.0f}s, {len(a.segments)} 段, "
                    f"BPM {a.bpm:.1f}, 边界卡拍 {near}/{len(a.segments) - 1}, "
                    f"动态范围 {a.extra.get('dynamic_range')}")

        c("真实音乐分析", real_analysis)


# ==================================================================
# 3. demo 解析与亮点抽取
# ==================================================================
def test_demo(c: Check) -> None:
    from . import demo as D

    holder: dict = {}

    def parse_demo():
        t0 = time.time()
        dem = D.load_demo(str(config.DEFAULT_DEMO), with_ticks=True)
        holder["demo"] = dem
        is_true(dem.rounds.height > 0, "回合表为空")
        is_true(dem.kills.height > 0, "击杀表为空")
        is_true(dem.ticks.height > 1000, "逐 tick 表为空")
        return (f"{dem.rounds.height} 回合, {dem.kills.height} 击杀, "
                f"{dem.ticks.height} tick 行, {time.time() - t0:.1f}s")

    c("解析 demo", parse_demo)

    def cache_hit():
        t0 = time.time()
        D.load_demo(str(config.DEFAULT_DEMO), with_ticks=True)
        dt = time.time() - t0
        is_true(dt < 1.0, f"缓存未命中, 耗时 {dt:.2f}s")
        return f"二次加载 {dt * 1000:.0f}ms"

    c("demo 解析缓存生效", cache_hit)

    def kill_events():
        ev = D.extract_kill_events(holder["demo"])
        is_true(len(ev) > 50, f"击杀事件过少: {len(ev)}")
        ticks = [e.tick for e in ev]
        is_true(ticks == sorted(ticks), "击杀事件未按 tick 排序")
        is_true(all(e.attacker for e in ev), "存在 attacker 为空的事件")
        # 事件分类必须覆盖所有条目, 且 world 伪玩家不得出现在"可建卡"类别里
        kinds = Counter(e.kind for e in ev)
        bad = [e.kind for e in ev if e.kind not in
               ("player_kill", "suicide_or_fall", "non_player_death")]
        is_true(not bad, f"存在未分类事件: {Counter(bad)}")
        is_true(not any(e.kind == "player_kill" and e.attacker == "world" for e in ev),
                "world 伪玩家被归为可建卡的击杀")
        return f"{len(ev)} 条击杀 {dict(kinds)}"

    c("击杀事件抽取、排序与分类", kill_events)

    def no_pseudo_player_cards():
        """不得出现 player="world"/空阵营的卡片, 且非玩家死亡不得进入卡片.

        真实缺陷: `attacker or "world"` 会为自杀/坠落造出 player="world" 的卡;
        若它恰好是回合首个事件, 还会把 opening_kill(+34) 送给这张 world 卡。
        同时"非玩家实体死亡"(打鸡等)必须被丢弃, 否则会污染 per_player。
        """
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=40)
        world = [c["id"] for c in res["highlights"] if c["player"] in ("world", "")]
        is_true(not world, f"出现 world/空玩家卡片: {world}")
        no_side = [c["id"] for c in res["highlights"] if c["player_side"] not in ("t", "ct")]
        is_true(not no_side, f"卡片缺少合法阵营: {no_side}")
        # 每张卡的击杀必须都是真实玩家击杀
        for c in res["highlights"]:
            for k in c["kills"]:
                is_true(k.get("kind", "player_kill") == "player_kill",
                        f"{c['id']} 含非玩家击杀: {k.get('kind')}")
        sk = res["skipped_events"]
        is_true(set(sk) >= {"non_player_death", "suicide_or_fall"},
                f"缺少事件分类计数: {sk}")
        return (f"{res['highlight_count']} 卡无 world; 丢弃 {sk}")

    c("无 world 伪玩家卡片", no_pseudo_player_cards)

    def clutch_scoring_consistent():
        """残局加分必须与 find_clutch 的判定一致 (不得因分组而丢失).

        真实缺陷: find_clutch 只要求残局窗口内 >=1 杀, 而调用方额外要求
        "这一分组 >=2 杀"。于是 1v2 靠 1 杀 + 拆包/超时赢下的残局拿不到 +80;
        更糟的是 12s 分组把同一次残局切成 1+1 时两组都 <2, 整个残局的加分与
        标签**全部丢失**。
        """
        cards = D.cards_from_dicts(
            D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=40)["highlights"])
        # score_highlight 本身支持单杀残局 (这是判定的基准)
        single = D.KillEvent(tick=100, attacker="A", attacker_side="t", victim="B",
                             victim_side="ct", weapon="ak47", headshot=False)
        sc_plain, tags_plain = D.score_highlight([single], is_clutch=False, is_opening=False)
        sc_clutch, tags_clutch = D.score_highlight([single], is_clutch=True, is_opening=False)
        approx(sc_clutch - sc_plain, D.SCORE_CLUTCH_WIN, 1e-6)
        is_true("clutch" in tags_clutch, f"单杀残局没有 clutch 标签: {tags_clutch}")
        # 真 demo 上被标为残局的卡片必须带 clutch 标签与敌人数
        clutched = [c for c in cards if c.clutch_enemies]
        for c in clutched:
            is_true("clutch" in c.tags, f"{c.id} 有残局敌人数却没有 clutch 标签")
            is_true(c.clutch_enemies >= 2, f"{c.id} 残局敌人数异常: {c.clutch_enemies}")
        return (f"单杀残局 {sc_plain:.0f}->{sc_clutch:.0f}; "
                f"真 demo 残局卡 {len(clutched)} 张")

    c("残局加分与判定口径一致", clutch_scoring_consistent)

    def tag_mapping_explicit():
        """多杀标签必须是显式映射: 3 杀 = triple (旧代码把 3 杀标成 multi)."""
        def mk(n):
            return [D.KillEvent(tick=100 + i, attacker="A", attacker_side="t",
                                victim=f"B{i}", victim_side="ct", weapon="ak47",
                                headshot=False)
                    for i in range(n)]
        got = {}
        for n in (2, 3, 4, 5):
            _, tags = D.score_highlight(mk(n), is_clutch=False, is_opening=False)
            got[n] = next((t for t in tags if t in
                           ("double", "triple", "quad", "ace", "multi")), None)
        is_true(got[2] == "double", f"2 杀标签错误: {got}")
        is_true(got[3] == "triple", f"3 杀标签错误 (旧代码给 multi): {got}")
        is_true(got[4] == "quad", f"4 杀标签错误: {got}")
        is_true(got[5] == "ace", f"5 杀标签错误: {got}")
        return f"{got}"

    c("多杀标签映射显式", tag_mapping_explicit)

    def long_range_unit_is_meters():
        """long_range 阈值必须是米制 (旧阈值 1500 是死代码)."""
        far = D.KillEvent(tick=100, attacker="A", attacker_side="t", victim="B",
                          victim_side="ct", weapon="awp", headshot=False,
                          distance=D.LONG_RANGE_M + 1.0)
        near = D.KillEvent(tick=101, attacker="A", attacker_side="t", victim="C",
                           victim_side="ct", weapon="ak47", headshot=False,
                           distance=10.0)
        _, tags_far = D.score_highlight([far], is_clutch=False, is_opening=False)
        _, tags_near = D.score_highlight([near], is_clutch=False, is_opening=False)
        is_true("long_range" in tags_far, f"{far.distance}m 未判为远距离: {tags_far}")
        is_true("long_range" not in tags_near, f"10m 被判为远距离: {tags_near}")
        is_true(0 < D.LONG_RANGE_M < 200,
                f"阈值 {D.LONG_RANGE_M} 不像米制 (实测击杀最大 67m)")
        return f"阈值 {D.LONG_RANGE_M}m: {far.distance}m -> 命中, 10m -> 不命中"

    c("远距离阈值是米制", long_range_unit_is_meters)

    def post_pad_not_clipped():
        """尾留白必须用 official_end 做上界 (否则制胜击杀卡没有收尾).

        真实缺陷: 多数回合 `rounds.end == 最后一杀 tick`, 于是
        `end = min(lastkill+POST_PAD, rend)` 把 2.2s 尾留白整段吃掉。
        """
        dem = holder["demo"]
        bounds = D._round_bounds(dem)
        is_true(any(b[2] > b[1] for b in bounds.values()),
                "official_end 全都不晚于 end, 说明上界没生效")
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=40)
        past = [c["id"] for c in res["highlights"]
                if c["end_tick"] > bounds[c["round_num"]][1]
                and c["end_tick"] <= bounds[c["round_num"]][2]]
        beyond = [c["id"] for c in res["highlights"]
                  if c["end_tick"] > bounds[c["round_num"]][2]]
        is_true(not beyond, f"卡片超出官方回合结束: {beyond}")
        return f"{len(past)} 张卡用上了尾留白 (共 {len(res['highlights'])} 张)"

    c("尾留白不被回合末截断", post_pad_not_clipped)

    def max_cards_zero_means_zero():
        """max_cards=0 必须是"不要卡片", 不能变成"不截断"."""
        zero = D.build_highlights(holder["demo"], max_cards=0, with_utility=False)
        is_true(zero == [], f"max_cards=0 却返回 {len(zero)} 张卡")
        one = D.build_highlights(holder["demo"], max_cards=1, with_utility=False)
        is_true(len(one) == 1, f"max_cards=1 返回 {len(one)} 张")
        return "0 -> 0 张, 1 -> 1 张"

    c("max_cards=0 语义正确", max_cards_zero_means_zero)

    def highlights():
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=40)
        is_true(res["highlight_count"] > 0, "没有抽出任何亮点")
        cards = D.cards_from_dicts(res["highlights"])
        holder["cards"] = cards
        # 评分必须降序
        scores = [x.score for x in cards]
        is_true(scores == sorted(scores, reverse=True), "亮点未按评分降序")
        # 时间窗合法
        for x in cards:
            is_true(x.end_tick > x.start_tick, f"{x.id} 时间窗非法")
            is_true(x.duration > 0.5, f"{x.id} 时长过短")
            is_true(x.score >= 25.0, f"{x.id} 低于评分阈值")
        return f"{len(cards)} 条, 最高 {cards[0].score:.0f}"

    c("亮点抽取与评分排序", highlights)

    def same_round_limit():
        cards = holder["cards"]
        cnt = Counter(x.round_num for x in cards)
        worst = cnt.most_common(1)[0]
        is_true(worst[1] <= 2, f"同一回合素材过多: 回合 {worst[0]} 有 {worst[1]} 条")
        return f"单回合最多 {worst[1]} 条"

    c("同回合素材不过度集中", same_round_limit)

    def utility_attached():
        cards = holder["cards"]
        with_util = [x for x in cards if x.utility]
        is_true(len(with_util) > 0, "没有任何素材带道具事件")
        total = sum(len(x.utility) for x in cards)
        kinds = {u["type"] for x in cards for u in x.utility}
        return f"{total} 个道具事件, 类型 {sorted(kinds)}"

    c("道具事件随素材附带", utility_attached)

    def json_roundtrip():
        cards = holder["cards"][:5]
        payload = [x.to_dict() for x in cards]
        text = json.dumps(payload, ensure_ascii=False)
        back = D.cards_from_dicts(json.loads(text))
        is_true(len(back) == len(cards), "往返后数量不一致")
        for a, b in zip(cards, back):
            is_true(a.id == b.id, "id 不一致")
            is_true(a.start_tick == b.start_tick, "start_tick 不一致")
            is_true(len(a.kills) == len(b.kills), "击杀数不一致")
        return f"{len(back)} 条往返一致"

    c("亮点卡片 JSON 往返", json_roundtrip)


# ==================================================================
# 4. 编排 (EDL)
# ==================================================================
def test_planner(c: Check) -> None:
    from types import SimpleNamespace

    from . import demo as D, music as M, planner as P

    a = M.analyze_music(config.WORK_DIR / "selftest_track.wav")
    res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=30)
    cards = D.cards_from_dicts(res["highlights"])
    holder: dict = {"analysis": a, "cards": cards}

    def allocation_exact():
        """分配器必须精确填满, 且每条落在 [lo, hi] 内 —— 这是修过的关键缺陷."""
        d = P._allocate_durations(24.0, 6, 2.0, 5.0)
        approx(sum(d), 24.0, 1e-6)
        is_true(all(2.0 - 1e-6 <= x <= 5.0 + 1e-6 for x in d), f"越界: {d}")
        # 无法填满时 (上限不够) 不得超出, 也不得越界
        d2 = P._allocate_durations(100.0, 3, 2.0, 5.0)
        is_true(sum(d2) <= 15.0 + 1e-6, f"超出容量: {sum(d2)}")
        is_true(all(x <= 5.0 + 1e-6 for x in d2), f"越界: {d2}")
        return f"精确 {sum(d):.3f}/24.0, 容量受限时 {sum(d2):.1f}/15.0"

    c("镜头时长精确分配", allocation_exact)

    def allocation_uniform():
        d = P._allocate_durations(18.0, 3, 3.0, 9.0)
        approx(sum(d), 18.0, 1e-6)
        is_true(max(d) - min(d) < 1e-6, f"应均分: {d}")
        return f"{[round(x, 2) for x in d]}"

    c("段落内镜头均分", allocation_uniform)

    def clips_for_span_monotone():
        """时长越长, 允许的镜头数不得减少."""
        prev = 0
        for span in (5, 10, 20, 40, 80):
            n = P._clips_for_span(float(span), 0.9, 99)
            is_true(n >= prev, f"span={span} 时条数反而减少")
            prev = n
        return "单调递增"

    c("镜头数随时长单调", clips_for_span_monotone)

    def fallback_plan_ok():
        edl = P.fallback_plan(a, cards, max_clips=14)
        holder["edl"] = edl
        is_true(len(edl.clips) > 0, "没有生成任何剪辑")
        is_true(edl.total_duration > 0, "总时长为 0")
        is_true(edl.total_duration <= edl.music_duration + 1e-6, "超出音乐长度")
        return (f"{len(edl.clips)} 段, {edl.total_duration:.1f}s / "
                f"{edl.music_duration:.1f}s")

    c("确定性编排可用", fallback_plan_ok)

    def timeline_contiguous():
        edl = holder["edl"]
        prev = None
        for cl in edl.clips:
            is_true(cl.duration > 0, f"clip{cl.index} 时长非正")
            if prev is not None:
                gap = cl.out_start - prev
                is_true(abs(gap) < 0.05, f"clip{cl.index} 有 {gap:+.3f}s 缝隙")
            prev = cl.out_end
        approx(edl.clips[0].out_start, 0.0, 0.02)
        return f"0 -> {edl.clips[-1].out_end:.2f}s 无缝隙"

    c("时间轴首尾相接", timeline_contiguous)

    def no_duplicate_highlights():
        edl = holder["edl"]
        ids = [cl.highlight_id for cl in edl.clips]
        is_true(len(set(ids)) == len(ids), f"素材被重复使用: {len(ids) - len(set(ids))} 次")
        return f"{len(ids)} 段全部唯一"

    c("素材不重复使用", no_duplicate_highlights)

    def source_ranges_valid():
        edl = holder["edl"]
        by_id = {x.id: x for x in cards}
        for cl in edl.clips:
            card = by_id.get(cl.highlight_id)
            is_true(card is not None, f"素材不存在: {cl.highlight_id}")
            is_true(cl.src_end_tick > cl.src_start_tick, f"clip{cl.index} 源区间非法")
            is_true(cl.src_start_tick >= card.start_tick, f"clip{cl.index} 源起点越界")
            is_true(cl.src_end_tick <= card.end_tick, f"clip{cl.index} 源终点越界")
        return f"{len(edl.clips)} 段源区间全部合法"

    c("源 tick 区间不越界", source_ranges_valid)

    def edl_json_roundtrip():
        edl = holder["edl"]
        p = config.WORK_DIR / "selftest_edl.json"
        edl.save(p)
        back = P.EDL.load(p)
        is_true(len(back.clips) == len(edl.clips), "往返后段数不一致")
        approx(back.total_duration, edl.total_duration, 1e-6)
        for x, y in zip(edl.clips, back.clips):
            is_true(x.highlight_id == y.highlight_id, "素材 id 不一致")
            is_true(x.src_start_tick == y.src_start_tick, "源区间不一致")
        p.unlink(missing_ok=True)
        return f"{len(back.clips)} 段往返一致"

    c("EDL JSON 往返", edl_json_roundtrip)

    def validator_rejects_bad():
        """校验器必须丢弃非法输入 (越界/不存在素材/零时长)."""
        good = holder["edl"].clips[0]
        by_id = {x.id: x for x in cards}
        bad = [
            P.EDLClip(index=0, music_segment=0, out_start=0.0, out_end=2.0,
                      highlight_id="不存在的素材", src_start_tick=0, src_end_tick=100),
            P.EDLClip(index=1, music_segment=0, out_start=2.0, out_end=4.0,
                      highlight_id=good.highlight_id,
                      src_start_tick=good.src_start_tick - 99999,
                      src_end_tick=good.src_end_tick + 99999),
        ]
        fixed, notes = P.validate_edl(bad, by_id, a.duration)
        is_true(len(fixed) < len(bad), "非法剪辑未被丢弃")
        is_true(any("不存在" in n for n in notes), "未记录丢弃原因")
        return f"丢弃 {len(bad) - len(fixed)} 条, {len(notes)} 条说明"

    c("EDL 校验器拦截非法输入", validator_rejects_bad)

    def split_overlong_clips():
        """超长镜头必须被拆开, 且返回新列表.

        真实缺陷: `_split_overlong_clips` 原先只返回说明字符串, 拆分结果被
        丢弃, 调用方写成 `notes = _split(...)` 就等于什么都没做。后果是
        **最激烈的段落反而镜头最长** —— 实测 LLM 路径高激烈段平均 10.41s,
        其余 7.66s (比值 1.24), 节奏与音乐完全相反。

        这个用例之所以必须存在, 是因为该 bug 不会报错、只会让成片变难看。
        """
        from .planner import EDLClip, _segment_limits, _split_overlong_clips

        class FakeSeg:
            arousal = 1.0
            rel_energy = 0.2
            label = "drop"

        class FakeCard:
            def __init__(self, cid, start, end):
                self.id = cid
                self.start_tick = start
                self.end_tick = end
                self.round_num = 1
                self.score = 50.0
                self.tags = []
                self.places = []
                self.kills = []
                self.player = "p"
                self.player_side = "t"

        # 上限必须按**测试实际使用的 pacing** 推算。
        # 原先写死 `_segment_limits(1.0)[2]` (默认 balanced), 而 _split 现在会
        # 收到 pacing —— 万一 pacing 挂钩坏了, 这个用例反而发现不了。
        hot_hi = _segment_limits(1.0, 0.2, "fast")[2]
        base = FakeCard("base", 1000, 2000)
        spares = [FakeCard(f"s{i}", 3000 + i * 100, 4000 + i * 100) for i in range(3)]
        by_id = {"base": base, **{c.id: c for c in spares}}
        clip = EDLClip(index=0, music_segment=0, out_start=0.0, out_end=15.0,
                       highlight_id="base", src_start_tick=1000, src_end_tick=2000,
                       speed=1.0, transition=None, reason="test")

        res = _split_overlong_clips([clip], [FakeSeg()], by_id,
                                    spare_by_seg={}, spare_global=list(spares),
                                    pacing="fast")
        is_true(isinstance(res, tuple) and len(res) == 2,
                "必须返回 (clips, notes) 元组, 否则拆分结果会被丢弃")
        new_clips, notes = res
        is_true(len(new_clips) >= 3, f"15s 镜头未被拆开, 仍是 {len(new_clips)} 条")
        is_true(len(notes) >= 1, "未记录拆分说明")
        # 每条都不该超过该段上限 (允许 15% 余量)
        for c in new_clips:
            is_true(c.duration <= hot_hi * 1.15 + 1e-6,
                    f"拆分后仍有 {c.duration:.2f}s 超过上限 {hot_hi}")
        # 时间轴必须连续且总时长不变
        approx(sum(c.duration for c in new_clips), 15.0, 0.02)
        for a, b in zip(new_clips, new_clips[1:]):
            approx(a.out_end, b.out_start, 0.02)
        # 应尽量换用不同素材
        used = {c.highlight_id for c in new_clips}
        is_true(len(used) >= 2, f"拆分未换素材, 仍是同一段画面: {used}")

        # 无富余素材时也必须能拆 (退化为同素材不同源片段)
        res2 = _split_overlong_clips([clip], [FakeSeg()], by_id,
                                     spare_by_seg={}, spare_global=[],
                                     pacing="fast")
        c2, _ = res2
        is_true(len(c2) >= 3, f"无富余素材时未拆分: {len(c2)} 条")
        for c in c2:
            is_true(c.src_end_tick > c.src_start_tick, "源区间非法")

        # pacing 必须真的影响拆分的粒度: cinematic 上限更高 -> 拆得更少
        n_fast = len(_split_overlong_clips(
            [clip], [FakeSeg()], by_id, spare_by_seg={},
            spare_global=list(spares), pacing="fast")[0])
        n_cine = len(_split_overlong_clips(
            [clip], [FakeSeg()], by_id, spare_by_seg={},
            spare_global=list(spares), pacing="cinematic")[0])
        is_true(n_cine < n_fast,
                f"pacing 未影响拆分粒度: fast={n_fast} cinematic={n_cine}")
        return (f"15.00s -> {len(new_clips)} 条 (换素材 {len(used)} 种), "
                f"无富余时 {len(c2)} 条, pacing fast={n_fast}/cinematic={n_cine}")

    c("超长镜头拆分 (节奏倒挂回归)", split_overlong_clips)

    def pacing_not_inverted():
        """成片节奏必须与音乐一致: 高激烈段的镜头应更短."""
        edl = holder["edl"]
        segs = {s.index: s for s in a.segments}
        hot, calm = [], []
        for cl in edl.clips:
            ar = segs[cl.music_segment].arousal
            (hot if ar >= 0.66 else calm).append(cl.duration)
        if not hot or not calm:
            return f"段落分布不足 (高激烈 {len(hot)} 条), 跳过比值检查"
        rh, rc = sum(hot) / len(hot), sum(calm) / len(calm)
        # 素材池极小时可能无法做到, 但正常情况必须 < 1
        is_true(rh <= rc * 1.05,
                f"高激烈段镜头 {rh:.2f}s 反而长于其余 {rc:.2f}s (节奏倒挂)")
        return f"高激烈 {rh:.2f}s vs 其余 {rc:.2f}s (比值 {rh / rc:.2f})"

    c("节奏不与音乐倒挂", pacing_not_inverted)

    def max_clips_is_real_cap():
        """max_clips 必须是**最终段数**的硬上限, 不是候选池大小.

        真实缺陷: 原先 max_clips 只用来削减候选池, 完全不约束最终段数 ——
        用户设了 4, 实际还是出 27 段 (每段仍按"铺满音乐"推导条数)。偏好把它
        存下来也毫无意义。修的时候还踩了第二个坑: 额度若按 arousal 分配, 时间上
        靠后的高激烈段会先把额度拿走, 前面的段落拿不到 -> 时间轴出现**内部空隙**。
        所以额度必须按时间顺序耗尽, 空隙只允许出现在尾部。
        """
        from . import music as M
        from .planner import plan_edit

        track = config.WORK_DIR / "selftest_track.wav"
        if not track.is_file():
            M.make_test_track(track)
        a = M.analyze_music(track)
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=40)
        cards = D.cards_from_dicts(res["highlights"])

        detail = []
        for mc in (4, 8, 14):
            edl = plan_edit(a, cards, use_llm=False, max_clips=mc)
            is_true(len(edl.clips) <= mc,
                    f"max_clips={mc} 却出了 {len(edl.clips)} 段")
            gaps = [i for i, c in enumerate(edl.clips)
                    if i and abs(c.out_start - edl.clips[i - 1].out_end) >= 0.05]
            is_true(not gaps, f"max_clips={mc} 时时间轴内部有空隙: {gaps}")
            approx(edl.clips[0].out_start, 0.0, 0.02)
            for x, y in zip(edl.clips, edl.clips[1:]):
                is_true(y.out_start >= x.out_end - 1e-6, "时间轴倒退")
            detail.append(f"{mc}->{len(edl.clips)}")
        # 不设上限时应铺满整首歌 (上限只在设了之后才截断)
        free = plan_edit(a, cards, use_llm=False, max_clips=40)
        is_true(abs(free.total_duration - free.music_duration) < 0.6,
                f"未设上限时未铺满: {free.total_duration:.1f}/{free.music_duration:.1f}")
        detail.append(f"40->{len(free.clips)}(铺满)")
        return " ".join(detail)

    c("max_clips 是最终段数硬上限", max_clips_is_real_cap)

    def prefs_reach_prompt():
        """用户偏好必须进提示词, 且提示词区间要包住代码实际夹取的范围.

        为什么必须进提示词: 偏好如果只在代码里事后夹取, LLM 会按自己的风格排出
        一个"超限方案"再被硬裁 —— 结果既不是它想排的, 也不完全符合用户偏好。
        把节奏、段数上限、单条时长区间提前告诉它, 才能一次排对。

        为什么区间必须**向外取整**: 最初用 round(), 提示词报的下界 (1.1) 比实际
        可夹取的下界 (1.05) 更窄, LLM 严格照提示词排反而会被代码判越界。
        """
        from . import music as M
        from . import planner as P

        track = config.WORK_DIR / "selftest_track.wav"
        if not track.is_file():
            M.make_test_track(track)
        a = M.analyze_music(track)
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=12)
        cards = D.cards_from_dicts(res["highlights"])

        # --- 提示词里确实有偏好 ---
        prefs = {"pacing": "cinematic", "max_clips": 9,
                 "show_hud": False, "shake_px": 0.0, "fade_in": 0.0, "fade_out": 0.0}
        prompt = P._build_user_prompt(a, cards, max_clips=9, prefs=prefs)
        is_true('"user_preferences"' in prompt, "提示词缺少 user_preferences 段")
        is_true('"cinematic"' in prompt, "提示词缺少节奏偏好")
        is_true('"max_clips": 9' in prompt, "提示词缺少段数上限")
        is_true("电影感" in prompt, "提示词缺少节奏的自然语言说明")
        is_true("不显示 HUD" in prompt, "画面偏好未进提示词")
        is_true("user_preferences" in P.SYSTEM_PROMPT,
                "系统提示词未引用 user_preferences")
        is_true("优先于" in P.SYSTEM_PROMPT, "系统提示词未说明偏好优先级")

        # --- 提示词区间 >= 代码实际夹取区间 (三种节奏都要成立) ---
        bad = []
        for pacing in ("fast", "balanced", "cinematic"):
            brief = P.build_preference_brief({"pacing": pacing}, a.segments,
                                             max_clips=22)
            rng = brief["clip_duration_range"]
            for seg in a.segments:
                _, lo, hi = P._segment_limits(seg.arousal, seg.rel_energy, pacing)
                label = seg.label
                if label not in rng:
                    continue
                blo, bhi = rng[label]
                if not (blo <= lo + 1e-9 and hi <= bhi + 1e-9):
                    bad.append((pacing, seg.index, label, lo, hi, blo, bhi))
        is_true(not bad, f"提示词区间比实际夹取更窄 (LLM 会排出越界方案): {bad[:3]}")

        # --- 节奏偏好必须真的改变成片节奏 ---
        stats = {}
        for pacing in ("fast", "balanced", "cinematic"):
            edl = P.plan_edit(a, cards, use_llm=False, max_clips=40,
                              prefs={"pacing": pacing})
            durs = [c.duration for c in edl.clips]
            stats[pacing] = sum(durs) / len(durs)
        is_true(stats["fast"] < stats["cinematic"],
                f"节奏偏好没生效: fast 平均 {stats['fast']:.2f}s 未短于 "
                f"cinematic {stats['cinematic']:.2f}s")
        return (f"提示词含偏好 + 三种节奏区间一致; "
                f"平均单条 fast={stats['fast']:.2f}s "
                f"balanced={stats['balanced']:.2f}s "
                f"cinematic={stats['cinematic']:.2f}s")

    c("用户偏好进提示词且口径一致", prefs_reach_prompt)

    def preference_inference():
        """从对话推断偏好: 抽取、投票、置信度、进提示词.

        这段逻辑踩过两个坑, 所以用例要盯住:
          1. 饱和度分母最初设成 3.0 (要求"权重累到 3 才算证据充分"), 结果用户
             明确下达的指令也只得到 0.3 分 —— 恰好卡在进提示词的阈值上, 一条
             结论都进不去, 整个功能形同虚设。
          2. 反过来若只按权重算, 单条证据又会直接给到 1.0 (等于完全确定),
             这也不对。所以要加"证据数量天花板"。
        """
        import shutil

        from . import profile as PF
        from . import store as S

        tmp = config.WORK_DIR / "_profiletest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        db = tmp / "p.db"

        # --- 抽取: 能从原话里认出偏好, 并保留依据 ---
        hits = PF.extract_observations("先不管git 先做其他")
        dims = {d for d, _v, _q in hits}
        is_true("workflow" in dims, f"没识别出 defer_vcs: {hits}")
        is_true(any(q for _d, _v, q in hits), "证据没有保留原话")

        hits2 = PF.extract_observations(
            "是不是没有实现存储用户信息偏好 实现使用 SQLite在本地目录新增一个文件夹中存储")
        is_true(any(d == "tech_stack" and v == "sqlite_local" for d, v, _q in hits2),
                f"没识别出 sqlite 偏好: {hits2}")

        hits3 = PF.extract_observations("今天天气不错")
        is_true(not hits3, f"无关文本不该产生证据: {hits3}")

        # --- 接受型证据: 用户认可助手方向也算偏好 ---
        got = PF.ingest_exchange(
            "运动模糊太贵，建议默认关闭，保留为可选项", "按你给出的顺序开始", path=db)
        is_true(got.get("motion_fx") == 1, f"未识别接受型证据: {got}")
        # 未表示接受时不该记录
        got2 = PF.ingest_exchange(
            "运动模糊太贵，建议默认关闭", "不行，我要开", path=db)
        is_true(not got2, f"用户没接受却记录了证据: {got2}")

        # --- 置信度: 单条证据不能满分, 也不能低于可用阈值 ---
        PF.ingest_conversation(["实现使用 SQLite在本地目录新增一个文件夹中存储"],
                               speaker="conversation", path=db)
        rows = PF.build_profile(path=db)
        prof = {r["dimension"]: r for r in rows}
        is_true("tech_stack" in prof, f"未汇总出 tech_stage: {list(prof)}")
        c1 = prof["tech_stack"]["confidence"]
        is_true(c1 <= 0.86, f"单条证据置信度过高: {c1}")
        is_true(c1 >= 0.3, f"单条明确指令置信度过低 ({c1}), 进不了提示词")

        # 追加同向证据后应上升
        PF.ingest_conversation(["用 SQLite 落盘存储偏好"],
                               speaker="conversation", path=db)
        rows2 = {r["dimension"]: r for r in PF.build_profile(path=db)}
        is_true(rows2["tech_stack"]["confidence"] > c1,
                "同向证据增加后置信度没上升")

        # --- 进提示词 ---
        brief = PF.profile_brief(path=db)
        is_true("tech_stack" in brief, f"画像未进 brief: {brief}")
        from . import music as M
        from . import planner as P
        track = config.WORK_DIR / "selftest_track.wav"
        if not track.is_file():
            M.make_test_track(track)
        a = M.analyze_music(track)
        res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=6)
        cards = D.cards_from_dicts(res["highlights"])
        prompt = P._build_user_prompt(a, cards, max_clips=8, prefs={}, profile=brief)
        is_true('"user_profile"' in prompt, "提示词里没有 user_profile 段")
        is_true("sqlite" in prompt.lower(), "画像内容没进提示词")
        is_true("user_profile" in P.SYSTEM_PROMPT or "user_preferences" in P.SYSTEM_PROMPT,
                "系统提示词未提及偏好/画像")

        # --- 可遗忘 ---
        S.clear_observations("tech_stack", path=db)
        rows3 = {r["dimension"]: r for r in PF.build_profile(path=db)}
        is_true("tech_stack" not in rows3, "删除证据后结论仍在")

        shutil.rmtree(tmp, ignore_errors=True)
        return (f"抽取 3 例 + 接受型证据 + 置信度 {c1:.2f}→"
                f"{rows2['tech_stack']['confidence']:.2f} + 进提示词 + 可遗忘")

    c("从对话推断偏好", preference_inference)

    def coverage_note_detects_gap():
        from .planner import EDL
        fake = EDL(music_path="x", music_duration=100.0, fps=30,
                   clips=[P.EDLClip(index=0, music_segment=0, out_start=0.0, out_end=60.0,
                                    highlight_id="x", src_start_tick=0, src_end_tick=100)])
        note = P.coverage_note(fake)
        is_true(note is not None and "40" in note, f"未检出覆盖不足: {note}")
        full = EDL(music_path="x", music_duration=60.0, fps=30,
                   clips=[P.EDLClip(index=0, music_segment=0, out_start=0.0, out_end=60.0,
                                    highlight_id="x", src_start_tick=0, src_end_tick=100)])
        is_true(P.coverage_note(full) is None, "完整覆盖却报警")
        return "正确检出 40s 缺口"

    c("覆盖缺口检测", coverage_note_detects_gap)

    def topup_fills_tail():
        """结尾补齐机制 (LLM 常只排到一半)."""
        from .planner import EDL
        by_id = {x.id: x for x in cards}
        partial = EDL(
            music_path="x", music_duration=120.0, fps=30,
            clips=[P.EDLClip(index=0, music_segment=0, out_start=0.0, out_end=30.0,
                             highlight_id=cards[0].id, src_start_tick=cards[0].start_tick,
                             src_end_tick=cards[0].start_tick + 600)],
        )
        clips, notes = P._top_up_tail(partial.clips, by_id, cards,
                                      music_duration=120.0, fps=30)
        end = clips[-1].out_end
        is_true(end > 100.0, f"补齐后仍只到 {end:.1f}s")
        is_true(len(notes) > 0, "未记录补齐动作")
        # 补齐后不能有时间倒流
        prev = None
        for cl in clips:
            if prev is not None:
                is_true(cl.out_start >= prev - 1e-6, "补齐后时间轴倒退")
            prev = cl.out_end
        return f"30s -> {end:.1f}s, {len(clips)} 段"

    c("结尾空白自动补齐", topup_fills_tail)

    def validator_anchors_timeline():
        """首条 out_start > 0 时必须整体前移 —— 渲染端只按 duration 拼接.

        真实缺陷: `_expand` 只为"被挑中的段落"排镜头, 若 LLM 跳过前几段,
        首条 out_start 就是 20s。渲染端 `concat_clips` 只按 Σduration 拼接、
        **从不读 out_start**, 于是成片比时间轴短 20s (视频只有一半长度),
        而 coverage_note 那时还不看这个偏移, 完全静默。
        """
        def seg(i, s, e):
            return SimpleNamespace(index=i, start=s, end=e, duration=e - s,
                                   arousal=0.5, rel_energy=0.0, label="verse")
        def card(cid, s, e):
            return SimpleNamespace(id=cid, start_tick=s, end_tick=e, score=60.0,
                                   round_num=1, tags=[], places=[], kills=[],
                                   player="p", player_side="t")
        segs = [seg(i, i * 10.0, (i + 1) * 10.0) for i in range(4)]
        an = SimpleNamespace(segments=segs, beat_times=[], extra={"has_climax": True},
                             path="x", duration=40.0)
        by_id = {f"c{i}": card(f"c{i}", 1000 + i * 2000, 2000 + i * 2000)
                 for i in range(4)}
        picks = [{"segment": 2, "highlight_id": "c0"},
                 {"segment": 2, "highlight_id": "c1"},
                 {"segment": 3, "highlight_id": "c2"}]
        cls = P._expand(picks, an, by_id, music_duration=40.0, fps=30,
                        snap_beats=True, max_clips=10, pacing="balanced")
        is_true(cls[0].out_start > 1.0,
                f"前提不成立: 首条起点只有 {cls[0].out_start}")
        fixed, notes = P.validate_edl(cls, by_id, 40.0, fps=30)
        approx(fixed[0].out_start, 0.0, 1e-6)
        approx(fixed[-1].out_end - fixed[0].out_start,
               sum(c.duration for c in fixed), 0.02)
        is_true(any("锚定" in n for n in notes), f"没有记录锚定动作: {notes}")
        return f"{cls[0].out_start:.1f}s -> 0.0s, {len(fixed)} 段, {len(notes)} 条说明"

    c("时间轴锚定到 0", validator_anchors_timeline)

    def extract_json_always_dict():
        """LLM 回包解析必须**永远**返回 dict, 否则调用方按 dict 用会崩.

        真实缺陷: `_extract_json('[{"clips": []}]')` 返回 list, 调用方
        `data.get("clips")` 直接 AttributeError -> 整条流水线失败。
        """
        cases = {
            '[{"clips": []}]': {},                     # JSON 数组
            '{"clips": ["x"]}': {"clips": ["x"]},       # clips 元素不是 dict
            '{"clips": [{"segment":0}]}': {"clips": [{"segment": 0}]},
            'not json': {},
            '```json\n{"clips": []}\n```': {"clips": []},
            '': {},
        }
        for raw, want in cases.items():
            got = P._extract_json(raw)
            is_true(isinstance(got, dict), f"{raw[:24]!r} -> {type(got).__name__}")
            is_true(got == want, f"{raw[:24]!r} -> {got}, 期望 {want}")
        # LLM 明确"无可用片段"时不能崩, 且必须能识别为空
        empty = P._extract_json('{"clips": []}')
        is_true(isinstance(empty, dict) and not empty.get("clips"),
                f"空 clips 解析异常: {empty}")
        return f"{len(cases)} 组输入全部返回 dict"

    c("LLM 回包解析健壮性", extract_json_always_dict)

    def declared_duration_fits_source():
        """声明时长必须撑得住源素材 —— 否则渲染时末帧冻结.

        真实缺陷: `_top_up_tail` 曾按"剩余尾巴"声明 4.5~9.0s, 而单杀素材只有
        3.8s, 4 条补充镜头全部中招。修法是声明时长同时受源素材上限约束,
        并在补齐后再收敛一次。这里直接验**生产路径的最终 EDL**。
        """
        edl = P.plan_edit(a, cards, use_llm=False, max_clips=12)
        bad = []
        for cl in edl.clips:
            src_sec = ((cl.src_end_tick - cl.src_start_tick) / P.TICKRATE
                       / max(cl.speed, 1e-6))
            if src_sec + 0.05 < cl.duration:
                bad.append((cl.index, round(cl.duration, 2), round(src_sec, 2)))
        is_true(not bad, f"声明时长超过源素材 (会冻结末帧): {bad}")
        return f"{len(edl.clips)} 段全部撑得住声明时长"

    c("声明时长不超过源素材", declared_duration_fits_source)


# ==================================================================
# 5. 渲染 (坐标变换 + 画幅布局 + 实际出帧)
# ==================================================================
def test_render(c: Check) -> None:
    from PIL import Image

    from . import demo as D, mapview as mv, radar as R

    dem = D.load_demo(str(config.DEFAULT_DEMO), with_ticks=True)
    res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=6)
    cards = D.cards_from_dicts(res["highlights"])
    card = cards[0]
    holder: dict = {"dem": dem, "card": card, "map": res["map_name"]}

    def viewport_math():
        vp = mv.Viewport(-1000.0, 1000.0, -1000.0, 1000.0, size=1024)
        # 中心应映射到画面中心
        px, py = vp.to_px(0.0, 0.0)
        approx(px, 512.0, 1.5)
        approx(py, 512.0, 1.5)
        # Y 必须翻转: Y 越大, py 越小
        _, py_hi = vp.to_px(0.0, 900.0)
        _, py_lo = vp.to_px(0.0, -900.0)
        is_true(py_hi < py_lo, "Y 轴未翻转")
        # 左上角
        px0, py0 = vp.to_px(-1000.0, 1000.0)
        is_true(abs(px0) < 2 and abs(py0) < 2, f"左上角映射错误: {px0},{py0}")
        # 向量化与逐点必须一致
        ax, ay = vp.to_px_array(np.array([123.0]), np.array([-456.0]))
        bx, by = vp.to_px(123.0, -456.0)
        approx(float(ax[0]), bx, 1e-6)
        approx(float(ay[0]), by, 1e-6)
        return "中心/翻转/左上角/向量化一致"

    c("坐标变换正确性", viewport_math)

    def viewport_nan_safe():
        xs = np.array([np.nan, 10.0, 20.0, np.nan, 30.0])
        ys = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
        vp = mv.Viewport.from_positions(xs, ys)
        is_true(np.isfinite(vp.x_min) and np.isfinite(vp.x_max), "NaN 未过滤")
        empty = mv.Viewport.from_positions(np.array([]), np.array([]))
        is_true(np.isfinite(empty.x_min), "空输入未兜底")
        return "NaN/空输入均有兜底"

    c("视口对 NaN/空输入健壮", viewport_nan_safe)

    def viewport_span_clamped():
        far = np.array([0.0, 100000.0])   # 离群点
        vp = mv.Viewport.from_positions(far, far)
        is_true(vp.span_x <= 3400.0 + 1e-6, f"取景未被夹住: {vp.span_x}")
        return f"span={vp.span_x:.0f} (上限 3400)"

    c("取景范围被夹住", viewport_span_clamped)

    def layout_math():
        for name, (w, h) in config.ASPECT_PRESETS.items():
            L = R.Layout.from_config(name)
            is_true(L.width == w and L.height == h, f"{name} 画布尺寸错误")
            mx, my = L.map_origin
            is_true(mx >= 0 and my >= 0, f"{name} 地图原点为负")
            is_true(mx + L.map_size <= w, f"{name} 地图横向溢出")
            is_true(my + L.map_size <= h, f"{name} 地图纵向溢出")
            rect = L.panel_rect
            if rect:
                x0, y0, x1, y1 = rect
                is_true(x1 <= w and y1 <= h, f"{name} 面板溢出画布")
                # 面板不能压在地图上。两种排布方式不同:
                #   right  面板在地图右侧 -> 面板左边界应在地图右边界之后
                #   bottom 面板在地图正下方 -> 横向与地图对齐, 纵向在地图之下
                if L.panel == "right":
                    is_true(x0 >= mx + L.map_size - 2, f"{name} 面板压住地图")
                elif L.panel == "bottom":
                    is_true(y0 >= my + L.map_size - 2, f"{name} 面板压住地图")
                    is_true(x0 == mx, f"{name} 面板未与地图左对齐")
                    is_true(x1 == mx + L.map_size, f"{name} 面板未与地图右对齐")
        return "三种画幅地图/面板均不溢出且不重叠"

    c("画幅布局不溢出", layout_math)

    def tick_table():
        t = R.build_tick_table(dem)
        is_true(len(t.names) >= 10, f"玩家数过少: {len(t.names)}")
        for i in range(len(t.names)):
            arr = t.ticks[i]
            is_true(arr.size > 0, f"{t.names[i]} 无 tick 数据")
            is_true(bool(np.all(np.diff(arr) >= 0)), f"{t.names[i]} tick 未排序")
            is_true(bool(np.all(np.isfinite(t.xs[i]))), f"{t.names[i]} X 含 NaN")
            is_true(bool(np.all(np.isfinite(t.ys[i]))), f"{t.names[i]} Y 含 NaN")
        # searchsorted 行为
        i0 = t.index_at(0, int(t.ticks[0][0]) - 1000)
        is_true(i0 == -1, "过早 tick 应返回 -1")
        return f"{len(t.names)} 个玩家组, 无 NaN 且有序"

    c("逐 tick 表无 NaN 且有序", tick_table)

    def render_all_aspects():
        table = R.build_tick_table(dem)
        xs_all, ys_all = [], []
        for i in range(len(table.names)):
            sl = table.slice_at(i, card.start_tick, card.end_tick)
            if sl.stop > sl.start:
                xs_all.append(table.xs[i][sl])
                ys_all.append(table.ys[i][sl])
        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        places = mv.collect_places(dem, card.kills)
        sizes = {}
        for name, (w, h) in config.ASPECT_PRESETS.items():
            L = R.Layout.from_config(name)
            vp = mv.Viewport.from_positions(xs, ys, size=L.map_size)
            base = mv.render_base(vp, places=places, map_name=holder["map"])
            rend = R.RadarRenderer(dem, table, vp, base_image=base,
                                   layout=L, title=holder["map"])
            it = rend.render_clip(card, fps=30, places=places,
                                  map_name=holder["map"], score_text="9 - 6",
                                  progress=(1, 6))
            img = None
            for _ in range(30):
                img = next(it)
            is_true(isinstance(img, Image.Image), f"{name} 未产出图像")
            is_true(img.size == (w, h), f"{name} 尺寸错误: {img.size}")
            # 画面不能是全黑 (说明确实画了东西)
            arr = np.asarray(img.convert("L"))
            is_true(arr.std() > 1.0, f"{name} 画面几乎全黑")
            sizes[name] = f"{img.size[0]}x{img.size[1]}"
            holder.setdefault("frames", {})[name] = img
        return str(sizes)

    c("三种画幅实际出帧", render_all_aspects)

    def per_clip_framing():
        """逐段跟拍: 按素材参与者取景, 且比全场取景明显更紧.

        真实教训: 最初用**全场 10 个玩家**算取景, 到中心的 95 分位距离达
        1995~6484 units, 所需跨度 4000~13000 —— 40 条素材里有 39 条撞上跨度
        上限, 逐段跟拍完全退化成"永远最远"。必须只框住本段参与者。
        """
        table = R.build_tick_table(dem)
        L = R.Layout.from_config("wide")
        # 全片兜底视口 (旧行为)
        xs_all, ys_all = [], []
        for i in range(len(table.names)):
            sl = table.slice_at(i, 0, 10 ** 9)
            if sl.stop > sl.start:
                xs_all.append(table.xs[i][sl])
                ys_all.append(table.ys[i][sl])
        clamp = mv.Viewport.from_positions(np.concatenate(xs_all),
                                           np.concatenate(ys_all), size=L.map_size)

        spans = []
        for c in cards:
            rend = R.RadarRenderer(dem, table, clamp, layout=L)
            vp = rend.clip_viewport(c, zoom=1.2, clamp_to=clamp)
            spans.append(vp.span_x)
            is_true(vp.span_x > 0, f"{c.id} 取景跨度为 0")

            # 主角必须留在画面内 (名字标注画在圆点上方, 太靠边会被切)
            for i, nm in enumerate(table.names):
                if nm != c.player:
                    continue
                sl = table.slice_at(i, c.start_tick, c.end_tick)
                if sl.stop <= sl.start:
                    continue
                mx = float(np.median(table.xs[i][sl]))
                my = float(np.median(table.ys[i][sl]))
                px, py = vp.to_px(mx, my)
                margin = min(px, py, L.map_size - px, L.map_size - py)
                is_true(margin > 40, f"{c.id} 主角距画面边缘仅 {margin:.0f}px")
                break

        med = float(np.median(spans))
        is_true(med < clamp.span_x * 0.75,
                f"逐段取景中位 {med:.0f}u 未明显紧于全场 {clamp.span_x:.0f}u")
        # 逐段取景必须有差异 (否则等于没跟拍)
        is_true(len({round(s / 50) for s in spans}) >= 3,
                f"各段取景几乎相同, 跟拍未生效: {sorted(set(round(s) for s in spans))[:5]}")
        return (f"{len(spans)} 段: 中位 {med:.0f}u vs 全场 {clamp.span_x:.0f}u, "
                f"范围 {min(spans):.0f}~{max(spans):.0f}u")

    c("逐段跟拍取景有效", per_clip_framing)

    def zoom_affects_span():
        """激烈度越高镜头越近: zoom 增大必须让跨度变小 (到下限为止).

        注意必须挑一条"自然跨度高于下限"的素材 —— 否则它已经顶在下限上,
        zoom 再怎么加都不会变, 看起来像"zoom 没生效"。这一点在新 demo 上
        真的踩到了: 原先固定用 cards[0], 它在新 demo 上跨度恰好是下限 1150。
        """
        table = R.build_tick_table(dem)
        L = R.Layout.from_config("wide")
        rend = R.RadarRenderer(dem, table, mv.Viewport(-2000, 1000, -500, 2500,
                                                       size=L.map_size), layout=L)
        picked = None
        for cand in cards:
            base_vp = rend.clip_viewport(cand, zoom=1.0)
            if base_vp.span_x > 1150.0 * 1.05:      # 明显在下限之上
                picked = cand
                break
        is_true(picked is not None, "所有素材都顶在取景下限, 无法验证 zoom")

        seq = []
        prev = None
        for z in (1.0, 1.2, 1.45):
            vp = rend.clip_viewport(picked, zoom=z)
            seq.append(round(vp.span_x))
            if prev is not None:
                is_true(vp.span_x <= prev + 1e-6,
                        f"zoom={z} 时跨度反而变大: {prev} -> {vp.span_x}")
            prev = vp.span_x
        is_true(seq[0] > seq[-1], f"zoom 完全没起作用: {seq}")
        return f"{picked.id[:20]} zoom 1.0/1.2/1.45 -> span {seq}"

    c("zoom 随激烈度收紧镜头", zoom_affects_span)

    def fade_in_out_works():
        """段首尾必须真的变暗 (软化硬切), 中段不受影响."""
        import numpy as np
        from PIL import Image

        table = R.build_tick_table(dem)
        L = R.Layout.from_config("wide")
        b = R.RadarRenderer(dem, table, mv.Viewport(-2000, 1000, -500, 2500,
                                                    size=L.map_size), layout=L)
        vp = b.clip_viewport(card, zoom=1.2)
        st = R.RenderStyle(fade_in=0.18, fade_out=0.22)
        rend = R.RadarRenderer(dem, table, vp, layout=L, style=st)
        rend.rebase(vp, mv.collect_places(dem, card.kills), "de_dust2")

        bright = []
        mx, my = L.map_origin
        for img in rend.render_clip(card, fps=30, places={}, map_name="de_dust2"):
            arr = np.asarray(img.convert("L"))
            bright.append(float(arr[my:my + L.map_size, mx:mx + L.map_size].mean()))
        bright = np.asarray(bright)
        is_true(bright.size > 60, "渲染帧数过少, 无法验证淡入淡出")
        mid = float(bright[bright.size // 3: 2 * bright.size // 3].mean())
        is_true(mid > 1.0, "中段画面全黑, 无法判断")
        is_true(bright[0] < mid * 0.15, f"首帧未淡入: {bright[0]:.1f} vs 中段 {mid:.1f}")
        is_true(bright[-1] < mid * 0.15, f"末帧未淡出: {bright[-1]:.1f} vs 中段 {mid:.1f}")
        # 中段不能一直被压暗 (只有首尾该暗)
        is_true(bright[20] > mid * 0.85, f"第 20 帧仍偏暗: {bright[20]:.1f}")
        return f"首 {bright[0]:.1f} / 中段 {mid:.1f} / 末 {bright[-1]:.1f}"

    c("段首尾淡入淡出生效", fade_in_out_works)

    def shake_offset_decays():
        """击杀震屏: 位移有界且随时间衰减到 0."""
        table = R.build_tick_table(dem)
        L = R.Layout.from_config("wide")
        st = R.RenderStyle(shake_px=9.0, shake_sec=0.22)
        rend = R.RadarRenderer(dem, table, mv.Viewport(-2000, 1000, -500, 2500,
                                                       size=L.map_size),
                               layout=L, style=st)
        # 造一条击杀事件在 tick 1000
        from .demo import KillEvent
        rend._kill_events = [KillEvent(tick=1000, attacker="a", attacker_side="t",
                                       victim="b", victim_side="ct", weapon="ak47",
                                       headshot=True)]
        # 击杀瞬间应有位移
        off0 = rend._shake_offset(1000)
        is_true(off0 != (0, 0), "击杀瞬间没有位移")
        # 幅度有界
        is_true(max(abs(off0[0]), abs(off0[1])) <= st.shake_px,
                f"位移超过设定幅度: {off0}")
        # 衰减到 0
        is_true(rend._shake_offset(1000 + int(0.3 * 64)) == (0, 0), "震屏未在时限内结束")
        is_true(rend._shake_offset(500) == (0, 0), "击杀前不应震屏")
        seq = [max(abs(v) for v in rend._shake_offset(1000 + int(k / 64 * 64)))
               for k in range(0, 16)]
        is_true(seq[0] >= seq[-1], f"震屏幅度未衰减: {seq[:6]}")
        return f"幅度 {off0} -> 衰减序列 {seq[:6]}"

    c("击杀震屏有界且衰减", shake_offset_decays)

    def vignette_baked_into_base():
        """暗角必须烘焙进底图 (每帧合成一次太贵), 且方向正确.

        注意: 不能直接断言"角落比中心暗" —— 角落可能正好有亮网格线交叉,
        本来就更亮。正确做法是**对照**: 同一视口分别开/关暗角, 比较边缘
        区域的下降幅度。
        """
        import numpy as np

        table = R.build_tick_table(dem)
        L = R.Layout.from_config("wide")
        vp = mv.Viewport(-2000, 1000, -500, 2500, size=L.map_size)
        places = mv.collect_places(dem, card.kills)

        def baked(with_vig: bool):
            rend = R.RadarRenderer(dem, table, vp, layout=L,
                                   style=R.RenderStyle(vignette=with_vig))
            rend.rebase(vp, places, "de_dust2")
            return rend, rend._base

        rend_on, base_on = baked(True)
        _, base_off = baked(False)

        is_true(base_on is not None and base_on.mode == "RGBA",
                "底图未预转 RGBA (每帧 convert 约 8ms 的浪费)")

        def edge_mean(img):
            a = np.asarray(img.convert("L")).astype("float64")
            h, w = a.shape
            m = max(int(min(h, w) * 0.06), 4)
            edge = np.concatenate([a[:m].ravel(), a[-m:].ravel(),
                                   a[:, :m].ravel(), a[:, -m:].ravel()])
            return float(edge.mean())

        e_on, e_off = edge_mean(base_on), edge_mean(base_off)
        h, w = L.map_size, L.map_size
        is_true(e_off > 1.0, "未开暗角时边缘也全黑, 无法比较")
        # 阈值说明: 底图本身很暗 (背景 #0d1117, 亮度约 17), 遮罩边缘约 0.94,
        # 所以边缘亮度降幅只有 6% 左右 (1 个灰阶)。断言按实际掩码行为校准,
        # 同时另测掩码本身的强度, 避免"几乎没生效"被放过。
        is_true(e_on < e_off * 0.97,
                f"暗角未生效或方向反了: 开 {e_on:.2f} vs 关 {e_off:.2f}")
        vig = np.asarray(rend_on._get_vignette(L.map_size)).astype("float64") / 255.0
        is_true(vig[h // 2, w // 2] > 0.99, f"遮罩中心应保留全亮: {vig[h // 2, w // 2]:.3f}")
        is_true(vig.min() < 0.85, f"遮罩边缘压暗不足: 最小 {vig.min():.3f}")
        # 中心不该被压暗
        a_on = np.asarray(base_on.convert("L")).astype("float64")
        a_off = np.asarray(base_off.convert("L")).astype("float64")
        h, w = a_on.shape
        c = (slice(int(h * 0.4), int(h * 0.6)), slice(int(w * 0.4), int(w * 0.6)))
        is_true(abs(a_on[c].mean() - a_off[c].mean()) < 3.0,
                f"中心也被压暗了 (暗角方向反了): {a_on[c].mean():.1f} vs {a_off[c].mean():.1f}")

        # 遮罩缓存
        is_true(rend_on._get_vignette(L.map_size) is rend_on._get_vignette(L.map_size),
                "暗角遮罩未缓存")
        return (f"边缘 {e_off:.2f} -> {e_on:.2f} (降 {(1 - e_on / e_off) * 100:.0f}%), "
                f"遮罩中心 {vig[h // 2, w // 2]:.2f} / 最暗 {vig.min():.2f}")

    c("暗角烘焙进底图且方向正确", vignette_baked_into_base)

    def clutch_definition():
        """残局判定必须追踪"谁是最后一人", 并且需要可靠的幸存者数据.

        这段逻辑先后踩过三个坑, 所以用例要覆盖到位:
          1. 旧实现只看"哪一方先掉到 1 人", 不追踪幸存者是谁 -> 队友送完时
             已经阵亡的玩家也被当成残局主角 (判定过宽)。
          2. 幸存者不能用"没出现在死亡事件里"推断 —— 既可能是活着也可能是
             数据缺失。实测回合 9 靠事件推断会把实际存活的队友算成阵亡,
             把 2v0 的收尾误判成 1vX。必须用逐 tick 血量。
          3. 残局起点是"**主角本人**成为全队最后一人"的时刻, 而不是"最后一个
             队友阵亡"的时刻。名单残缺时后者会晚得多。
        """
        from .demo import KillEvent, find_clutch

        def K(t, a, asd, v, vsd):
            return KillEvent(tick=t, attacker=a, attacker_side=asd, victim=v,
                             victim_side=vsd, weapon="ak47", headshot=False)

        roster = {**{n: "t" for n in "ABCDE"}, **{f"P{i}": "ct" for i in range(1, 6)}}

        # --- 真残局: A 独自面对 5 人并全歼, 回合末只剩他 ---
        clutch_kills = [
            K(100, "P1", "ct", "B", "t"), K(200, "P1", "ct", "C", "t"),
            K(300, "P2", "ct", "D", "t"), K(400, "P3", "ct", "E", "t"),
            K(500, "A", "t", "P1", "ct"), K(600, "A", "t", "P2", "ct"),
            K(700, "A", "t", "P3", "ct"), K(800, "A", "t", "P4", "ct"),
            K(900, "A", "t", "P5", "ct"),
        ]
        c = find_clutch(clutch_kills, "t", roster, {"A"})
        is_true(c is not None, "教科书式 1v5 残局未被识别")
        is_true(c.player == "A" and c.enemies == 5,
                f"残局主角/敌人数错误: {c}")
        is_true(c.start_tick == 400, f"残局起点错误: {c.start_tick}")

        # --- 赢方收尾剩 2 人 -> 不是残局 (真实回合 9 就是这种) ---
        two_alive = [
            K(100, "P1", "ct", "B", "t"), K(200, "P1", "ct", "C", "t"),
            K(300, "P2", "ct", "D", "t"), K(400, "P3", "ct", "E", "t"),
            K(500, "A", "t", "P1", "ct"), K(600, "A", "t", "P2", "ct"),
            K(700, "A", "t", "P3", "ct"),
        ]
        is_true(find_clutch(two_alive, "t", roster, {"A", "B"}) is None,
                "赢方剩 2 人却判成了残局")

        # --- 主角成为最后一人时敌人只有 1 个 -> 1v1 收尾, 不是残局 ---
        one_v_one = [
            K(100, "B", "t", "P2", "ct"), K(120, "C", "t", "P3", "ct"),
            K(140, "D", "t", "P4", "ct"), K(160, "E", "t", "P5", "ct"),
            K(200, "P1", "ct", "B", "t"), K(220, "P1", "ct", "C", "t"),
            K(240, "P1", "ct", "D", "t"), K(260, "P1", "ct", "E", "t"),
            K(300, "A", "t", "P1", "ct"),
        ]
        is_true(find_clutch(one_v_one, "t", roster, {"A"}) is None,
                "1v1 收尾被误判成残局")

        # --- 数据不足时不得判定 (宁可不标也不要标错) ---
        is_true(find_clutch(clutch_kills, "t", None, {"A"}) is None, "无名单却判定了")
        is_true(find_clutch(clutch_kills, "t", roster, None) is None, "无幸存者信息却判定了")
        is_true(find_clutch([], "t", roster, {"A"}) is None, "空击杀列表却判定了")
        is_true(find_clutch(clutch_kills, None, roster, {"A"}) is None, "无胜方却判定了")
        return "真残局识别 + 3 类误判拦截 + 4 项数据缺失保护"

    c("残局判定定义正确", clutch_definition)

    def roster_from_ticks():
        """名单与幸存者必须来自逐 tick 表, 且按回合区分 (玩家会换边).

        只验证**结构性不变量**, 不写死某个回合的具体人数 —— 那是某一局 demo 的
        专有事实, 换一局就会失败 (原来的写法断言"回合 9 的 ct 幸存者应为 2 人",
        换到新 demo 立刻误报)。
        """
        rosters, survivors = D.build_roster(dem, max_rounds=30)
        is_true(len(rosters) >= 20, f"名单回合数过少: {len(rosters)}")
        r1 = rosters[min(rosters)]
        is_true(len(r1) == 10, f"回合名单人数异常: {len(r1)}")
        n_t = sum(1 for v in r1.values() if v == "t")
        n_ct = sum(1 for v in r1.values() if v == "ct")
        is_true(n_t == 5 and n_ct == 5, f"阵营分配异常: t={n_t} ct={n_ct}")

        # 幸存者必须非空、且都出现在该回合名单里
        bad_subset = [rn for rn, s in survivors.items()
                      if rn in rosters and not s <= set(rosters[rn])]
        is_true(not bad_subset, f"有回合的幸存者不在名单里: {bad_subset[:3]}")
        empty = [rn for rn, s in survivors.items() if not s]
        is_true(not empty, f"有回合幸存者为空: {empty[:3]}")

        # 每回合幸存者总数不应超过 10
        too_many = [(rn, len(s)) for rn, s in survivors.items() if len(s) > 10]
        is_true(not too_many, f"幸存者超过 10 人: {too_many[:3]}")

        # 换边必须被正确处理: 同一名字在不同回合可能属于不同阵营
        flips = set()
        first = rosters[min(rosters)]
        for rn, r in rosters.items():
            for n, sd in r.items():
                if n in first and first[n] != sd:
                    flips.add(n)
        # rounds 表与 ticks 表是两套 round_num, 必须交叉校验 —— 一旦错位,
        # rosters.get(rn) 全为 None, 所有残局会静默消失
        bounds = D._round_bounds(dem)
        missing = [rn for rn in bounds if rn not in rosters and rn not in survivors]
        is_true(not missing,
                f"回合号在 rounds 表与 ticks 表之间对不上: {missing[:5]}")
        n_surv = sum(len(s) for s in survivors.values())
        return (f"{len(rosters)} 回合名单 / {len(survivors)} 回合幸存者, "
                f"首回合 5v5, 换边玩家 {len(flips)} 人, 幸存者实例 {n_surv} 个, "
                f"回合号交叉校验通过 ({len(bounds)} 回合)")

    c("名单与幸存者取自逐 tick 表", roster_from_ticks)

    def clutch_not_overmarked():
        """真实 demo 上不得把普通回合标成残局."""
        rosters, survivors = D.build_roster(dem, max_rounds=30)
        bounds = D._round_bounds(dem)
        kills = D.extract_kill_events(dem)
        by_round: dict[int, list] = {}
        for e in kills:
            rn = D._tick_to_round(e.tick, bounds)
            if rn is not None:
                by_round.setdefault(rn, []).append(e)
        found = []
        for rn, rk in by_round.items():
            cl = D.find_clutch(rk, bounds[rn][3], rosters.get(rn), survivors.get(rn))
            if cl:
                # 每个残局都必须满足: 主角是赢方唯一幸存者
                win_alive = {n for n in (survivors.get(rn) or set())
                             if (rosters.get(rn) or {}).get(n) == bounds[rn][3]}
                is_true(win_alive == {cl.player},
                        f"回合 {rn} 残局主角 {cl.player} 不是唯一幸存者 {win_alive}")
                is_true(cl.enemies >= 2, f"回合 {rn} 残局敌人不足 2")
                found.append(rn)
        return f"{len(by_round)} 回合中判出 {len(found)} 个残局 {found}"

    c("真实 demo 残局不过度标记", clutch_not_overmarked)

    def render_deterministic():
        """同一输入两次渲染必须逐像素一致 (可复现性)."""
        table = R.build_tick_table(dem)
        L = R.Layout.from_config("square")
        vp = mv.Viewport(-2500.0, 1500.0, -1200.0, 3000.0, size=L.map_size)
        base = mv.render_base(vp, places={}, map_name="de_dust2")
        sigs = []
        for _ in range(2):
            rend = R.RadarRenderer(dem, table, vp, base_image=base, layout=L)
            it = rend.render_clip(card, fps=30, places={}, map_name="de_dust2")
            for _ in range(20):
                img = next(it)
            sigs.append(np.asarray(img.convert("L")).sum())
        is_true(sigs[0] == sigs[1], f"两次渲染结果不同: {sigs}")
        return f"像素校验和一致 ({sigs[0]})"

    c("渲染可复现", render_deterministic)

    def frame_count_matches():
        L = R.Layout.from_config("square")
        vp = mv.Viewport(-2500.0, 1500.0, -1200.0, 3000.0, size=L.map_size)
        base = mv.render_base(vp, places={}, map_name="de_dust2")
        rend = R.RadarRenderer(dem, table=R.build_tick_table(dem), viewport=vp,
                               base_image=base, layout=L)
        for speed in (0.5, 1.0, 2.0):
            n = sum(1 for _ in rend.render_clip(
                card, fps=30, speed=speed, places={}, map_name="de_dust2"))
            # 渲染器给的帧数应约等于 时长/帧长/speed
            expect = card.duration_ticks / (64 / 30) / speed
            rel = abs(n - expect) / max(expect, 1)
            is_true(rel < 0.05, f"speed={speed} 帧数 {n} 偏离期望 {expect:.0f}")
        return "0.5x/1x/2x 帧数均符合预期"

    c("倍速与帧数一致", frame_count_matches)


# ==================================================================
# 6. 合成 (ffmpeg)
# ==================================================================
def test_compose(c: Check) -> None:
    from PIL import Image

    from . import compose

    # 测试用视频路径集中管理 —— 后面的用例依赖前面产出的文件,
    # 之前把名字散落在各处, 一旦一个用例写错文件名就连续失败三项。
    VID_A = config.WORK_DIR / "selftest_a.mp4"
    VID_B = config.WORK_DIR / "selftest_b.mp4"

    def encode_square():
        # 造 30 帧带移动亮块的图案。
        # 注意像素坐标必须取模 —— 最初的写法 (x + i, y) 在 x=300, i=29 时
        # 会越界到 329, 直接 IndexError。
        size = 320
        frames = []
        for i in range(30):
            img = Image.new("RGB", (size, size), (10, 10, 10))
            for x in range(0, size, 20):
                for y in range(0, size, 20):
                    img.putpixel(((x + i) % size, y % size), (255, 80 + i * 4, 80))
            frames.append(img)
        p = compose.render_clip_video(frames, VID_A, fps=30, duration=1.0,
                                      size=(size, size))
        dur = compose.probe_duration(p)
        approx(dur, 1.0, 0.15)
        is_true(p.stat().st_size > 500, "输出文件过小")
        return f"{p.name} {dur:.2f}s, {p.stat().st_size / 1024:.0f} KB"

    c("编码方形片段", encode_square)

    def encode_wide():
        frames = [Image.new("RGB", (640, 360), (20 + i, 30, 40)) for i in range(30)]
        p = compose.render_clip_video(frames, VID_B, fps=30, duration=1.0,
                                      size=(640, 360))
        dur = compose.probe_duration(p)
        approx(dur, 1.0, 0.15)
        return f"{p.name} {dur:.2f}s"

    c("编码非方形片段", encode_wide)

    def finalize_pads():
        """补长机制: 慢放段帧数偏少时必须补到精确时长."""
        frames = [Image.new("RGB", (320, 320), (50, 50, 50)) for _ in range(10)]
        raw = compose.render_clip_video(frames, config.WORK_DIR / "selftest_raw.mp4",
                                        fps=30, duration=0.33, size=(320, 320))
        d0 = compose.probe_duration(raw)
        out = compose.finalize_clip(raw, config.WORK_DIR / "selftest_fin.mp4",
                                    target_duration=2.0, size=(320, 320))
        d1 = compose.probe_duration(out)
        is_true(abs(d1 - 2.0) < 0.15, f"补长后 {d1:.2f}s, 期望 2.0s")
        is_true(d1 > d0 + 1.0, f"没有实际补长: {d0:.2f} -> {d1:.2f}")
        is_true(not raw.exists(), "临时文件未清理")
        return f"{d0:.2f}s -> {d1:.2f}s"

    c("片段补长到精确时长", finalize_pads)

    def concat_and_mux():
        from . import compose as C, music as M
        music = config.WORK_DIR / "selftest_track.wav"
        if not music.is_file():
            M.make_test_track(music)
        items = []
        for p, dur in ((VID_A, 1.0), (VID_B, 1.0)):
            if p.is_file():          # 依赖前两项的产物, 缺了就直接跳过
                items.append(C.ConcatItem(path=p, duration=dur))
        is_true(len(items) == 2,
                f"缺少待拼接片段 (A={VID_A.is_file()}, B={VID_B.is_file()})")
        out = C.concat_clips(items, config.WORK_DIR / "selftest_final.mp4",
                             music_path=music, total_duration=2.0, fade_out=0.5)
        dur = C.probe_duration(out)
        approx(dur, 2.0, 0.3)
        # 必须有音轨
        ffprobe = config.find_ffprobe()
        import subprocess
        r = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=codec_name", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(out)],
                           capture_output=True, text=True)
        is_true("aac" in r.stdout.lower(), f"没有音轨: {r.stdout!r}")
        return f"{dur:.2f}s 含 AAC 音轨"

    c("拼接 + 铺音乐", concat_and_mux)

    def concat_accepts_relative_paths():
        """拼接必须接受相对路径的片段.

        真实缺陷 (端到端跑真歌+新 demo 时才暴露): 清单文件写在 `work/` 下,
        而 ffmpeg 的 concat demuxer 以**清单所在目录**为相对路径基准 ——
        片段路径若写成 `out/xxx/clips/c1.mp4`, 会被解析成
        `work/out/xxx/clips/c1.mp4`, 而 ffmpeg 只报
        "Impossible to open ... Error opening input file <清单>" ——
        错误信息指向清单本身, 完全看不出真正原因是路径基准。
        """
        a, b = config.WORK_DIR / "selftest_a.mp4", config.WORK_DIR / "selftest_b.mp4"
        if not (a.is_file() and b.is_file()):
            raise AssertionError("缺少待拼接片段 (依赖上一项产物)")
        # 故意用相对路径 (相对进程 CWD), 这正是触发缺陷的用法
        import os
        rel = [os.path.relpath(p, Path.cwd()) for p in (a, b)]
        is_true(not Path(rel[0]).is_absolute(), "构造的不是相对路径, 用例无意义")
        out = config.WORK_DIR / "selftest_rel.mp4"
        compose.concat_clips(
            [compose.ConcatItem(path=Path(p), duration=1.0) for p in rel],
            out, total_duration=2.0,
        )
        approx(compose.probe_duration(out), 2.0, 0.3)
        # 片段缺失时必须是"片段不存在", 不能是 ffmpeg 指向清单的报错
        try:
            compose.concat_clips(
                [compose.ConcatItem(path=Path("no/such/clip.mp4"), duration=1.0)],
                out, total_duration=1.0,
            )
            raise AssertionError("片段缺失却没有报错")
        except FileNotFoundError as e:
            is_true("片段" in str(e), f"错误信息没有指向片段: {e}")
        out.unlink(missing_ok=True)
        return f"相对路径 {rel[0]} 拼接成功; 缺片段时报错指向片段"

    c("拼接接受相对路径且报错指向片段", concat_accepts_relative_paths)

    def thumbnail():
        p = config.WORK_DIR / "selftest_final.mp4"
        if not p.is_file():
            raise AssertionError("缺少测试视频")
        t = compose.make_thumbnail(p, config.WORK_DIR / "selftest_thumb.png", at=0.5)
        is_true(t.is_file() and t.stat().st_size > 500, "缩略图为空")
        return f"{t.stat().st_size / 1024:.0f} KB"

    c("抽帧生成缩略图", thumbnail)

    def missing_ffmpeg_message():
        """找不到 ffmpeg 时必须给出明确错误, 而不是静默失败."""
        orig = config.FFMPEG
        try:
            config.FFMPEG = None
            try:
                config.require_ffmpeg()
                raise AssertionError("未抛错")
            except RuntimeError as e:
                is_true("ffmpeg" in str(e), f"错误信息不清晰: {e}")
            return "缺失时给出明确提示"
        finally:
            config.FFMPEG = orig

    c("ffmpeg 缺失时明确报错", missing_ffmpeg_message)


# ==================================================================
# 7. 端到端
# ==================================================================
def test_e2e(c: Check, music: Path, aspects: list[str], use_llm: bool) -> None:
    from . import pipeline

    holder: dict = {}
    for aspect in aspects:
        tag = f"{aspect}-{'llm' if use_llm else 'nollm'}"

        def run_one(aspect=aspect, tag=tag):
            t0 = time.time()
            out = config.OUT_DIR / f"selftest_{tag}"
            if out.exists():
                import shutil
                shutil.rmtree(out, ignore_errors=True)
            state = pipeline.run(
                music, None,
                out_dir=out,
                aspect=aspect,
                max_clips=12,
                max_cards=24,
                use_llm=use_llm,
                verbose=False,
            )
            holder[tag] = (state, out, time.time() - t0)
            vp = Path(state.get("video_path", ""))
            is_true(vp.is_file(), f"成片不存在: {vp}")
            return f"{vp.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - t0:.0f}s"

        c(f"端到端出片 ({tag})", run_one)

    def verify_outputs():
        msgs = []
        import subprocess

        from . import compose
        ffprobe = config.find_ffprobe()
        for tag, (state, out, _) in holder.items():
            vp = Path(state["video_path"])
            edl = state["edl"]
            # 1. 视频规格
            r = subprocess.run(
                [str(ffprobe), "-v", "error", "-show_entries",
                 "stream=codec_type,width,height,duration", "-of", "json", str(vp)],
                capture_output=True, text=True)
            info = json.loads(r.stdout)
            streams = {s["codec_type"]: s for s in info["streams"]}
            is_true("video" in streams, f"{tag} 无视频轨")
            is_true("audio" in streams, f"{tag} 无音频轨")
            aspect = tag.split("-")[0]
            w, h = config.aspect_size(aspect)
            is_true(streams["video"]["width"] == w and streams["video"]["height"] == h,
                    f"{tag} 分辨率错误: {streams['video']['width']}x{streams['video']['height']}")
            vd = float(streams["video"]["duration"])
            ad = float(streams["audio"]["duration"])
            is_true(abs(vd - ad) < 0.5, f"{tag} 音视频时长差 {abs(vd - ad):.2f}s")
            is_true(abs(vd - edl.total_duration) < 0.5,
                    f"{tag} 成片 {vd:.2f}s 与 EDL {edl.total_duration:.2f}s 不符")
            # 2. 中间产物齐全
            for f in ("edl.json", "music_analysis.json", "demo_analysis.json"):
                is_true((out / f).is_file(), f"{tag} 缺产物 {f}")
            # 3. EDL 自洽
            prev = None
            for cl in edl.clips:
                if prev is not None:
                    is_true(abs(cl.out_start - prev) < 0.05, f"{tag} 时间轴有缝隙")
                prev = cl.out_end
            # 4. 画面不是全黑
            thumb = compose.make_thumbnail(vp, out / "_check.png", at=min(2.0, vd / 2))
            from PIL import Image
            arr = np.asarray(Image.open(thumb).convert("L"))
            is_true(arr.std() > 1.0, f"{tag} 画面几乎全黑")
            thumb.unlink(missing_ok=True)
            msgs.append(f"{tag}: {vd:.1f}s {w}x{h} 差{abs(vd - ad):.2f}s")
        return "; ".join(msgs)

    c("端到端产物完整性", verify_outputs)


# ==================================================================
# 7. 偏好存储 (sqlite)
# ==================================================================
def test_store(c: Check) -> None:
    """偏好存储必须可靠: 会拒绝非法值, 读坏库不崩, 优先级正确."""
    import shutil

    from . import pipeline as P
    from . import store as S

    # 测试库建在**工作区内**: 沙箱只允许工作区内写入, 系统临时目录会
    # "unable to open database file"。
    tmpdir = config.WORK_DIR / "_storetest"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    db = tmpdir / "test.db"

    def creates_schema():
        prefs = S.load_prefs(db)
        is_true(db.is_file(), "首次读取未建库")
        is_true(set(prefs) == set(S.PREFS), "默认偏好键不全")
        # 表必须真的建出来 (连表都没有说明 init_db 没跑)
        with S.connect(db) as conn:
            names = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("preferences", "runs", "clips"):
            is_true(t in names, f"表缺失: {t}")
        return f"{len(prefs)} 项偏好, 表 {sorted(names & {'preferences','runs','clips'})}"

    c("偏好库自动建表", creates_schema)

    def type_coercion():
        up = S.set_prefs({
            "aspect": "tall", "max_clips": "30", "use_llm": "false",
            "fade_in": "0.5", "trail_blur": 2, "vignette": "0",
        }, db)
        is_true(up["aspect"] == "tall", "str 偏好未写入")
        is_true(up["max_clips"] == 30 and isinstance(up["max_clips"], int),
                f"字符串未转 int: {up['max_clips']!r}")
        is_true(up["use_llm"] is False, f"字符串未转 bool: {up['use_llm']!r}")
        is_true(up["fade_in"] == 0.5, f"字符串未转 float: {up['fade_in']!r}")
        is_true(up["trail_blur"] == 2.0, f"int 未转 float: {up['trail_blur']!r}")
        is_true(up["vignette"] is False, f"'0' 未转 False: {up['vignette']!r}")
        # 重新打开仍在 (确认真的落盘而不是内存)
        again = S.load_prefs(db)
        is_true(again["aspect"] == "tall" and again["max_clips"] == 30,
                "重新打开后偏好丢失")
        is_true(again["fps"] == 30, "未修改的项被破坏")
        return "str→int/bool/float 均正确, 重开仍在"

    c("偏好类型转换与落盘", type_coercion)

    def rejects_invalid():
        """非法值必须被拒, 且不得污染已存数据."""
        before = S.load_prefs(db)["aspect"]
        for bad, err in (({"aspect": "cinema"}, ValueError),
                         ({"no_such_key": 1}, KeyError),
                         ({"fps": "abc"}, ValueError)):
            try:
                S.set_prefs(bad, db)
                raise AssertionError(f"非法输入未被拒绝: {bad}")
            except err:
                pass
        is_true(S.load_prefs(db)["aspect"] == before, "拒绝后库内值被改动")
        return "非法画幅/未知键/非数字 均被拒且不污染"

    c("偏好拒绝非法值", rejects_invalid)

    def precedence():
        """显式传参 > 已存偏好 > 内置默认."""
        S.set_prefs({"aspect": "tall", "max_clips": 99}, db)
        orig = S.load_prefs
        S.load_prefs = lambda path=None: orig(db)      # 让 pipeline 用测试库
        try:
            r1 = P.resolve_params({}, use_prefs=True)
            is_true(r1["aspect"] == "tall" and r1["max_clips"] == 99,
                    "未显式传参时未采用偏好")
            r2 = P.resolve_params({"aspect": "wide", "max_clips": None},
                                  use_prefs=True)
            is_true(r2["aspect"] == "wide", "显式传参未覆盖偏好")
            is_true(r2["max_clips"] == 99, "显式 None 不应覆盖偏好")
            r3 = P.resolve_params({}, use_prefs=False)
            is_true(r3["aspect"] == config.DEFAULT_ASPECT and r3["max_clips"] == 22,
                    "--no-prefs 未回落到默认")
        finally:
            S.load_prefs = orig
        return "偏好生效 / 显式覆盖 / None 不覆盖 / --no-prefs 回落 全部正确"

    c("参数优先级正确", precedence)

    def effect_bridge():
        """画面偏好必须能转成渲染器吃的 effects 字段."""
        eff = S.effect_overrides({"fade_in": 0.3, "shake_px": 0.0,
                                  "vignette": False, "trail_blur": 1.5})
        need = {"fade_in", "fade_out", "shake_px", "trail_blur",
                "vignette", "show_places", "show_hud"}
        is_true(need <= set(eff), f"缺少字段: {need - set(eff)}")
        is_true(eff["fade_in"] == 0.3 and eff["shake_px"] == 0.0
                and eff["vignette"] is False, "字段值传递错误")
        # 转出来的 effects 必须能被 RenderStyle 接受
        from . import radar as R
        st = R.RenderStyle.from_effects(eff)
        is_true(st.fade_in == 0.3 and st.shake_px == 0.0 and st.trail_blur == 1.5,
                "RenderStyle 未正确接收偏好")
        return "偏好 -> effects -> RenderStyle 链路通"

    c("画面偏好转 effects", effect_bridge)

    def run_history():
        """运行记录: 登记/回填/逐段明细/统计."""
        # list_runs 会列出**未完成**的运行, stats 只数已完成的 —— 两者语义不同,
        # 断言必须分开写 (最初把 runs_before 取在 finish 之前, 导致"未累加"误报)。
        runs_before = len(S.list_runs(limit=99, path=db))
        rid = S.start_run("d.dem", "m.mp3", "out/x", {"aspect": "wide"}, db)
        is_true(isinstance(rid, int) and rid > 0, "未返回 run_id")
        is_true(len(S.list_runs(limit=99, path=db)) == runs_before + 1,
                "登记后 list_runs 未增加")

        done_before = S.stats(db)["runs"]
        S.finish_run(rid, map_name="de_dust2", aspect="wide", fps=30,
                     use_llm=False, planner="fallback", clips=3,
                     duration_sec=72.0, video_path="out/x/final.mp4",
                     video_bytes=1234, elapsed_sec=9.9, path=db)
        is_true(S.stats(db)["runs"] == done_before + 1,
                "完成后 stats 未增加")

        n = S.record_clips(rid, [
            {"music_segment": 0, "out_start": 0.0, "out_end": 5.0,
             "highlight_id": "h1", "player": "A", "round_num": 1,
             "score": 50.0, "tags": ["multi"], "speed": 1.0},
            {"music_segment": 1, "out_start": 5.0, "out_end": 9.0,
             "highlight_id": "h2", "player": "B", "round_num": 2,
             "score": 40.0, "tags": [], "speed": 1.25},
        ], db)
        is_true(n == 2, f"逐段明细写入数不对: {n}")
        last = S.last_run(db)
        is_true(last is not None and last["id"] == rid, "最近一次运行取错")
        is_true(last["video_path"].endswith("final.mp4"), "产物路径丢失")
        st = S.stats(db)
        is_true(st["clips"] >= 2, f"统计里剪辑数不对: {st['clips']}")
        # 未完成的运行不该出现在 last_run 里
        rid2 = S.start_run("e.dem", "n.mp3", "out/y", {}, db)
        is_true(S.last_run(db)["id"] == rid, "未完成的运行污染了 last_run")
        S.finish_run(rid2, clips=1, video_path="out/y/f.mp4", path=db)
        is_true(S.last_run(db)["id"] == rid2, "完成后 last_run 未更新")
        return (f"run_id={rid}, 明细 2 段, 登记/完成各自累加正确, "
                f"last_run 只取已完成")

    c("运行记录可登记与统计", run_history)

    def clear_works():
        S.set_prefs({"aspect": "tall", "fps": 60}, db)
        n = S.clear_prefs(["aspect"], db)
        is_true(n == 1 and S.load_prefs(db)["aspect"] == config.DEFAULT_ASPECT,
                "按键清除失败")
        S.clear_prefs(None, db)
        is_true(S.load_prefs(db) == S.defaults(), "清空全部后未回到默认")
        return "按键清除 + 清空全部均正确"

    c("偏好可清除", clear_works)

    def corrupt_db_safe():
        """库损坏时读偏好不得抛错 —— 存储问题不该让出片失败."""
        bad = tmpdir / "bad.db"
        bad.write_bytes(b"definitely not a sqlite database")
        got = S.load_prefs(bad)
        is_true(got == S.defaults(), "损坏库未回落到默认值")
        try:
            S.get_pref("nope", db)
            raise AssertionError("未知键未抛 KeyError")
        except KeyError:
            pass
        return "损坏库回落默认, 未知键抛 KeyError"

    c("存储在损坏时不影响出片", corrupt_db_safe)

    shutil.rmtree(tmpdir, ignore_errors=True)


# ==================================================================
# 8. Web 界面 (进度事件 + 路径安全 + 真实 HTTP 服务)
# ==================================================================
def test_web(c: Check, *, full: bool = True) -> None:
    import socket
    import subprocess as sp

    from . import media, progress, webapp

    def stage_math():
        """百分比必须单调、且六个阶段权重加起来正好 100%."""
        is_true(abs(progress.stage_percent("compose", 1.0) - 100.0) < 0.05,
                f"末尾阶段不是 100%: {progress.stage_percent('compose', 1.0)}")
        prev = -1.0
        evs: list[progress.Event] = []
        bus = progress.Bus(evs.append)
        for stage in progress.STAGE_ORDER:
            bus.stage_start(stage, "")
            bus.progress(stage, 0.5, "")
            bus.stage_done(stage, "")
        # 阶段顺序 -> 百分比必须不减
        for e in evs:
            if e.affects_percent:
                is_true(e.percent >= prev - 1e-9,
                        f"百分比倒退: {prev} -> {e.percent} ({e.stage})")
                prev = e.percent
        return f"{len(evs)} 个事件, 单调递增到 {prev:.1f}%"

    c("进度事件的百分比单调", stage_math)

    def interleaved_stages_stay_monotonic():
        """并发阶段交错时百分比也不得倒退.

        真实缺陷: analyze_music 与 analyze_demo 在图里是并行节点, 事件交错
        到达。各自按自己权重算出的百分比实测会出现 8.0 → 0.0 → 30.0 ——
        进度条看起来在倒退。Bus 现在统一取历史最高值。
        """
        evs: list[progress.Event] = []
        bus = progress.Bus(evs.append)
        bus.stage_start("analyze_music", "")
        bus.stage_done("analyze_music", "")      # 8%
        bus.stage_start("analyze_demo", "")      # 本身是 8%, 不能倒退到 0
        bus.stage_done("analyze_demo", "")       # 30%
        vals = [e.percent for e in evs if e.affects_percent]
        is_true(vals == sorted(vals), f"交错阶段导致倒退: {vals}")
        return f"{vals}"

    c("并发阶段交错时百分比不倒退", interleaved_stages_stay_monotonic)

    def log_events_do_not_move_bar():
        """log 事件不得推进度条 (它可能从任何阶段提前打出来)."""
        evs: list[progress.Event] = []
        bus = progress.Bus(evs.append)
        bus.stage_start("analyze_music", "")
        bus.log("plan 阶段的告警提前打出来了", stage="plan")
        moved = [e for e in evs if e.affects_percent]
        is_true(len(moved) == 1, f"log 被当成进度事件: {[e.kind for e in moved]}")
        return "只有 stage_* / progress / done 推进度"

    c("日志事件不推进度条", log_events_do_not_move_bar)

    def callback_failure_is_harmless():
        """进度回调抛异常不得影响出片."""
        def boom(ev):
            raise RuntimeError("UI 坏了")
        bus = progress.Bus(boom)
        for stage in progress.STAGE_ORDER:
            bus.stage_start(stage, "")
            bus.stage_done(stage, "")
        bus.done("ok")
        return f"回调每次抛错, {bus.seq} 条事件全部被丢弃且未抛出"

    c("进度回调异常不影响出片", callback_failure_is_harmless)

    def media_path_safety():
        """产物路径必须锁在工作区内; 素材路径允许任意盘但要是真文件."""
        for bad in ("../../../.env", "..\\..\\..\\.env", "cs2clipper/../../.env"):
            try:
                media.safe_under(bad, config.ROOT)
                raise AssertionError(f"越界路径未被拒绝: {bad}")
            except media.PathNotAllowed:
                pass
        inside = media.safe_under("cs2clipper/music.py", config.ROOT)
        is_true(inside.is_file(), "工作区内的正常路径被误拒")
        # 素材: 存在的音乐文件通过, 目录/错扩展名/空值被拒
        music = config.WORK_DIR / "selftest_track.wav"
        is_true(media.resolve_media(str(music), media.MUSIC_EXT, what="音乐").is_file(),
                "正常音乐文件被拒")
        for raw, err in ((str(config.WORK_DIR), IsADirectoryError),
                         (str(config.ROOT / "README.md"), ValueError),
                         ("", ValueError)):
            try:
                media.resolve_media(raw, media.MUSIC_EXT, what="音乐")
                raise AssertionError(f"非法素材未被拒绝: {raw!r}")
            except err:
                pass
        return "越界/目录/错扩展名/空值 均被拒"

    c("Web 端路径安全", media_path_safety)

    def app_builds_and_serves_static():
        app = webapp.create_app()
        paths = {getattr(r, "path", "") for r in app.routes}
        for need in ("/", "/api/meta", "/api/jobs", "/api/prefs", "/api/runs",
                     "/api/profile", "/api/artifact", "/api/download"):
            is_true(need in paths, f"路由缺失: {need}")
        static = webapp.STATIC_DIR
        for f in ("index.html", "app.css", "app.js"):
            is_true((static / f).is_file(), f"前端资源缺失: {f}")
        return f"{len(paths)} 条路由, 前端 3 个文件"

    c("Web 应用可构建", app_builds_and_serves_static)

    def frontend_static_consistency():
        """JS 里的选择器 / API 路径必须真实存在 (没有浏览器时的替代验证)."""
        import re
        static = webapp.STATIC_DIR
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "app.js").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        js_ids = set(re.findall(r"""\$\(\s*['"]#([A-Za-z0-9_\-]+)['"]""", js))
        missing = sorted(js_ids - html_ids)
        is_true(not missing, f"JS 选到了不存在的 id: {missing}")
        api_paths = {p.rstrip("/") or p for p in
                     re.findall(r"""['"](/api/[A-Za-z0-9_\-/]+)""", js)}
        app = webapp.create_app()
        routes = {getattr(r, "path", "") for r in app.routes}
        unknown = [p for p in api_paths
                   if p not in routes and not any(
                       r.startswith(p) or re.fullmatch(
                           re.sub(r"\{[^}]+\}", "[^/]+", r), p)
                       for r in routes if r.startswith("/api/"))]
        is_true(not unknown, f"JS 调用了不存在的接口: {unknown}")
        return f"{len(js_ids)} 个 id 选择器 / {len(api_paths)} 个接口 全部有效"

    c("前端选择器与接口路径有效", frontend_static_consistency)

    def hidden_elements_really_hide():
        """带 hidden 属性的元素必须真的被隐藏.

        真实缺陷 (用户报"关闭按钮点了没反应"): 浏览器 UA 样式里的
        `[hidden] { display: none }` 会被**任何**作者样式里的 display 覆盖,
        例如 `.modal { display: grid }`。JS 里 `el.hidden = true` 设上了、
        属性也确实变成 true, 元素却照样显示 —— 弹窗一直盖在页面上。

        这里逐条比对: index.html 里每个带 hidden 的元素 (以及 JS 里显隐过的
        id), 它的 class 在 CSS 里**不能**有 display 声明, 除非 CSS 里有
        `[hidden]` 的 !important 兜底规则。
        """
        import re
        static = webapp.STATIC_DIR
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "app.js").read_text(encoding="utf-8")
        css = (static / "app.css").read_text(encoding="utf-8")

        guarded = bool(re.search(r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", css))
        is_true(guarded, "CSS 里缺少 `[hidden] { display: none !important }` 兜底规则")

        # CSS: class -> 是否声明了 display
        display_cls = set()
        for block in re.findall(r"([^{}]+)\{([^}]*)\}", css):
            selectors, body = block
            if "display" not in body:
                continue
            for sel in selectors.split(","):
                for cls in re.findall(r"\.([A-Za-z][A-Za-z0-9_\-]*)", sel):
                    display_cls.add(cls)

        # 显隐目标: HTML 里带 hidden 的元素 + JS 里 show()/hidden 操作过的 id
        targets: dict[str, set[str]] = {}
        for tag in re.findall(r"<[^>]*\bhidden\b[^>]*>", html):
            mid = re.search(r'id="([^"]+)"', tag)
            mcl = re.search(r'class="([^"]+)"', tag)
            if mid:
                targets[mid.group(1)] = set((mcl.group(1) if mcl else "").split())
        for cid in set(re.findall(r"""show\(\s*\$\('#([A-Za-z0-9_\-]+)'\)""", js)) | \
                set(re.findall(r"""\$\('#([A-Za-z0-9_\-]+)'\)\.hidden""", js)) | \
                set(re.findall(r"""show\(\s*\$\('#([A-Za-z0-9_\-]+)'\)""", js)):
            # 从 HTML 里找这个 id 的 class
            m = re.search(rf'<[^>]*id="{re.escape(cid)}"[^>]*>', html)
            if m:
                mcl = re.search(r'class="([^"]+)"', m.group(0))
                targets.setdefault(cid, set((mcl.group(1) if mcl else "").split()))

        is_true(len(targets) >= 4, f"没找到显隐目标, 用例失效: {sorted(targets)}")
        broken = sorted(
            f"#{cid} (class={sorted(cl)} 里有 display 声明)"
            for cid, cl in targets.items() if cl & display_cls
        )
        ok_without_guard = sorted(
            cid for cid, cl in targets.items() if not (cl & display_cls)
        )
        # 兜底规则存在时, 即使 class 带 display 也是安全的
        is_true(guard_ok := (not broken or guarded),
                f"这些元素设了 hidden 也隐藏不掉: {broken}")
        return (f"{len(targets)} 个显隐目标; 需兜底规则的 {len(broken)} 个 "
                f"({broken}); 兜底规则={'有' if guarded else '无'}; "
                f"本身安全的 {len(ok_without_guard)} 个")

    c("hidden 元素真的会隐藏", hidden_elements_really_hide)

    def frontend_runs_without_top_level_error():
        """用 node + DOM 桩把 app.js **真执行**一遍.

        为什么必须真跑: index.html 用的是普通 <script> (没有 defer)。普通脚本
        一旦顶层抛出未捕获异常, 就从那一行起整体中断 —— 排在后面的
        addEventListener 全部不会绑定, 症状正是"某个按钮怎么点都没反应",
        而页面看起来一切正常。选择器/接口路径的静态检查抓不到这种问题
        (选择器全都对, 坏的是执行顺序)。

        真实缺陷: `$('#btn-demo-refresh')` 等一批绑定排在前面, 任何一行抛错
        都会让后面的 #modal-close 永远不绑 —— 用户点了没反应。
        脚本 tools/dev/domrun.js 会: 记录顶层异常、检查关键控件是否都绑上了
        监听器、模拟点击关闭并断言 modal 真的被隐藏。
        """
        import shutil
        import subprocess
        node = shutil.which("node")
        script = config.ROOT / "tools" / "dev" / "domrun.js"
        if not node or not script.is_file():
            return "跳过 (没有 node 或缺少 tools/dev/domrun.js)"
        r = subprocess.run(
            [node, str(script)], cwd=str(config.ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        is_true(r.returncode == 0, "前端无头执行失败:\n" + out[-1500:])
        tail = [ln.strip() for ln in out.splitlines()
                if "未绑定" in ln or "触发监听器" in ln or "关闭生效" in ln]
        return " | ".join(tail[-3:]) or "app.js 无头执行通过"

    c("前端可无头执行且控件都绑上了", frontend_runs_without_top_level_error)

    def assets_are_cache_busted():
        """静态资源 URL 必须带版本号.

        真实缺陷: 前端资源原来是固定路径 `/static/app.js`。浏览器对静态资源
        缓存优先级很高, 改了代码后普通刷新仍可能继续用旧文件 —— 表现就是
        "你改了我这儿还是老毛病", 用户被迫手动 Ctrl+F5。
        """
        import re
        app = webapp.create_app()
        # 直接调用 index handler 拿 HTML
        idx = next(r for r in app.routes if getattr(r, "path", "") == "/")
        import asyncio
        resp = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            idx.endpoint(None))
        html = resp.body.decode("utf-8")
        v_js = re.search(r'src="/static/app\.js\?v=(\d+)"', html)
        v_css = re.search(r'href="/static/app\.css\?v=(\d+)"', html)
        is_true(bool(v_js), "app.js 没有带版本号 (浏览器可能继续用旧缓存)")
        is_true(bool(v_css), "app.css 没有带版本号")
        is_true(resp.headers.get("cache-control") == "no-store",
                f"页面本身缺少 no-store: {resp.headers.get('cache-control')}")
        return f"app.js?v={v_js.group(1) if v_js else '-'}, 页面 no-store"

    c("静态资源带版本号避免旧缓存", assets_are_cache_busted)

    if not full:
        return

    def real_http_service():
        """真起一个 uvicorn 进程, 打一遍关键端点 + 跑一次两段出片看 SSE.

        用真进程而不是 TestClient: 要验证的正是"服务起得来、SSE 真能流"。
        """
        import json as _json
        import time as _time
        import urllib.error
        import urllib.parse
        import urllib.request

        sys_mod = __import__("sys")
        py = sys_mod.executable
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        proc = sp.Popen([py, "-m", "cs2clipper.webapp", "--port", str(port),
                         "--no-browser"],
                        cwd=str(config.ROOT), stdout=sp.PIPE, stderr=sp.STDOUT,
                        text=True, encoding="utf-8", errors="replace")

        def _get(path):
            req = urllib.request.Request(base + path)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    ctype = r.headers.get("content-type", "")
                    raw = r.read()
                    return r.status, (_json.loads(raw.decode("utf-8"))
                                      if "json" in ctype else raw)
            except urllib.error.HTTPError as e:
                try:
                    return e.code, _json.loads(e.read().decode("utf-8"))
                except Exception:
                    return e.code, {}

        def _post(path, body):
            req = urllib.request.Request(
                base + path, data=_json.dumps(body).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, _json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                try:
                    return e.code, _json.loads(e.read().decode("utf-8"))
                except Exception:
                    return e.code, {}

        try:
            t0 = _time.time()
            up = False
            while _time.time() - t0 < 40:
                with socket.socket() as s:
                    s.settimeout(0.4)
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        up = True
                        break
                _time.sleep(0.4)
            is_true(up, "uvicorn 40 秒内没起来")

            st, body = _get("/api/meta")
            is_true(st == 200 and len(body.get("pref_specs", {})) >= 15,
                    f"/api/meta 异常: {st}")
            st, body = _get("/api/artifact?path=" + urllib.parse.quote("../../../.env"))
            is_true(st == 403, f"路径穿越未被拒: {st}")
            st, body = _get("/api/runs?limit=3")
            is_true(st == 200 and "runs" in body, f"/api/runs 异常: {st}")

            st, job = _post("/api/jobs", {
                "music_path": str(config.WORK_DIR / "selftest_track.wav"),
                "demo_path": str(config.DEFAULT_DEMO),
                "aspect": "square", "max_clips": 2, "max_cards": 4,
                "use_llm": False, "use_prefs": False, "record": False,
                "out_dir": "out/_selftest_web",
            })
            is_true(st == 201 and job.get("ok"), f"建任务失败: {st} {job}")
            jid = job["job"]["id"]

            kinds, percents, stream_err = [], [], None
            req = urllib.request.Request(f"{base}/api/jobs/{jid}/events")
            t1 = _time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    while _time.time() - t1 < 300:
                        line = r.readline()
                        if not line:
                            break
                        line = line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        d = _json.loads(line[5:].strip())
                        if d.get("kind") == "__close__":
                            break
                        kinds.append(d["kind"])
                        if d.get("affects_percent"):
                            percents.append(d.get("percent", 0))
            except Exception as exc:
                stream_err = f"{type(exc).__name__}: {exc}"
            is_true(stream_err is None, f"SSE 流中断: {stream_err}")
            is_true("done" in kinds, f"SSE 没收到 done: {sorted(set(kinds))}")
            is_true(percents == sorted(percents), f"SSE 百分比倒退: {percents}")
            is_true(percents and percents[-1] == 100.0, f"SSE 未到 100%: {percents[-1:]}")

            st, snap = _get(f"/api/jobs/{jid}")
            js = snap.get("job", {})
            is_true(js.get("status") == "done", f"任务状态 {js.get('status')}: {js.get('error')}")
            vp = Path(js.get("video_path") or "")
            is_true(vp.is_file(), f"成片不存在: {vp}")
            return (f"端点全部可用; 任务 {len(kinds)} 条事件到 "
                    f"{percents[-1] if percents else '-'}%, 成片 "
                    f"{vp.stat().st_size // 1024 if vp.is_file() else 0} KB")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except sp.TimeoutExpired:
                proc.kill()

    c("真实 HTTP 服务 + 出片 + SSE", real_http_service)


# ==================================================================
# 9. HLAE 游戏内录制 (画面源 = hlae)
# ==================================================================
def test_hlae(c: Check) -> None:
    """HLAE 录制链路中**不需要启动游戏**就能验证的部分.

    真实验证"HLAE 注入 CS2 + 录到画面"需要 GUI、需要手动启动游戏, 自检里做不到
    (会一直等在那儿)。所以这里覆盖游戏以外的全部环节: 体检是否健壮、计划与脚本
    生成是否正确、录完之后能不能把素材接回流水线。
    """
    import shutil

    from PIL import Image

    from . import compose
    from . import hlaerec as H

    def preflight_shape():
        """体检必须**不抛异常**地返回结构化结果 (环境缺啥都只算 problems)."""
        pf = H.preflight(check_running=False)
        d = pf.to_dict()
        for key in ("ok", "problems", "warnings", "info"):
            is_true(key in d, f"体检结果缺少 {key}")
        is_true(isinstance(d["problems"], list), "problems 不是列表")
        for key in ("hlae_dir", "hlae_version", "cs2_exe", "hlae_config"):
            is_true(key in d["info"], f"体检缺少 info.{key}")
        is_true("体检结果" in pf.text(), "体检文本渲染异常")
        # 找不到 HLAE 时必须是 problems 而**不是**抛异常
        orig = H.hlae_dir
        try:
            H.hlae_dir = lambda: config.ROOT / "_no_such_hlae_dir"   # type: ignore
            bad = H.preflight(check_running=False)
            is_true(not bad.ok and bad.problems, "HLAE 缺失时体检竟然通过")
            is_true(any("HLAE.exe" in p for p in bad.problems),
                    f"问题描述没说清缺什么: {bad.problems}")
        finally:
            H.hlae_dir = orig                                        # type: ignore
        return (f"HLAE={d['info']['hlae_version'] or '未装'} "
                f"problems={len(d['problems'])} warnings={len(d['warnings'])}")

    c("HLAE 环境体检健壮", preflight_shape)

    def cs2_pick_largest_install():
        """多份 CS2 安装时必须挑**体积最大**的那份 (空壳安装真实存在)."""
        exe = H.find_cs2_exe()
        if exe is None:
            return "本机没装 CS2, 跳过"
        root = exe.parents[3]
        is_true(exe.is_file(), f"选中的 cs2.exe 不存在: {exe}")
        size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30
        is_true(size_gb > 1, f"选中了疑似空壳安装 ({size_gb:.1f} GB): {root}")
        return f"{root.name} ({size_gb:.1f} GB)"

    c("CS2 安装按体积择优", cs2_pick_largest_install)

    def plan_math():
        """EDL → 录制计划的时间换算与变速比必须自洽."""
        class Clip:
            def __init__(self, i, s, e, out, speed):
                self.index, self.src_start_tick, self.src_end_tick = i, s, e
                self.out_start, self.out_end = 0.0, out
                self.duration = out          # EDLClip.duration 是 out_end-out_start
                self.highlight_id, self.speed = f"c{i}", speed
                self.music_segment, self.effects = 0, {}

        class Card:
            def __init__(self, cid):
                self.id, self.player, self.round_num = cid, "P", 1

        tps = config.DEMO_TICKRATE
        clips = [
            Clip(0, tps * 100, tps * 102, 2.0, 1.0),    # 2s 素材 / 2s 成片 -> 1.0
            Clip(1, tps * 200, tps * 202, 4.0, 2.0),    # 2s 素材 / 4s 成片 -> 0.5
            Clip(2, tps * 300, tps * 304, 2.0, 1.0),    # 4s 素材 / 2s 成片 -> 2.0
        ]
        edl = type("EDL", (), {"clips": clips})()
        cards = [Card(f"c{i}") for i in range(3)]
        plan = H.plan_from_edl(edl, cards)
        is_true(len(plan) == 3, f"计划段数不对: {len(plan)}")
        approx(plan[0].demo_start_sec, 100.0, 1e-6)
        approx(plan[0].demo_end_sec, 102.0, 1e-6)
        approx(plan[0].realtime_factor, 1.0, 1e-6)
        approx(plan[1].realtime_factor, 0.5, 1e-6)
        approx(plan[2].realtime_factor, 2.0, 1e-6)
        is_true(plan[0].demo_span > 0, "demo 跨度为 0")
        names = [s.name for s in plan]
        is_true(len(set(names)) == len(names), f"片段名重复: {names}")
        is_true(isinstance(plan[0].to_dict(), dict), "to_dict 不是 dict")
        return f"3 段: realtime {[round(s.realtime_factor, 2) for s in plan]}"

    c("录制计划时间换算正确", plan_math)

    def scripts_are_safe():
        """生成的 CS2 脚本必须满足几条关键不变量.

        最重要的是**每段独立的 record name**: 第一版只在开头设一次名字, 结果
        各段输出落到同一目录互相覆盖, 最后只剩最后一段。
        """
        plan = [
            H.RecSegment(index=i, highlight_id=f"c{i}", player=f"P{i}",
                         round_num=i + 1, demo_start_sec=10.0 * i,
                         demo_end_sec=10.0 * i + 2.5, out_duration=2.5,
                         speed=1.0, name=f"clip_{i + 1:03d}")
            for i in range(4)
        ]
        boot, stop, clips = H.build_cs2_config(
            plan, demo_path=r"D:\x\a.dem", output_dir=r"E:\out\rec", fps=60)
        is_true(len(clips) == 4, f"片段脚本数量不对: {len(clips)}")
        is_true("playdemo" in boot, "引导脚本里没有 playdemo")
        is_true("mirv_streams record fps 60" in boot, "引导脚本没设录制帧率")
        is_true("bind F8" in boot, "引导脚本没绑停录键")
        is_true("mirv_streams record end" in stop, "停录脚本没有 record end")

        rec_names: list[str] = []
        for name, text in clips:
            is_true(name.endswith(".cfg"), f"片段脚本名不对: {name}")
            is_true("mirv_streams record start" in text, f"{name} 没有 record start")
            is_true("mirv_skip time to" in text, f"{name} 没有定位到片段起点")
            for line in text.splitlines():
                if line.startswith("mirv_streams record name"):
                    rec_names.append(line)
        is_true(len(rec_names) == len(set(rec_names)),
                f"各段 record name 有重复 -> 输出会互相覆盖: {rec_names}")
        for i, (_, text) in enumerate(clips):
            want = f"mirv_skip time to {10.0 * i:.3f}"
            is_true(want in text, f"片段 {i + 1} 没跳到自己的起点 (期望 {want})")

        # 玩家名里的引号/分号必须被转义, 否则会破坏 cfg (用户名可以随便起)
        weird = [H.RecSegment(index=0, highlight_id="c", player='A"; quit; "B',
                              round_num=1, demo_start_sec=1.0, demo_end_sec=2.0,
                              out_duration=1.0, speed=1.0, name="clip_001")]
        _, _, wclips = H.build_cs2_config(
            weird, demo_path="d", output_dir="o", fps=60)
        spec = [l for l in wclips[0][1].splitlines() if l.startswith("spec_player")]
        is_true(len(spec) == 1, f"spec_player 行数异常: {spec}")
        is_true(spec[0].count('"') == 2, f"玩家名引号没转义: {spec[0]}")
        return f"{len(clips)} 个片段脚本, record name 全唯一, 恶意玩家名已转义"

    c("录制脚本不变量", scripts_are_safe)

    def frames_to_exact_duration():
        """帧序列 → 精确时长片段 (与 compose.finalize_clip 的契约一致).

        这是"录制素材能接回流水线"的关键: 无论录了多少帧, 产出片段的时长必须
        **严格等于** EDL 声明的时长, 否则视频轨与音乐轨会对不上。
        """
        tmp = config.WORK_DIR / "_hlae_frames"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(30):      # 30 帧 @60fps = 0.5s 素材
                Image.new("RGB", (160, 90), (i * 8 % 256, 40, 90)).save(
                    tmp / f"frame_{i:08d}.png")
            out = config.WORK_DIR / "_hlae_clip_test.mp4"
            H.frames_to_clip(tmp, out, fps=60, target_duration=2.0, size=(160, 90))
            dur = compose.probe_duration(out)
            is_true(abs(dur - 2.0) < 0.15,
                    f"产出时长 {dur:.2f}s 与目标 2.0s 不符 (慢放/拉伸没生效?)")
            out.unlink(missing_ok=True)
            return f"30 帧(0.5s) -> 拉伸到 {dur:.2f}s"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    c("帧序列规整到精确时长", frames_to_exact_duration)

    def missing_material_is_reported():
        """录制素材缺失时必须**报出来**, 而不是静默少一段.

        少了不说, 成片就会比音乐短一截而且没人知道为什么。
        """
        tmp = config.WORK_DIR / "_hlae_rec_probe"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            empty = H.discover_recordings(tmp)
            is_true(empty["exists"] and not empty["frame_dirs"],
                    f"空目录探测结果异常: {empty}")
            plan = [
                H.RecSegment(index=0, highlight_id="a", player="P", round_num=1,
                             demo_start_sec=0.0, demo_end_sec=1.0,
                             out_duration=1.0, speed=1.0, name="clip_001"),
                H.RecSegment(index=1, highlight_id="b", player="Q", round_num=2,
                             demo_start_sec=2.0, demo_end_sec=3.0,
                             out_duration=1.0, speed=1.0, name="clip_002"),
            ]
            items, notes = H.build_from_recordings(plan, tmp, tmp / "clips", fps=60)
            is_true(items == [], f"没有素材却产出了片段: {items}")
            is_true(len(notes) == 2, f"两段都缺素材, 却只报 {len(notes)} 条")
            is_true(all("clip_00" in n for n in notes), f"说明里没带片段名: {notes}")
            return f"空目录 -> 0 片段 + {len(notes)} 条说明"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    c("素材缺失会明确报出", missing_material_is_reported)

    def pipeline_routing():
        """record_source 必须真的把 render 分流到 HLAE 路径.

        用"录制目录为空"这个必然状态做探针: HLAE 路径会明确报错并给出下一步,
        而**不会**去画雷达帧。若分流没生效, 这里会一路渲染成功 —— 那说明开关
        是假的。
        """
        from . import pipeline as P

        track = config.WORK_DIR / "selftest_track.wav"
        if not track.is_file():
            from . import music as M
            M.make_test_track(track)
        out = config.WORK_DIR / "_hlae_route"
        try:
            P.run(str(track), None, out_dir=out, max_clips=2, max_cards=4,
                  use_llm=False, verbose=False, use_prefs=False, record=False,
                  record_source="hlae")
            raise AssertionError("record_source=hlae 却没有走 HLAE 路径")
        except RuntimeError as e:
            msg = str(e)
            is_true("录制素材" in msg or "录制脚本" in msg,
                    f"HLAE 路径的报错没说清下一步: {msg[:160]}")
        # 未知取值必须被拒 (而不是静默当成 radar)
        try:
            P.run(str(track), None, out_dir=out, max_clips=2, max_cards=4,
                  use_llm=False, verbose=False, use_prefs=False, record=False,
                  record_source="nonsense")
            raise AssertionError("未知 record_source 未被拒绝")
        except ValueError:
            pass
        return "hlae 分流生效, 未知取值被拒"

    c("record_source 分流与校验", pipeline_routing)


# ==================================================================
# 汇总
# ==================================================================
def summarize() -> int:
    groups: dict[str, list[tuple[str, str]]] = {}
    for g, n, s in RESULTS:
        groups.setdefault(g, []).append((n, s))
    passed = sum(1 for _, _, s in RESULTS if s.startswith("PASS"))
    failed = len(RESULTS) - passed
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    for g, items in groups.items():
        ok = sum(1 for _, s in items if s.startswith("PASS"))
        print(f"\n[{g}]  {ok}/{len(items)} 通过")
        for n, s in items:
            if not s.startswith("PASS"):
                print(f"    ✗ {n}\n      {s}")
    print("\n" + "-" * 78)
    print(f"总计 {len(RESULTS)} 项: {passed} 通过, {failed} 失败")
    if failed:
        print("\n失败项:")
        for g, n, s in RESULTS:
            if not s.startswith("PASS"):
                print(f"  - [{g}] {n}: {s}")
    print("-" * 78)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cs2clipper 完整自检")
    ap.add_argument("--e2e", action="store_true", help="跑端到端出片 (慢)")
    ap.add_argument("--llm", action="store_true", help="端到端时额外验证 LLM 编排")
    ap.add_argument("--music", default=None, help="真实音乐路径 (不传则用合成测试曲)")
    ap.add_argument("--aspects", default="wide", help="端到端画幅, 逗号分隔")
    ap.add_argument("--no-web", action="store_true",
                    help="跳过 Web 界面的真实服务测试 (那一步会真起 uvicorn 并出一次片)")
    args = ap.parse_args(argv)

    music = Path(args.music) if args.music else config.WORK_DIR / "selftest_track.wav"
    if not music.is_file():
        from . import music as M
        M.make_test_track(music)

    print("=" * 78)
    print(f"cs2clipper 自检   音乐={music.name}   e2e={args.e2e}   llm={args.llm}")
    print("=" * 78)

    print("\n[1/10] 环境")
    test_environment(Check("环境"))
    print("\n[2/10] 音乐分析")
    test_music(Check("音乐分析"), Path(args.music) if args.music else None)
    print("\n[3/10] demo 解析")
    test_demo(Check("demo"))
    print("\n[4/10] 编排 (EDL)")
    test_planner(Check("编排"))
    print("\n[5/10] 渲染")
    test_render(Check("渲染"))
    print("\n[6/10] 合成")
    test_compose(Check("合成"))
    print("\n[7/10] 偏好存储")
    test_store(Check("存储"))
    print("\n[8/10] Web 界面")
    test_web(Check("Web"), full=not args.no_web)
    print("\n[9/10] HLAE 游戏内录制")
    test_hlae(Check("HLAE"))

    if args.e2e:
        aspects = [a.strip() for a in args.aspects.split(",") if a.strip()]
        modes = [False, True] if args.llm else [False]
        for use_llm in modes:
            print(f"\n[10/10] 端到端 (llm={use_llm})")
            test_e2e(Check(f"端到端 llm={use_llm}"), music, aspects, use_llm)
    else:
        print("\n[10/10] 端到端  (跳过, 加 --e2e 启用)")

    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
