"""Stage 1 —— 音乐分析.

把一首歌拆成"可剪辑的时间结构":
    * BPM 与节拍时刻 (卡点用)
    * 能量曲线 (RMS) —— 决定剪辑密度
    * 频谱质心/滚降 —— 决定画面"亮度"
    * 低频能量占比 —— 决定冲击感
    * 自动分段 (intro / build / drop / outro ...) + 情绪标签

设计原则: 全部离线、确定性、可复现。LLM 只消费这里产出的**结构化数字**,
不让 LLM 直接"听"音频, 这样剪辑决策才稳定、可调试。

情绪模型采用轻量二维:
    arousal (激烈度) = 能量 + 低频冲击 + 节拍密度
    valence (正负向) = 频谱质心 + 高频占比  (亮=正向, 暗=压抑)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config  # noqa: F401  (环境重定向, 必须最先)

import librosa  # noqa: E402

# 分析参数
SAMPLE_RATE = 22050
HOP_LENGTH = 512          # ≈ 23ms per frame @22050
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH   # ≈ 43.07

# 低频段 (kick / bass): 用于"冲击感"; 高频亮度改用频谱质心 + rolloff 描述,
# 不再单独统计高频占比 (质心/rolloff 已是频率尺度, 语义更直接)。
LOW_BAND = (20.0, 160.0)


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------
@dataclass
class Segment:
    """一个音乐段落."""

    index: int
    start: float
    end: float
    label: str              # 结构化标签: intro/verse/build/chorus/drop/breakdown/outro
    emotion: str            # 中文描述, 给 LLM 看的
    arousal: float          # 0-1 激烈度 (决定剪辑密度)
    brightness: float       # 0-1 频谱亮度 (决定画面色彩/特效)
    energy: float           # 0-1 归一化 RMS
    low_ratio: float        # 低频能量占比 (冲击感)
    centroid: float         # 归一化频谱质心
    onset_rate: float       # 每秒 onset 数 (节拍密度)
    bpm_local: float        # 段内局部 BPM
    rel_energy: float = 0.0  # 相对全曲中位的能量 (-1..1), 供编排判断轻重

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


@dataclass
class MusicAnalysis:
    """整首歌的分析结果."""

    path: str
    duration: float
    bpm: float
    beat_times: list[float]
    beats_per_second: float
    segments: list[Segment]
    energy_curve: list[float]          # 归一化, 实际采样率见 energy_curve_hz (~2/s)
    energy_curve_hz: float
    duration_sec: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "bpm": round(self.bpm, 2),
            "beat_count": len(self.beat_times),
            "beats_per_second": round(self.beats_per_second, 3),
            "segments": [s.to_dict() for s in self.segments],
            "energy_curve": [round(v, 4) for v in self.energy_curve],
            "energy_curve_hz": self.energy_curve_hz,
            **self.extra,
        }

    def llm_view(self) -> dict[str, Any]:
        """压缩后给 LLM 的视图 —— 不含逐帧数据, 只保留段落级结构."""
        return {
            "duration": round(self.duration, 2),
            "bpm": round(self.bpm, 2),
            "beat_count": len(self.beat_times),
            "dynamic_range": self.extra.get("dynamic_range"),
            "peak_vs_median": self.extra.get("peak_vs_median"),
            "peak_vs_second": self.extra.get("peak_vs_second"),
            "has_climax": self.extra.get("has_climax"),
            "segments": [
                {
                    "index": s.index,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "duration": round(s.duration, 2),
                    "label": s.label,
                    "emotion": s.emotion,
                    "arousal": round(s.arousal, 3),
                    "brightness": round(s.brightness, 3),
                    "onset_rate": round(s.onset_rate, 2),
                    "bpm_local": round(s.bpm_local, 1),
                }
                for s in self.segments
            ],
        }


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _norm(x: np.ndarray) -> np.ndarray:
    """按分位数归一化到 0-1, 对离群值稳健 (逐帧特征用)."""
    if x.size == 0:
        return x
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _norm_segments(vals: np.ndarray) -> np.ndarray:
    """段落级归一化: 按段间 min-max 拉伸, **不做百分位裁剪**.

    段落通常只有 4-12 个, 用百分位裁剪会把最强/最弱段压成 0 或 1,
    导致情绪标签反转 (例如最炸的 drop 被判成"最暗")。这里改成
    相对极值拉伸, 保留段间序关系。
    """
    if vals.size == 0:
        return vals
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-9:
        return np.full_like(vals, 0.5)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def _band_ratio(spec: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """某频段能量占总能量的比例 (逐帧)."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    total = spec.sum(axis=0) + 1e-9
    return spec[mask].sum(axis=0) / total


def _label_segments(
    stats: list[dict[str, float]],
    has_climax: bool = True,
) -> list[str]:
    """按"歌曲自身的动态结构"给段落打标签.

    旧实现有两个真实缺陷 (用真实音乐才暴露出来):
      1. 硬编码"排名第 0 的段 = drop"。对《英雄主义》这种能量接近平稳
         (RMS 0.07~0.13, 全曲无明显高潮) 的歌, 首段因为微弱领先就被判成
         "高潮爆发段", 属于凭空捏造结构。
      2. `rank <= n//4` 配 6 个段落时是 2 个, 于是前 4 段里 3 段都叫 verse,
         标签几乎没有信息量。
      3. 取最强段做 drop 时会忽略位置: 歌曲**开头**最响的那段被叫 drop 是错的
         (drop 是"落差之后"的意思, 必须有前文铺垫)。

    新实现基于**峰值显著度** (见 energy_stats):
      * drop  : 最响段显著高于全曲中位, 且明显高于次高段 -> 真有落差
      * intro : 开头且明显弱于中位
      * outro : 结尾且明显弱于中位
      * build : 能量低于中位, 且仍在爬升 (用于铺垫)
      * breakdown : 明显弱于中位又不在首尾的留白段
      * verse : 其余常态段
    """
    n = len(stats)
    if n == 0:
        return []
    energy = np.array([s["energy"] for s in stats])
    med = float(np.median(energy))
    lo, hi = float(energy.min()), float(energy.max())
    span = max(hi - lo, 1e-9)
    rel = (energy - med) / span          # 相对全曲中位, 已按极差缩放

    labels = [""] * n
    for i, s in enumerate(stats):
        pos = i / max(n - 1, 1)           # 0=开头 1=结尾
        e = energy[i]
        rising = False
        if i + 1 < n:
            rising = energy[i + 1] > e + 0.02 * span

        # 最强段: 必须**全曲真有高潮** (见 energy_stats), 且显著高于中位,
        # 位置也不能在开头 (drop 是"落差之后", 必须有前文铺垫)。
        if (has_climax and i == int(np.argmax(energy)) and rel[i] >= 0.12
                and n > 2 and 0.1 < pos < 0.95):
            labels[i] = "drop"
            continue
        # 首尾的安静段 —— 位置要求收紧到 12%: 否则歌曲 19% 处的弱段会被
        # 叫成 "intro"(开场), 与实际位置矛盾。
        if pos <= 0.12 and rel[i] <= -0.10:
            labels[i] = "intro"
            continue
        if pos >= 0.88 and rel[i] <= -0.10:
            labels[i] = "outro"
            continue
        # 中间明显偏弱的留白
        if rel[i] <= -0.20:
            labels[i] = "breakdown"
            continue
        # 低于中位且在爬升 -> 铺垫
        if rel[i] < 0 and rising:
            labels[i] = "build"
            continue
        labels[i] = "verse"
    # 全曲偏平时不给"高潮"标签 —— 宁可老实叫 verse, 也不要编造结构。
    # 判据由调用方通过 has_climax 传入 (基于峰值显著度), 这里只负责兜底:
    # 任何调用方都拿不到假 drop。
    if not has_climax:
        labels = ["verse" if lb == "drop" else lb for lb in labels]
    return labels


def _emotion_text(
    arousal: float,
    brightness: float,
    label: str,
    *,
    rel_energy: float = 0.0,
) -> str:
    """二维描述 → 中文 (给 LLM 的自然语言线索).

    两个轴是**独立**的, 不假设"亮=好/炸":
        arousal    激烈度 —— 决定剪辑密度与镜头时长
        brightness 频谱亮度 —— 决定画面色彩/对比/特效强度
    底鼓重的 drop 可以同时是"高激烈度 + 暗", 这是正确且常见的。

    注意: 旧版在"高激烈度 + 暗"时输出"沉重压迫", 这在**全曲最激烈的段落**
    上会自相矛盾 (既是最炸的又是被压迫的)。所以这里用 rel_energy —— 该段相对
    全曲中位的能量 —— 来消歧: 相对更响的暗色段是"力量感", 更轻的才是"压抑"。
    """
    a = "高" if arousal >= 0.66 else ("中" if arousal >= 0.33 else "低")
    b = "亮" if brightness >= 0.66 else ("中性" if brightness >= 0.33 else "暗")

    # 结构标签优先: 段落在全曲里的角色比频谱亮度更能说明用途
    if label == "drop":
        return "高潮爆发段，适合整段最精彩的连杀与快切"
    if label == "build":
        return "推进上行段，适合逐渐加快节奏、铺垫情绪"
    if label == "intro":
        return "开场铺垫段，适合地图全景、入场站位或慢镜引入"
    if label == "outro":
        return "收束段，适合结尾定格或余韵镜头"
    if label == "breakdown":
        return "留白段，适合残局对峙、慢镜头或静步架枪"

    # 暗色高激烈需要二选一, 用相对能量决定
    if a == "高" and b == "暗":
        if rel_energy >= 0.0:
            return "厚重有力，适合强攻、道具覆盖与正面交火（全曲偏响）"
        return "压制感强，适合劣势局、被压制的防守回合（全曲偏轻）"
    if a == "高" and b == "亮":
        return "亢奋明亮，适合高光连杀与快切"
    if a == "高":
        return "紧张推进，适合交火与残局"

    table = {
        ("中", "亮"): "舒展上行，适合转点与运营",
        ("中", "中性"): "平稳推进，适合常规交火",
        ("中", "暗"): "克制悬疑，适合静步与架枪",
        ("低", "亮"): "安静空灵，适合开局准备与地图全景",
        ("低", "中性"): "平静铺垫，适合入场与站位",
        ("低", "暗"): "低沉收敛，适合劣势与等待",
    }
    return f"{table.get((a, b), '中性')}（{label}，激烈度{a}／亮度{b}）"


def build_beat_grid(beat_times: list[float], duration: float, step: float = 0.5) -> list[float]:
    """把节拍时刻转成可用于卡点的网格 (过滤过密/过疏的拍点)."""
    if not beat_times:
        return [round(t, 3) for t in np.arange(0.0, duration, step)]
    out: list[float] = []
    last = -1e9
    for t in beat_times:
        if t - last >= 0.25:      # 至少间隔 250ms, 避免切太碎
            out.append(round(float(t), 3))
            last = t
    return out


def snap_to_beat(t: float, beat_times: list[float], tol: float = 0.35) -> float:
    """把时间点吸附到最近节拍 (容差内), 用于卡点剪辑."""
    if not beat_times:
        return t
    arr = np.asarray(beat_times)
    i = int(np.argmin(np.abs(arr - t)))
    return float(arr[i]) if abs(arr[i] - t) <= tol else t


# ------------------------------------------------------------------
# 主分析函数
# ------------------------------------------------------------------
# --- "这首歌到底有没有高潮" 的判据 ---
# 实测教训: 旧判据是 (max-min)/median, 分子取了**含最安静段**的极差。淡出段
# (RMS 0.0084) 会把极差撑到中位的 1.14 倍, 于是 docstring 自己举例的
# "RMS 0.07~0.13、全曲无明显高潮"的《英雄主义》被判定 has_climax=True,
# 保护彻底失效 -> 平稳歌照样被判出 drop。而且该比值量纲无界 (实测三首歌
# 全在 0.74~1.14), 拿它跟固定的 0.35 比较本身就是量纲错配。
#
# 现在只认**峰值显著度**: 最响的那段必须同时 (a) 绝对高于全曲中位一档,
# (b) 明显高于次高段 —— 也就是真的"有落差/有铺垫", 而不只是赢在淡出段
# 把极差撑大。两个条件都是无量纲比值, 与曲目整体响度无关。
PEAK_VS_MEDIAN = 1.25      # 峰值段至少比中位段响 25%
PEAK_VS_SECOND = 1.15      # 峰值段至少比次高段响 15% (否则只是并列最响)
PEAK_PROMINENCE = 0.35     # 或 (P90-P10)/median 足够大 (存在"平 plateau vs 峰值"落差)
# 硬下限: 无论走哪条路, 峰值段都必须比中位段响这么多。
# 为什么取 1.24 而不是更松的值: 实测一本"几乎没有动态"的歌(段级 RMS 只在中位
# 的 0.92~1.22 倍之间浮动, 极差仅 4%)靠 prominence 这一条也能凑出 1.215 的
# peak/median。判定宁可保守 —— 少标一个 drop 只是少一次快切, 凭空造一个 drop
# 会让整段情绪与实际音乐相反。
PEAK_FLOOR = 1.24


def smooth_frame(x: "np.ndarray", win: int) -> "np.ndarray":
    """滑动平均 (edge padding), 长度不变.

    为什么不用 `np.convolve(x, kern, mode="same")`: 它把窗口外的部分当 **0**,
    于是曲线首尾各约 win/2 被压成假淡入/假淡出 (实测真歌曲首点低 48%、
    末点低 98%)。除了"导出的 energy_curve 头尾是人工痕迹"之外, start/end
    附近的 |diff| 会被人为抬高 —— 调用方若传小 min_segment 就会在这些位置
    选出假边界。

    正确做法: 窗口覆盖到信号之外时, 用**边界值本身**补齐 (edge padding),
    而不是补 0。这个实现满足一个便于测试的不变量: 常值序列逐点不变
    (零填充会把首尾压低约 half/win)。
    """
    x = np.asarray(x, dtype=float)
    win = max(int(win), 1)
    n = x.size
    if win == 1 or n == 0:
        return x.copy()
    half = win // 2
    out = np.empty(n, dtype=float)
    # 中间部分: 窗口 = [i-half, i+half] (win+1 个点, half 为整数; 只影响窗口
    # 比 win 多一个点, 与 np.convolve 'same' 的偶数窗口对齐方式略有不同,
    # 但不影响平滑语义)。用累积和算, O(n)。
    lo, hi = min(half, n), max(n - half, 0)
    if hi > lo:
        csum = np.concatenate(([0.0], np.cumsum(x)))
        idx = np.arange(lo, hi)
        out[lo:hi] = (csum[np.minimum(idx + half + 1, n)]
                      - csum[np.maximum(idx - half, 0)]) / (2 * half + 1)
    # 首尾: 窗口越界处用**边界值本身**补齐 (edge padding), 而不是补 0
    for i in list(range(lo)) + list(range(hi, n)):
        l, r = i - half, i + half
        part = x[max(l, 0):min(r + 1, n)]
        pad = (r - l + 1) - part.size
        edge = x[0] if l < 0 else x[-1]
        out[i] = (np.sum(part) + pad * edge) / (2 * half + 1)
    return out


def energy_stats(raw_energies: "np.ndarray") -> dict[str, float]:
    """由段落级原始能量算"动态范围 / 峰值显著度 / 是否真有高潮"。

    独立成函数是为了可测: 这些量必须能从**实测值**复算, 而不是只能在
    完整 librosa 流程里看到结果。
    """
    e = np.asarray(raw_energies, dtype=float)
    med = float(np.median(e)) if e.size else 0.0
    denom = max(abs(med), 1e-9)
    peak = float(e.max()) if e.size else 0.0
    srt = np.sort(e)[::-1] if e.size else np.array([0.0])
    second = float(srt[1]) if srt.size > 1 else 0.0
    peak_vs_median = peak / denom
    peak_vs_second = peak / max(abs(second), 1e-9)
    p90, p10 = (float(np.percentile(e, 90)), float(np.percentile(e, 10))) if e.size else (0.0, 0.0)
    prominence = (p90 - p10) / denom
    has_climax = (
        e.size >= 3
        and peak_vs_median >= PEAK_FLOOR
        and (
            peak_vs_median >= PEAK_VS_MEDIAN
            or peak_vs_second >= PEAK_VS_SECOND
            or prominence >= PEAK_PROMINENCE
        )
    )
    return {
        "energy_median": med,
        "energy_peak": peak,
        "peak_vs_median": round(peak_vs_median, 4),
        "peak_vs_second": round(peak_vs_second, 4),
        "peak_prominence": round(prominence, 4),
        # 离硬下限还有多少余量: 余量很小的曲子换个分段/编码就可能翻转结论,
        # 直接报出来, 免得把"勉强通过"当成"确定有高潮"。
        "peak_margin": round(peak_vs_median / PEAK_FLOOR - 1.0, 4),
        # 保留旧字段名与量纲, 但它现在**只用于展示**, 不再参与判定
        "dynamic_range": round((peak - float(e.min())) / denom if e.size else 0.0, 4),
        "has_climax": bool(has_climax),
    }


def analyze_music(
    path: str | Path,
    *,
    max_duration: float | None = None,
    min_segment: float = 6.0,
    target_segments: int | None = None,
) -> MusicAnalysis:
    """分析一首歌, 返回 MusicAnalysis.

    Args:
        path: 音频文件路径 (wav/mp3/flac/m4a 等, 由 soundfile/audioread 支持)
        max_duration: 只分析前 N 秒 (调试用)
        min_segment: 段落最小长度(秒), 避免切太碎
        target_segments: 期望段落数; 默认按曲长自适应 (每 ~20s 一段)
    """
    path = Path(path)
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=max_duration)
    if y.size == 0:
        raise ValueError(f"音频为空或无法解码: {path}")
    duration = float(len(y) / sr)

    # --- 节拍 ---
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)]

    # --- 逐帧特征 ---
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr)
    low_ratio = _band_ratio(S, freqs, LOW_BAND)

    n_frames = min(len(rms), len(onset_env), len(cent), len(rolloff), len(low_ratio))
    rms, onset_env, cent, rolloff, low_ratio = (
        rms[:n_frames], onset_env[:n_frames], cent[:n_frames],
        rolloff[:n_frames], low_ratio[:n_frames],
    )

    rms_n = _norm(rms)
    onset_n = _norm(onset_env)
    cent_n = _norm(cent)
    low_n = _norm(low_ratio)

    # 综合指标
    arousal_frame = np.clip(0.5 * rms_n + 0.3 * low_n + 0.2 * onset_n, 0, 1)
    brightness_frame = np.clip(0.7 * cent_n + 0.3 * _norm(rolloff), 0, 1)

    # --- 平滑后找段落边界 ---
    # 见 smooth_frame: 必须用 edge padding 而不是 convolve(mode="same"),
    # 否则曲线首尾会被零填充压成假淡入/假淡出。
    smooth_win = max(int(2.0 * FRAMES_PER_SEC), 8)     # ~2 秒平滑
    arousal_s = smooth_frame(arousal_frame, smooth_win)

    if target_segments is None:
        # 每 ~12 秒一段: 20 秒太粗, 真实歌曲会出现 48-64 秒的超长段落,
        # 对剪辑来说没有参考价值 (一段里其实包含好几个情绪层次)。
        target_segments = int(np.clip(round(duration / 12.0), 4, 16))

    # 边界: 在 arousal 变化率最大的位置切, 且满足最小段落长度
    min_frames = int(min_segment * FRAMES_PER_SEC)
    diff = np.abs(np.diff(arousal_s))
    # 候选边界按变化强度降序, 贪心挑选, 保证间距
    cand = np.argsort(diff)[::-1]
    chosen: list[int] = []
    for c in cand:
        if len(chosen) >= target_segments - 1:
            break
        if c < min_frames or c > n_frames - min_frames:
            continue
        if all(abs(c - x) >= min_frames for x in chosen):
            chosen.append(int(c))
    bounds = [0] + sorted(chosen) + [n_frames]

    # 把段落边界吸附到最近的拍点: 段落起点对齐到拍子后, 剪辑点才会落在
    # 音乐的重音上, 而不是拍与拍之间 (旧版边界与节拍网格完全无关)。
    # 注意: 吸附必须是**逐边界顺序判断**的 —— 旧实现在吸附被拒时只回退当前
    # 边界, 却保留了上一个已吸附的边界, 于是"最小值"约束被绕过, 会切出
    # 短于 min_segment 的段落 (实测 min_segment=6s 时出现 5.5s 的段)。
    # 被拒的边界也不允许影响后续判断, 否则一次拒绝会连锁压缩后面的段落。
    if beat_frames.size:
        bf = np.asarray(beat_frames, dtype=int)
        snapped = [bounds[0]]
        for b in bounds[1:-1]:
            i = int(np.argmin(np.abs(bf - b)))
            sb = int(bf[i])
            if sb - snapped[-1] >= min_frames and n_frames - sb >= min_frames:
                snapped.append(sb)
            elif b - snapped[-1] >= min_frames and n_frames - b >= min_frames:
                snapped.append(b)
            else:
                # 原始边界同样不满足最小段长 -> 直接丢弃这个边界
                continue
        bounds = snapped + [bounds[-1]]

    # --- 汇总每段 ---
    raw_stats: list[dict[str, float]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg_onset = onset_env[a:b]
        # onset 密度: 每秒超过阈值的 onset 个数
        thr = float(np.percentile(onset_env, 75)) if onset_env.size else 0.0
        onset_rate = float((seg_onset > thr).sum() / max((b - a) / FRAMES_PER_SEC, 1e-6))
        raw_stats.append(
            {
                "start": a / FRAMES_PER_SEC,
                "end": b / FRAMES_PER_SEC,
                "energy": float(rms_n[a:b].mean()) if b > a else 0.0,
                "_raw_energy": float(rms[a:b].mean()) if b > a else 0.0,
                "arousal": float(arousal_s[a:b].mean()) if b > a else 0.0,
                "brightness": float(brightness_frame[a:b].mean()) if b > a else 0.0,
                "low_ratio": float(low_ratio[a:b].mean()) if b > a else 0.0,
                "centroid": float(cent_n[a:b].mean()) if b > a else 0.0,
                "onset_rate": onset_rate,
            }
        )

    # 段间再做一次相对归一化, 让"哪段最炸"更突出。
    # 用 _norm_segments: 段数少, 不能做百分位裁剪, 否则段间序关系会被破坏。
    for key in ("energy", "arousal", "brightness"):
        vals = np.array([s[key] for s in raw_stats])
        nn = _norm_segments(vals)
        for s, v in zip(raw_stats, nn):
            s[key] = float(v)

    # 动态范围 / 峰值显著度 / 是否真有高潮: 判定内聚在 energy_stats 里,
    # 这样任何调用方都拿不到"凭空造出来的 drop"。
    raw_energies = np.array([s["_raw_energy"] for s in raw_stats])
    estats = energy_stats(raw_energies)
    med_raw = estats["energy_median"]
    dynamic_range = estats["dynamic_range"]
    span_raw = max(float(raw_energies.max() - raw_energies.min()), 1e-9) if raw_energies.size else 1e-9
    for s in raw_stats:
        s["rel_energy"] = float((s["_raw_energy"] - med_raw) / span_raw)

    labels = _label_segments(raw_stats, has_climax=estats["has_climax"])

    segments: list[Segment] = []
    bt = np.asarray(beat_times) if beat_times else np.array([])
    for i, s in enumerate(raw_stats):
        # 段内局部 BPM: 必须用**段内拍点跨度**算, 并设拍数下限。
        # `拍数/段时长` 在淡出/留白段会塌陷 (实测末段 32.6 vs 全局 117.5),
        # 而这个数字会原样进 llm_view 喂给 LLM, 会把它误导成"这段 32 BPM"。
        if bt.size:
            in_seg = bt[(bt >= s["start"]) & (bt < s["end"])]
            if in_seg.size >= 4:
                span_beat = float(in_seg[-1] - in_seg[0])
                local_bpm = (in_seg.size - 1) / span_beat * 60.0 if span_beat > 1e-6 else bpm
            else:
                local_bpm = bpm
        else:
            local_bpm = bpm
        seg = Segment(
            index=i,
            start=round(s["start"], 3),
            # 帧数/FPS 会比真实音频时长多约一个 hop (~23ms), 掐到 duration
            end=round(min(s["end"], duration), 3),
            label=labels[i],
            emotion=_emotion_text(
                s["arousal"], s["brightness"], labels[i],
                rel_energy=s.get("rel_energy", 0.0),
            ),
            arousal=round(s["arousal"], 4),
            brightness=round(s["brightness"], 4),
            energy=round(s["energy"], 4),
            low_ratio=round(s["low_ratio"], 4),
            centroid=round(s["centroid"], 4),
            onset_rate=round(s["onset_rate"], 3),
            bpm_local=round(float(local_bpm), 2),
            rel_energy=round(float(s.get("rel_energy", 0.0)), 4),
        )
        segments.append(seg)

    # 给 LLM 的能量曲线: 约每秒 2 点。`curve_hz` 必须报**真实**采样率:
    # step = int(43.0664/2) = 21 -> 真实 2.0508Hz, 旧代码硬写 2.0, 任何
    # 用 i/hz 反推时间的下游都会累积漂移 (300s 约 3.7s)。
    step = max(int(FRAMES_PER_SEC / 2.0), 1)
    curve_hz = FRAMES_PER_SEC / step
    curve = arousal_s[::step]
    curve = [round(float(v), 4) for v in curve]

    bps = 0.0
    if duration > 0 and len(beat_times) >= 2:
        # 用拍点跨度而不是整曲时长: 无拍的前奏/尾奏会把平均拍率压低,
        # 与同一个 JSON 里的 bpm 自相矛盾 (实测 1.825 vs bpm 117.5)。
        beat_span = float(beat_times[-1] - beat_times[0])
        bps = (len(beat_times) - 1) / beat_span if beat_span > 1e-6 else 0.0

    return MusicAnalysis(
        path=str(path),
        duration=duration,
        bpm=bpm,
        beat_times=[round(t, 4) for t in beat_times],
        beats_per_second=bps,
        segments=segments,
        energy_curve=curve,
        energy_curve_hz=curve_hz,
        duration_sec=duration,
        extra={
            "sample_rate": sr,
            "tempo_raw": round(bpm, 2),
            # dynamic_range 只作展示; 判定用 peak_vs_median / peak_vs_second
            "dynamic_range": round(dynamic_range, 4),
            "peak_vs_median": estats["peak_vs_median"],
            "peak_vs_second": estats["peak_vs_second"],
            "peak_prominence": estats["peak_prominence"],
            "peak_margin": estats["peak_margin"],
            "has_climax": estats["has_climax"],
            "mean_onset_rate": round(float(np.mean([s.onset_rate for s in segments])), 3)
            if segments
            else 0.0,
        },
    )


def save_analysis(analysis: MusicAnalysis, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------------------------------------------------
# 测试用: 生成一首有明确段落结构的合成曲
# ------------------------------------------------------------------
def make_test_track(
    out_path: str | Path,
    *,
    bpm: float = 128.0,
    duration: float = 72.0,
    sr: int = SAMPLE_RATE,
) -> Path:
    """生成合成测试音轨: 安静 intro → 推进 → 爆发 drop → 收尾.

    用于在没有真实音乐时验证节拍检测与分段逻辑。
    结构 (72s @128bpm):
        0-12s   intro    : 只有柔和 pad
        12-24s  build    : 加入 hi-hat 8 分音符 + 渐强
        24-48s  drop     : 四踩 kick + bass + 亮 pad (最炸)
        48-60s  breakdown: 去掉 kick, 保留 pad
        60-72s  outro    : 渐弱
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(duration * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm          # 每拍秒数
    rng = np.random.default_rng(7)

    def add_kick(at: float, gain: float) -> None:
        """底鼓: 60Hz 指数衰减正弦 + 点击噪声."""
        dur = 0.18
        m = int(dur * sr)
        tt = np.arange(m) / sr
        env = np.exp(-tt * 28)
        tone = np.sin(2 * np.pi * (55 + 40 * np.exp(-tt * 30)) * tt)
        click = rng.normal(0, 1, m) * np.exp(-tt * 220) * 0.25
        sig = (tone + click) * env * gain
        i = int(at * sr)
        if i + m <= n:
            y[i:i + m] += sig.astype(np.float32)

    def add_hat(at: float, gain: float) -> None:
        dur = 0.05
        m = int(dur * sr)
        tt = np.arange(m) / sr
        env = np.exp(-tt * 90)
        sig = rng.normal(0, 1, m) * env * gain
        i = int(at * sr)
        if i + m <= n:
            y[i:i + m] += sig.astype(np.float32)

    def add_pad(a: float, b: float, freq: float, gain: float) -> None:
        """段落垫音: 三个谐波叠加, 缓慢起伏."""
        i0, i1 = int(a * sr), int(b * sr)
        if i1 <= i0:
            return
        tt = t[i0:i1]
        env = np.ones_like(tt)
        ramp = int(0.5 * sr)
        if len(env) > 2 * ramp:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        sig = (
            np.sin(2 * np.pi * freq * tt)
            + 0.5 * np.sin(2 * np.pi * freq * 2 * tt)
            + 0.25 * np.sin(2 * np.pi * freq * 3 * tt)
        )
        # 轻微颤音让它更像音乐
        sig *= 1.0 + 0.05 * np.sin(2 * np.pi * 0.7 * tt)
        y[i0:i1] += (sig * env * gain).astype(np.float32)

    def add_bass(a: float, b: float, freq: float, gain: float) -> None:
        i0, i1 = int(a * sr), int(b * sr)
        if i1 <= i0:
            return
        tt = t[i0:i1]
        y[i0:i1] += (np.sin(2 * np.pi * freq * tt) * gain).astype(np.float32)

    # --- intro: 柔和 pad ---
    add_pad(0, 12, 220.0, 0.10)
    # --- build: pad + 渐强 hat + 轻 kick ---
    add_pad(12, 24, 246.94, 0.13)
    k = 0.0
    while k < 24:
        if k >= 12:
            prog = (k - 12) / 12.0
            add_hat(k, 0.05 + 0.10 * prog)
            if abs((k / beat) % 1.0) < 1e-6:
                add_kick(k, 0.25 + 0.35 * prog)
        k += beat / 2   # 8 分音符
    # --- drop: 四踩 kick + bass + 亮 pad ---
    add_pad(24, 48, 293.66, 0.16)
    add_bass(24, 48, 82.41, 0.22)
    k = 24.0
    while k < 48:
        add_kick(k, 0.85)
        add_hat(k + beat / 2, 0.16)
        k += beat
    # --- breakdown: 去掉 kick ---
    add_pad(48, 60, 261.63, 0.11)
    # --- outro: 渐弱 ---
    add_pad(60, 72, 196.0, 0.07)

    # 归一化并加一点点底噪
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak * 0.9 + rng.normal(0, 0.0015, n)).astype(np.float32)

    import soundfile as sf

    sf.write(str(out_path), y, sr)
    return out_path


if __name__ == "__main__":  # pragma: no cover - 手工验证入口
    import sys

    if len(sys.argv) > 1:
        a = analyze_music(sys.argv[1])
    else:
        p = make_test_track(config.WORK_DIR / "test_track.wav")
        print(f"[生成测试音轨] {p}")
        a = analyze_music(p)

    print(f"时长 {a.duration:.1f}s | BPM {a.bpm:.1f} | 拍点 {len(a.beat_times)}")
    for s in a.segments:
        print(
            f"  [{s.start:6.1f}-{s.end:6.1f}] {s.label:<10} "
            f"arousal={s.arousal:.2f} brightness={s.brightness:.2f} "
            f"onset/s={s.onset_rate:5.2f} | {s.emotion}"
        )
