"""用真实歌曲核验 music.py 的修复 (S1 dynamic_range / M1 curve_hz / M3 bps / M4 bpm_local / M2 曲线首尾)。

三首歌与审查报告里用的一致: 英雄主义 / 陪我过个冬DJ / 苦海无涯。
"""
import sys

sys.path.insert(0, r"E:\agent_cs")

import numpy as np

from cs2clipper import music as M

SONGS = [
    r"D:\CloudMusic\在虚无中永存 - 英雄主义.mp3",
    r"D:\CloudMusic\电台节目\chen陌懿 - 陪我过个冬DJ【3D环绕版】.mp3",
    r"D:\CloudMusic\电台节目\饺子WTF - 苦海无涯.mp3",
]


def _raw_segment_rms(path):
    """复算段落级原始 RMS 均值 (与 analyze_music 内的口径一致).

    只为取证: 判定用的是归一化**之前**的段能量, 光看 MusicAnalysis 的字段
    看不出来, 所以这里按同样的分段边界重算一遍。
    """
    import librosa

    a = M.analyze_music(path)
    y, sr = librosa.load(str(path), sr=M.SAMPLE_RATE, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=M.HOP_LENGTH)[0]
    return np.array([
        float(rms[int(s.start * M.FRAMES_PER_SEC):int(s.end * M.FRAMES_PER_SEC)].mean())
        for s in a.segments
    ])

for p in SONGS:
    a = M.analyze_music(p)
    e = a.extra
    print("=" * 78)
    print(f"{p.split(chr(92))[-1]}  {a.duration:.1f}s  bpm {a.bpm:.1f}")
    print(f"  has_climax={e['has_climax']}  dynamic_range={e['dynamic_range']}"
          f"  peak/median={e['peak_vs_median']}  peak/2nd={e['peak_vs_second']}"
          f"  prominence={e['peak_prominence']}")
    print(f"  旧判据 (max-min)/median >= 0.35 -> {e['dynamic_range'] >= 0.35}"
          f"   (旧判据若为 True 而 has_climax 为 False, 正是本次修掉的缺陷)")
    labels = [(s.index, s.label, round(s.start, 1), round(s.end, 1)) for s in a.segments]
    print(f"  段落标签: {[l for _, l, _, _ in labels]}")
    drops = [(i, st, en) for i, l, st, en in labels if l == "drop"]
    print(f"  drop 段: {drops if drops else '无 (平稳曲不再凭空造 drop)'}")
    bps = a.beats_per_second * 60
    print(f"  beats_per_second {a.beats_per_second:.3f} -> 隐含 {bps:.1f} BPM"
          f"  (bpm {a.bpm:.1f}, 偏差 {abs(bps - a.bpm):.1f})")
    print(f"  curve_hz {a.energy_curve_hz:.4f}  {len(a.energy_curve)} 点 -> "
          f"反推时长 {len(a.energy_curve) / a.energy_curve_hz:.1f}s vs 音频 {a.duration:.1f}s")
    lo = min(a.segments, key=lambda s: s.bpm_local)
    print(f"  段内 bpm_local 范围 "
          f"{min(s.bpm_local for s in a.segments):.1f}~{max(s.bpm_local for s in a.segments):.1f}"
          f"  (最低段 idx={lo.index} label={lo.label})")
    print(f"  曲线首/末 {a.energy_curve[0]:.3f}/{a.energy_curve[-1]:.3f}  "
          f"P95 {float(np.percentile(a.energy_curve, 95)):.3f}")
    print(f"  段末是否超出时长: "
          f"{[s.index for s in a.segments if s.end > a.duration + 1e-9] or '无'}")
    # 判定用的是**原始**段能量 (归一化前的 RMS 均值), 这里直接展示它
    st = M.energy_stats(_raw_segment_rms(p))
    print(f"  原始段 RMS 相对中位: "
          f"{[round(x, 3) for x in sorted(_raw_segment_rms(p) / st['energy_median'], reverse=True)]}")
