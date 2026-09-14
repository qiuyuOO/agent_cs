"""复现审查报告里的 S1 / S2 / M4 / M6 等, 确认是否属实.

只做验证, 不改文件。
"""
import sys

sys.path.insert(0, r"E:\agent_cs")
from types import SimpleNamespace

from cs2clipper import planner as P

print("=" * 76)
print("S1: 首条 out_start > 0 时, 时间轴是否整体偏移且无告警")
print("=" * 76)


class Seg:
    def __init__(self, i, start, end, arousal=0.5, label="verse"):
        self.index = i
        self.start = start
        self.end = end
        self.duration = end - start
        self.arousal = arousal
        self.rel_energy = 0.0
        self.label = label


segs = [Seg(i, i * 10.0, (i + 1) * 10.0) for i in range(4)]   # 4 x 10s = 40s


class Card:
    def __init__(self, cid, s, e, score=60.0):
        self.id, self.start_tick, self.end_tick = cid, s, e
        self.round_num, self.score = 1, score
        self.tags, self.places, self.kills = [], [], []
        self.player, self.player_side = "p", "t"


by_id = {f"c{i}": Card(f"c{i}", 1000 + i * 2000, 2000 + i * 2000) for i in range(6)}
analysis = SimpleNamespace(segments=segs, beat_times=[], extra={"has_climax": True})

# 只为 seg2 / seg3 挑素材 -> 首条落在 20s
picks = [
    {"segment": 2, "highlight_id": "c0"},
    {"segment": 2, "highlight_id": "c1"},
    {"segment": 3, "highlight_id": "c2"},
    {"segment": 3, "highlight_id": "c3"},
]
clips = P._expand(picks, analysis, by_id, music_duration=40.0, fps=30,
                  snap_beats=True, max_clips=10, pacing="balanced")
print(f"\n_expand 产出 {len(clips)} 条:")
for c in clips:
    print(f"  [{c.out_start:6.2f}-{c.out_end:6.2f}] {c.highlight_id}")

first_start = clips[0].out_start
total = clips[-1].out_end
sum_dur = sum(c.duration for c in clips)
print(f"\n  首条 out_start = {first_start:.3f}s")
print(f"  Σduration      = {sum_dur:.3f}s")
print(f"  total_duration = {total:.3f}s  (= 末条 out_end)")
print(f"  **渲染端按 Σduration 拼接 -> 视频比时间轴短 {first_start:.3f}s**")

clips2, notes = P.validate_edl(clips, by_id, 40.0, fps=30)
print(f"\n  经 validate_edl 后首条 out_start = {clips2[0].out_start:.3f}s")
print(f"  notes = {notes}")

edl = P.EDL(music_path="x", music_duration=40.0, fps=30, clips=clips2)
print(f"  coverage_note = {P.coverage_note(edl)}")

print("\n" + "=" * 76)
print("S2: LLM 返回非 dict / clips 元素非 dict")
print("=" * 76)
for raw in ('[{"clips": []}]', '{"clips": ["x"]}', '{"clips": [{"segment":0}]}', 'not json'):
    got = P._extract_json(raw)
    print(f"  _extract_json({raw[:24]!r:28}) -> {type(got).__name__} {str(got)[:40]}")

print("\n" + "=" * 76)
print("M4: _top_up_tail 补的镜头, 源区间是否短于声明时长")
print("=" * 76)
cards = [Card(f"t{i}", 1000, 1243) for i in range(4)]   # 每个仅 243 tick ≈ 3.8s
by_id2 = {c.id: c for c in cards}
base_clips = [P.EDLClip(index=0, music_segment=0, out_start=0.0, out_end=5.0,
                        highlight_id="t0", src_start_tick=1000, src_end_tick=1243,
                        speed=1.0, transition=None)]
def audit(clips, tag):
    bad = []
    for c in clips:
        src_sec = (c.src_end_tick - c.src_start_tick) / 64 / c.speed
        if src_sec + 0.05 < c.duration:
            bad.append((c.index, round(c.duration, 2), round(src_sec, 2)))
    print(f"  {tag}: {len(clips)} 条, 声明时长 > 源时长(会冻结末帧)的: {bad if bad else '无'}")
    return bad


tops, tnotes = P._top_up_tail([P.EDLClip(**vars(base_clips[0]))], by_id2, cards,
                              music_duration=40.0, fps=30, hot=False,
                              max_clips=10, hi_limit=9.0)
print(f"  补出 {len(tops)} 条, notes={tnotes}")
audit(tops, "_top_up_tail 之后(中间态)")
fixed, vnotes = P.validate_edl(tops, by_id2, 40.0, fps=30)
audit(fixed, "再 validate_edl(生产路径)")
print(f"  收敛 notes = {vnotes}")

print("\n" + "=" * 76)
print("M6: fallback_plan 是否会让同一素材出现在两处")
print("=" * 76)
segs6 = [Seg(0, 0, 20, 0.3), Seg(1, 20, 40, 0.4), Seg(2, 40, 60, 0.5), Seg(3, 60, 62, 0.9)]
an6 = SimpleNamespace(segments=segs6, beat_times=[], extra={"has_climax": True},
                      path="x", duration=62.0)
cards6 = [Card(f"h{i}", 1000 + i * 1000, 2000 + i * 1000, 50.0 + i) for i in range(5)]
edl6 = P.fallback_plan(an6, cards6, max_clips=30, music_duration=62.0, fps=30)
ids = [c.highlight_id for c in edl6.clips]
from collections import Counter
dup = {k: v for k, v in Counter(ids).items() if v > 1}
print(f"  产出 {len(ids)} 条, 素材 id: {ids}")
print(f"  重复使用的素材: {dup if dup else '无'}")
print(f"  拆分说明: {[n for n in edl6.meta.get('notes', []) if '拆' in n]}")
print(f"  重复条目的 reason 是否都标了自动化: "
      f"{[c.reason for c in edl6.clips if ids.count(c.highlight_id) > 1][:3]}")
