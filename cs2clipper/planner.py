"""Stage 3 —— LLM 编排 (音乐结构 × 素材 → EDL).

设计取向: **LLM 只做它擅长的事**。
    不擅长: 算时间轴、算 tick、保证总时长 —— 这些交给代码。
    擅长:    根据"这段音乐是什么情绪"从候选素材里挑哪些、怎么排先后。

所以流程是:
    1. 代码把音乐段落结构 + 候选素材列表 (带评分与标签) 交给 LLM
    2. LLM 输出一个**高层编排 JSON**: 每个音乐段落选哪几条素材、什么顺序
    3. 代码把高层编排**展开**成精确时间轴 (吸附节拍、计算 tick 区间、清缝隙)
    4. 代码做校验与修复, 保证渲染器拿到的一定是合法 EDL

这样即使 LLM 输出的时间数字不精确, 成片也不会错乱 —— 它只影响"选谁/排序"。
`fallback_plan()` 提供完全确定性的兜底编排, 没有 API key 时也能出片。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

from . import config
from .demo import HighlightCard, TICKRATE
from .music import MusicAnalysis, snap_to_beat

# 每个音乐段落的镜头条数上限 (由 _expand 按段落时长推导后被它夹住)
MAX_CLIPS_PER_SEGMENT = 8
# 两段之间的最小间隙 (秒) —— 避免音频被切得过于零碎
MIN_GAP = 0.0
# 转场白名单 (与 ffmpeg xfade 的 transition 名一致)
ALLOWED_TRANSITIONS = {
    "fade", "fadeblack", "fadewhite", "wipeleft", "wiperight",
    "wipeup", "wipedown", "slideleft", "slideright", "circleopen",
    "circleclose", "radial", "smoothleft", "smoothright", "pixelize",
    "dissolve", "none",
}
# 渲染器支持的 effects 字段
ALLOWED_EFFECTS = {
    "trail_len", "trail_alpha", "dot_scale",
    "show_places", "show_hud", "vignette", "zoom",
}


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------
@dataclass
class EDLClip:
    """一条精确到 tick 的剪辑决策."""

    index: int
    music_segment: int
    out_start: float          # 在成片音轨上的开始时间
    out_end: float            # 结束时间
    highlight_id: str
    # 源 tick 区间 (已裁剪到素材范围内)
    src_start_tick: int
    src_end_tick: int
    speed: float = 1.0
    transition: str | None = None
    transition_duration: float = 0.4
    effects: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.out_end - self.out_start

    @property
    def src_duration_ticks(self) -> int:
        return self.src_end_tick - self.src_start_tick

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        d["src_duration"] = round(self.src_duration_ticks / TICKRATE, 3)
        return d


@dataclass
class EDL:
    """Edit Decision List —— 渲染器的唯一输入."""

    music_path: str
    music_duration: float
    fps: int
    clips: list[EDLClip]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        return self.clips[-1].out_end if self.clips else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "music_path": self.music_path,
            "music_duration": round(self.music_duration, 3),
            "fps": self.fps,
            "total_duration": round(self.total_duration, 3),
            "clip_count": len(self.clips),
            "clips": [c.to_dict() for c in self.clips],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        clips = []
        for c in d["clips"]:
            c = dict(c)
            c.pop("duration", None)
            c.pop("src_duration", None)
            clips.append(EDLClip(**c))
        return cls(
            music_path=d["music_path"],
            music_duration=d["music_duration"],
            fps=d["fps"],
            clips=clips,
            meta=d.get("meta", {}),
        )


# ------------------------------------------------------------------
# 提示词
# ------------------------------------------------------------------
SYSTEM_PROMPT = """你是一位专业的 CS2 集锦剪辑师，负责根据音乐情绪编排视频节奏。

你的输出会被程序展开成精确时间轴，所以你只需要决定：
1. 每个音乐段落用哪几条素材（按顺序）
2. 每条素材强调什么（是否快放、用什么转场）

必须遵守的规则：
- 只能使用给定列表里的 highlight id，不能编造。
- 每条素材只能使用一次。
- **严格遵守 user_preferences 里的约束**（用户已保存的偏好，优先于下面的默认风格）：
  * max_clips 是硬上限，你排出的剪辑总数不得超过它。
  * clip_duration_range 是每种段落允许的单条镜头时长区间（秒），不要超出。
  * pacing 决定整体节奏：fast=快切碎剪 / balanced=跟随情绪 / cinematic=长镜头。
- 激烈度(arousal)越高，分配越多、越短的镜头；平静段落用少而长的镜头。
- 开场段落优先用 opening_kill / 首杀类素材；高潮段落优先用 ace / quad / triple / clutch。
- 同一回合的素材不要连续堆叠。
- 整体节奏要有起伏，不要全程一个速度。

只输出 JSON，不要任何解释或 markdown 代码块。格式：
{"clips":[{"segment":0,"highlight_id":"r3_g0_xxx","emphasis":"normal","speed":1.0,"transition":"fade","reason":"简短中文理由"}]}
emphasis 只能取 "slow" | "normal" | "fast"。
"""

#: 节奏偏好 -> 给 LLM 的自然语言说明
PACING_HINT = {
    "fast": "快切风格：镜头短、切换密，重点在高光瞬间的密度与冲击力",
    "balanced": "均衡风格：节奏跟随音乐情绪起伏，快慢结合",
    "cinematic": "电影感：镜头更长、留白更多，让每个画面有余韵",
}


def build_preference_brief(
    prefs: dict[str, Any] | None,
    segments: Sequence[Any],
    *,
    max_clips: int,
) -> dict[str, Any]:
    """把用户偏好整理成 LLM 能直接执行的约束说明.

    为什么必须进提示词: 偏好如果只在代码里事后夹取, LLM 会按自己的风格排出
    一个"超限方案", 然后被硬裁 —— 结果既不是它想排的, 也不完全符合用户偏好。
    把节奏、段数上限、单条时长区间**提前告诉它**, 它才能一次排对。

    单条时长区间直接取自 `_segment_limits` (与展开器同一来源), 所以提示词里
    写的数字就是代码实际会夹取的范围, 两边不会打架。
    """
    prefs = prefs or {}
    pacing = str(prefs.get("pacing", "balanced") or "balanced")

    # 按"该曲实际出现的段落激烈度"给出区间, 而不是罗列全部三档
    buckets: dict[str, tuple[float, float]] = {}
    for seg in segments or []:
        arousal = float(getattr(seg, "arousal", 0.5))
        rel = getattr(seg, "rel_energy", None)
        t, lo, hi = _segment_limits(arousal, rel, pacing)
        label = getattr(seg, "label", "verse")
        prev = buckets.get(label)
        buckets[label] = (min(prev[0], lo), max(prev[1], hi)) if prev else (lo, hi)

    brief: dict[str, Any] = {
        "max_clips": int(max_clips),
        "pacing": pacing,
        "pacing_meaning": PACING_HINT.get(pacing, PACING_HINT["balanced"]),
        # **向外取整**: 下界向下、上界向上。用 round() 会让提示词比实际可夹取的
        # 范围更窄 (实测 fast 模式报 1.1, 实际是 1.05), LLM 按提示词排反而会被
        # 代码判为越界。宁可报宽一点, 也不能报窄。
        "clip_duration_range": {
            label: [math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10]
            for label, (lo, hi) in sorted(buckets.items())
        },
        "notes": [
            "max_clips 是总数上限，不是每段上限",
            "单条时长必须落在对应段落标签的区间内",
        ],
    }

    def _on(key: str) -> bool:
        return bool(prefs.get(key, True))

    # 画面类偏好会影响观感, 让 LLM 的理由与之呼应 (不是硬约束, 但能提升一致性)
    visual = []
    if not _on("show_hud"):
        visual.append("不显示 HUD 与比分")
    if not _on("show_places"):
        visual.append("雷达上不标注点位名")
    if not _on("vignette"):
        visual.append("关闭四周压暗")
    if float(prefs.get("shake_px", 9.0) or 0) <= 0:
        visual.append("关闭击杀震屏")
    if float(prefs.get("fade_in", 0.18) or 0) + float(prefs.get("fade_out", 0.22) or 0) <= 0:
        visual.append("关闭段首尾淡入淡出")
    elif float(prefs.get("fade_in", 0) or 0) >= 0.6:
        visual.append("段首淡入较长，适合慢节奏开场")
    if float(prefs.get("trail_blur", 0.0) or 0) > 0:
        visual.append("开启拖尾模糊（强调速度感，适合快切段落）")
    if visual:
        brief["visual_style"] = visual

    return brief


def _build_user_prompt(
    analysis: MusicAnalysis,
    cards: Sequence[HighlightCard],
    *,
    max_clips: int,
    prefs: dict[str, Any] | None = None,
    profile: dict[str, str] | None = None,
) -> str:
    music = analysis.llm_view()
    candidates = [c.llm_view() for c in cards]
    brief = build_preference_brief(prefs, analysis.segments, max_clips=max_clips)
    payload = {
        "user_preferences": brief,
        "music": music,
        "available_highlights": candidates,
        "output_constraints": {
            "max_clips": max_clips,
            "clips_per_segment_hint": "激烈段落 2-4 条, 平静段落 1-2 条",
        },
    }
    # 用户画像: 从历史对话推断出的稳定倾向。与 user_preferences 的区别是
    # 它带不确定性, 只作为编排口味的上下文, 不是硬约束。
    if profile:
        payload["user_profile"] = profile
    return (
        "请为下面这段音乐编排一个 CS2 集锦的时间轴。\n"
        "注意 user_preferences 是用户已保存的偏好，请严格遵守；"
        "user_profile 是从历史对话推断的稳定倾向，用作口味参考。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)
    )


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 回复里抠出 JSON (容忍 ```json 包裹与前后废话).

    **必须保证返回 dict** —— 注解声明了 dict, 但 json.loads 也可能解析出
    list/str/数字。实测 `[{"clips": []}]` 会返回 list, 调用方 `data.get("clips")`
    随即抛 AttributeError 把整条出片流程打断, 与"任何环节失败都回落 fallback"
    的承诺相矛盾。这里统一兜底成空 dict。
    """
    text = (text or "").strip()
    # 去掉 markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    def _as_dict(obj: Any) -> dict[str, Any]:
        return obj if isinstance(obj, dict) else {}

    try:
        return _as_dict(json.loads(text))
    except json.JSONDecodeError:
        pass
    # 退一步: 截取第一个 { 到最后一个 }
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return _as_dict(json.loads(text[i : j + 1]))
        except json.JSONDecodeError:
            pass
    return {}


# ------------------------------------------------------------------
# 时间轴展开 (核心: 把高层编排变成精确 tick 与时间)
# ------------------------------------------------------------------
def _allocate_durations(total: float, n: int, lo: float, hi: float) -> list[float]:
    """把 `total` 秒**精确**分给 n 条镜头, 每条落在 [lo, hi] 内.

    做法: 取"均分"与 hi 的较小值作为统一长度, 再用 [lo, hi] 的余量补齐
    差额, 最后把舍入残差丢掉。返回值之和 <= total (不会超出), 且每条
    都在 [lo, hi] 内 —— 这点很重要, 否则 drop 段会出现一条 15 秒的慢镜头。

    早先的实现是给每条算一个"目标时长"再各自 clamp, 结果段落剩余时间没人
    吃, 成片里就出现十几秒的空画面。
    """
    if n <= 0:
        return []
    if total <= 0:
        return []
    # 统一长度: 均分, 但不超过 hi
    uniform = min(total / n, hi)
    uniform = max(min(uniform, hi), min(lo, total / n))
    out = [uniform] * n
    # 剩余差额逐条补, 每条最多补到 hi
    rem = total - uniform * n
    guard = 0
    while rem > 1e-9 and guard < 100000:
        guard += 1
        progressed = False
        for i in range(n):
            if rem <= 1e-9:
                break
            room = hi - out[i]
            if room <= 1e-9:
                continue
            take = min(room, rem)
            out[i] += take
            rem -= take
            progressed = True
        if not progressed:
            break
    return out


# 每条镜头的目标时长 (按段落激烈度), 用来决定"这段音乐该配几条镜头"
TARGET_DUR = {"hot": 2.8, "mid": 4.0, "calm": 5.5}

#: 节奏偏好 -> (单条时长倍率, 快/慢允许范围倍率)。
#: 这是**提示词与展开器共用的唯一来源** —— LLM 按提示词里的数字排,
#: 代码用同一套数字夹取, 两边必须一致, 否则会出现"LLM 按 A 排、代码按 B 裁"。
PACING_PROFILES: dict[str, tuple[float, float, float]] = {
    # 名称:      (时长倍率, lo 倍率, hi 倍率)
    "fast":      (0.62, 0.75, 0.80),   # 快切: 单条更短, 上限也压低
    "balanced":  (1.00, 1.00, 1.00),   # 跟随情绪 (默认)
    "cinematic": (1.60, 1.30, 1.35),   # 长镜头电影感
}


def pacing_profile(pacing: str | None) -> tuple[float, float, float]:
    """取节奏倍率, 未知值回落到 balanced."""
    return PACING_PROFILES.get(str(pacing or "balanced"), PACING_PROFILES["balanced"])


def _segment_limits(
    arousal: float,
    rel_energy: float | None = None,
    pacing: str | None = None,
) -> tuple[float, float, float]:
    """按激烈度返回 (目标单条时长, 最短, 最长).

    `rel_energy` 是该段相对全曲中位的能量 (来自 music 模块)。真实歌曲里
    arousal 的段间差距会被压缩 (归一化后可能全是 0.7~1.0), 单看绝对阈值会
    让"平静的副歌"和"真正的爆发段"用同样的节奏。用相对能量可以把它们区分开。

    `pacing` 是用户的节奏偏好 (fast/balanced/cinematic), 按倍率整体缩放单条
    时长。提示词里报给 LLM 的区间就是这里算出来的, 保证两边一致。
    """
    a = arousal
    if rel_energy is not None and rel_energy < -0.05:
        # 全曲偏轻的段落 -> 放慢, 拉长镜头
        a = min(arousal, 0.32)
    if a >= 0.66:
        t, lo, hi = TARGET_DUR["hot"], 1.4, 4.5
    elif a <= 0.33:
        t, lo, hi = TARGET_DUR["calm"], 3.0, 9.0
    else:
        t, lo, hi = TARGET_DUR["mid"], 1.8, 6.0

    mult, lo_mult, hi_mult = pacing_profile(pacing)
    return (round(t * mult, 2),
            round(lo * lo_mult, 2),
            round(hi * hi_mult, 2))


def _pick_segment_limits(seg, *, has_climax: bool,
                         pacing: str | None = None) -> tuple[float, float, float]:
    """选这一段的剪辑节奏参数.

    真实音乐的一个坑: 像《英雄主义》这种全曲能量接近平稳的歌, 段落里可能
    **没有任何一段**会被标成高激烈度。旧逻辑下所有段落都走"平缓"分支, 结果
    全片都是长镜头, 活泼的歌也被剪得很温吞。

    这里改成: 若全曲没有真正的 drop 段, 就把"相对最激烈的那几段"当作高光段
    处理 —— 让节奏跟着歌**自己的**动态走, 而不是套固定阈值。
    """
    rel = getattr(seg, "rel_energy", None)
    t, lo, hi = _segment_limits(seg.arousal, rel, pacing)
    if not has_climax and seg.arousal >= 0.85:
        # 相对高潮段: 用更快的节奏, 但不至于像真正 drop 那么碎
        hot_t, hot_lo, hot_hi = _segment_limits(1.0, rel, pacing)
        t, lo, hi = (t + hot_t) / 2, hot_lo, (hi + hot_hi) / 2
    return t, lo, hi


def _clips_for_span(span: float, arousal: float, candidates: int,
                    pacing: str | None = None) -> int:
    """给定段落时长, 算最多能放几条镜头.

    **上限由时长决定**: span / lo。否则 24 秒的 drop 段只放 4 条,
    每条会被拉到 6 秒, 慢得不能看。planner 与 _expand 必须共用这个函数,
    否则两边算出的条数不一致 -> 段落时间没人吃 -> 成片出现长空隙。
    """
    _, lo, _ = _segment_limits(arousal, None, pacing)
    by_time = max(1, int(span // max(lo, 0.1)))
    return max(0, min(by_time, candidates, MAX_CLIPS_PER_SEGMENT))


def _budget_clips(
    spans: list[float],
    arousals: list[float],
    max_total: int,
    pacing: str | None = None,
) -> tuple[list[int], bool]:
    """把全局段数上限分配到各段落, 返回 (每段条数, 是否受上限约束).

    为什么需要这个: 原先 `max_clips` 只用来削减**候选池**, 完全不约束最终段数 ——
    用户设了 4, 实际还是出 27 段 (每段仍按"铺满音乐"推导条数)。这是语义落差。

    分配策略:
        1. 先按 arousal 从高到低, 给每段保底 `_clips_for_span` 条。
           高激烈段落优先占额度 —— 全片最精彩的部分不该被砍。
        2. 还有余额就继续按 arousal 从高到低加, 每段不超过它的时长上限。
        3. 额度用尽则停止; 此时音乐尾部会没有画面, 由上层给出提示。
    """
    n = len(spans)
    if n == 0:
        return [], False
    caps = [_clips_for_span(spans[i], arousals[i], MAX_CLIPS_PER_SEGMENT, pacing)
            for i in range(n)]
    total_cap = sum(caps)
    if max_total <= 0 or max_total >= total_cap:
        return caps, False

    # 额度按**时间顺序**耗尽 (不是按 arousal)。
    # 按 arousal 排序会让"时间上靠后的高激烈段"先把额度拿走, 前面的段落反而
    # 拿不到 -> 时间轴出现**内部空隙** (实测设 4 时出了 13 段却仍有洞)。
    # 按时间顺序耗尽, 空隙就只会出现在尾部, 成片只是提前结束。
    alloc = [0] * n
    remaining = max_total
    for i in range(n):
        if remaining <= 0:
            break
        take = min(caps[i], remaining)
        alloc[i] = take
        remaining -= take
    return alloc, True


def _pick_evenly(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """从 pool 里均匀挑 n 条 (而不是只取前 n 条).

    素材池已按评分降序, 但"只取前 n 条"会让所有段落都用同一批头部素材,
    后面的高光永远没机会出现。均匀取样能覆盖整个池子。
    """
    if n >= len(pool):
        return list(pool)
    if n <= 0:
        return []
    step = len(pool) / n
    return [pool[min(int(i * step), len(pool) - 1)] for i in range(n)]


def _expand(
    picks: list[dict[str, Any]],
    analysis: MusicAnalysis,
    by_id: dict[str, HighlightCard],
    *,
    music_duration: float,
    fps: int,
    snap_beats: bool = True,
    max_clips: int | None = None,
    pacing: str | None = None,
) -> list[EDLClip]:
    """把 LLM 的 picks 展开成 EDLClip 列表.

    关键点: 时长由**音乐段落**决定, 素材通过裁剪 tick 区间去适配它。
    这样总时长永远等于音乐长度, 不会因为素材太短/太长而错位。
    """
    segs = analysis.segments
    if not segs:
        return []

    # 1. 按段落归组 picks
    per_seg: dict[int, list[dict[str, Any]]] = {}
    for p in picks:
        try:
            si = int(p.get("segment", 0))
        except (TypeError, ValueError):
            continue
        if 0 <= si < len(segs):
            hid = str(p.get("highlight_id", ""))
            if hid in by_id:
                per_seg.setdefault(si, []).append(p)

    beats = analysis.beat_times
    clips: list[EDLClip] = []
    cursor = 0.0
    idx = 0

    # 每条镜头的目标时长已抽到 _segment_limits, 保证与 fallback_plan 一致。
    # 先算全局段数分配: max_clips 是**最终段数**的硬上限, 不是候选池大小。
    _spans: list[float] = []
    _arous: list[float] = []
    for si, seg in enumerate(segs):
        if not per_seg.get(si):
            _spans.append(0.0)
            _arous.append(seg.arousal)
            continue
        s0 = max(seg.start, 0.0)
        s1 = min(seg.end, music_duration)
        _spans.append(max(s1 - s0, 0.0))
        _arous.append(seg.arousal)
    if max_clips and max_clips > 0:
        alloc, capped = _budget_clips(_spans, _arous, int(max_clips), pacing)
    else:
        alloc, capped = None, False

    for si, seg in enumerate(segs):
        pool = per_seg.get(si, [])
        if not pool:
            continue

        # 2. 段落内按"激烈度"决定镜头条数 —— 必须由**段落长度**推导,
        #    否则一段 24 秒的 drop 只配 3 条镜头, 每条会被硬拉到 8 秒,
        #    慢得不能看。这里反过来: 时长 / 最短单条时长 = 条数上限。
        has_climax = bool(analysis.extra.get("has_climax", True))
        target, lo, hi = _pick_segment_limits(seg, has_climax=has_climax,
                                              pacing=pacing)

        seg_start = max(seg.start, cursor)
        seg_end = min(seg.end, music_duration)
        avail = seg_end - seg_start
        if avail <= 0.4:
            continue

        want_n = _clips_for_span(avail, seg.arousal, MAX_CLIPS_PER_SEGMENT, pacing)
        if alloc is not None and si < len(alloc):
            want_n = min(want_n, alloc[si])
        if want_n <= 0:
            # 额度已用完: 该段落留白 (尾部可能没有画面, 由上层提示)
            continue
        # 候选不足时用手上全部 (单条会变长, 由分配器夹在上限内)
        want_n = min(want_n, len(pool))
        chosen = _pick_evenly(pool, want_n)
        k = len(chosen)

        # 2b. 精确分配: 段落时间被 k 条镜头**完整**吃掉, 不留空隙
        durs = _allocate_durations(avail, k, lo, hi)
        if not durs or max(durs) < 0.5:
            continue

        # 3. 逐条展开
        t = seg_start
        for ci, p in enumerate(chosen):
            card = by_id[p["highlight_id"]]
            emphasis = str(p.get("emphasis", "normal")).lower()
            try:
                speed = float(p.get("speed", 1.0))
            except (TypeError, ValueError):
                speed = 1.0
            speed = max(0.5, min(speed, 3.0))
            if emphasis == "fast":
                speed = max(speed, 1.25)
            elif emphasis == "slow":
                speed = min(speed, 0.85)

            dur = durs[ci] if ci < len(durs) else 0.0
            if dur <= 0.3:
                continue

            out_start = t
            out_end = t + dur
            if snap_beats and beats:
                # 起点吸附到节拍 (卡点), 但不越过段落, 且保持时长不变
                snapped = snap_to_beat(out_start, beats)
                if seg_start <= snapped < out_end - 0.5:
                    out_start = snapped
                    out_end = out_start + dur

            # 素材源区间: 按 speed 换算出需要多少 tick, 从素材开头取。
            # 若素材本身不够长, 先尽量取满, 剩余由 validate_edl 降速兜底。
            need_ticks = int(round((out_end - out_start) * speed * TICKRATE))
            avail_ticks = card.end_tick - card.start_tick
            take = min(need_ticks, avail_ticks)
            src_start = card.start_tick
            src_end = card.start_tick + take

            trans = str(p.get("transition", "fade") or "").lower()
            if trans not in ALLOWED_TRANSITIONS:
                trans = "fade"

            effects: dict[str, Any] = {}
            raw_eff = p.get("effects")
            if isinstance(raw_eff, dict):
                effects = {k2: v for k2, v in raw_eff.items() if k2 in ALLOWED_EFFECTS}
            # 高激烈段落给更长的拖尾, 视觉更"快"
            if seg.arousal >= 0.66:
                effects.setdefault("trail_len", 128)
            elif seg.arousal <= 0.33:
                effects.setdefault("trail_len", 64)

            clips.append(
                EDLClip(
                    index=idx,
                    music_segment=si,
                    out_start=round(out_start, 3),
                    out_end=round(out_end, 3),
                    highlight_id=card.id,
                    src_start_tick=int(src_start),
                    src_end_tick=int(src_end),
                    speed=round(speed, 3),
                    transition=None if idx == 0 and ci == 0 else trans,
                    transition_duration=round(min(0.4, max(dur * 0.2, 0.15)), 3),
                    effects=effects,
                    reason=str(p.get("reason", ""))[:200],
                )
            )
            idx += 1
            t = out_end

        cursor = t

    return clips


# ------------------------------------------------------------------
# 校验与修复
# ------------------------------------------------------------------
def validate_edl(
    clips: list[EDLClip],
    by_id: dict[str, HighlightCard],
    music_duration: float,
    *,
    fps: int = config.FPS,
) -> tuple[list[EDLClip], list[str]]:
    """检查并修复 EDL; 返回 (修好的 clips, 问题说明列表).

    这里做的事: 丢弃素材不存在的镜头、把源 tick 区间夹回素材范围、
    保证输出时间单调不越界、在素材不够长时自动降速、闭合段落间缝隙。
    """
    notes: list[str] = []
    fixed: list[EDLClip] = []
    cursor = 0.0
    min_dur = 2.0 / fps          # 至少两帧

    for c in clips:
        card = by_id.get(c.highlight_id)
        if card is None:
            notes.append(f"clip{c.index}: 素材 {c.highlight_id} 不存在, 已丢弃")
            continue

        # --- 源 tick 区间必须落在素材范围内且有内容 ---
        s = max(int(c.src_start_tick), card.start_tick)
        e = min(int(c.src_end_tick), card.end_tick)
        if e - s < 8:
            notes.append(f"clip{c.index}: 源区间过短 ({e - s} ticks), 已丢弃")
            continue
        if (s, e) != (c.src_start_tick, c.src_end_tick):
            notes.append(f"clip{c.index}: 源区间越界, 已裁剪到 [{s},{e}]")

        # --- 输出时间必须单调、且不超音乐长度 ---
        start = max(c.out_start, cursor)
        dur = max(c.out_end - c.out_start, min_dur)
        end = start + dur
        if end > music_duration:
            end = music_duration
            dur = end - start
            if dur < min_dur:
                notes.append(f"clip{c.index}: 超出音乐长度, 已丢弃")
                continue
            notes.append(f"clip{c.index}: 尾部超出音乐长度, 已截断")

        # --- 输出时长与源时长一致性: 按 speed 反推需要的 tick ---
        need = int(round(dur * c.speed * TICKRATE))
        if need > (e - s):
            # 源素材不够长 -> 延长取的区间 (最多到素材末尾)
            new_e = min(s + need, card.end_tick)
            if new_e - s > (e - s):
                e = new_e
            else:
                # 实在不够 -> 降速播放
                actual = (e - s) / TICKRATE
                if actual > 0:
                    new_speed = max(actual / dur, 0.3)
                    if abs(new_speed - c.speed) > 0.05:
                        notes.append(
                            f"clip{c.index}: 素材不足, speed {c.speed}→{round(new_speed,2)}"
                        )
                    c.speed = round(new_speed, 3)

        c.out_start, c.out_end = round(start, 3), round(end, 3)
        c.src_start_tick, c.src_end_tick = int(s), int(e)
        c.index = len(fixed)
        fixed.append(c)
        cursor = end

    # 去掉首条的转场 (没有前一段可转)
    if fixed:
        fixed[0].transition = None

    # 闭合缝隙: 段落之间如果有几十毫秒的空档, 拼接后会累积成可见的音画漂移。
    # 把每条镜头的**输出时长**顺延到下一段开始处, 同时按比例多取一点源素材。
    for i in range(len(fixed) - 1):
        gap = fixed[i + 1].out_start - fixed[i].out_end
        if gap <= 0.001:
            continue
        cur = fixed[i]
        extra_out = gap
        # 按当前 speed 换算出需要多取多少 tick
        extra_ticks = int(round(extra_out * cur.speed * TICKRATE))
        new_end = min(cur.src_end_tick + max(extra_ticks, 0), card_end(cur, by_id))
        cur.src_end_tick = int(new_end)
        cur.out_end = round(cur.out_end + extra_out, 3)
        if gap > 0.05:
            notes.append(f"clip{i}: 闭合 {gap:.3f}s 缝隙 (顺延前一段)")

    # ---- 时间轴锚定: 首条必须从 0 开始 ----
    # 渲染端**只按 duration 顺序拼接**, 从不读 out_start
    # (pipeline 逐条渲染 → compose 按 duration 拼接 → 音轨裁到 total_duration)。
    # 所以若首条 out_start > 0, 视频总长 = Σduration = total_duration − 首条起点,
    # 而成片音轨仍有 total_duration 那么长 —— 结果整片音画错位、后半段没有画面。
    # 更糟的是上面的缝隙闭合只保证"相邻两条无空隙", 对头部空隙无能为力,
    # 而 _top_up_tail 会把尾部补满, 于是 coverage_note 算出 0 缺口, 完全静默。
    # 触发路径: LLM 没给开头几段挑素材, 或首条被吸附到 0 附近的拍点。
    if fixed and fixed[0].out_start > 1e-6:
        off = fixed[0].out_start
        dropped = 0
        kept: list[EDLClip] = []
        for c in fixed:
            c.out_start = round(c.out_start - off, 3)
            c.out_end = round(c.out_end - off, 3)
            if c.out_end <= 1e-6:
                # 整条落在原点之前 —— 它对应的源内容本来就是"开头之前", 丢掉
                dropped += 1
                continue
            if c.out_start < 0:
                # 只切掉一部分: 按比例裁源区间, 让源时长与新的输出时长匹配
                keep = c.out_end / max(c.duration + (c.out_start * -1), 1e-6)
                span = c.src_end_tick - c.src_start_tick
                cut = int(round(span * (1.0 - keep)))
                c.src_start_tick = int(min(c.src_start_tick + max(cut, 0),
                                           c.src_end_tick - 8))
                c.out_start = 0.0
            kept.append(c)
        fixed = kept
        for i, c in enumerate(fixed):
            c.index = i
        if fixed:
            fixed[0].transition = None
        notes.append(
            f"时间轴锚定: 原首条起点 {off:.3f}s, 已整体前移 (丢弃 {dropped} 条越界镜头)"
        )

    return fixed, notes


def card_end(clip: EDLClip, by_id: dict[str, HighlightCard]) -> int:
    """取该 clip 对应素材的结束 tick (拿不到就返回原值)."""
    c = by_id.get(clip.highlight_id)
    return c.end_tick if c is not None else clip.src_end_tick


# ------------------------------------------------------------------
# 确定性兜底编排 (无 LLM 也能出片)
# ------------------------------------------------------------------
def fallback_plan(
    analysis: MusicAnalysis,
    cards: Sequence[HighlightCard],
    *,
    max_clips: int = 18,
    music_duration: float | None = None,
    fps: int = config.FPS,
    pacing: str | None = None,
) -> EDL:
    """不调用 LLM 的确定性编排.

    规则:
        * 素材按评分降序
        * 每个音乐段落按 arousal 决定条数 (高→多)
        * 高 arousal 段落优先分到高评分素材
        * 转场按段落切换: 激烈段用短转场, 平缓段用长 dissolve
    """
    dur = music_duration if music_duration is not None else analysis.duration
    segs = analysis.segments
    by_id = {c.id: c for c in cards}
    if not segs or not cards:
        return EDL(
            music_path=analysis.path,
            music_duration=dur,
            fps=fps,
            clips=[],
            meta={"planner": "fallback", "note": "没有段落或没有素材"},
        )

    # 1. 候选池: 按评分降序
    pool = sorted(cards, key=lambda c: -c.score)

    n_segs = len(segs)

    # 2. 按"回合"交错排列池子, 让相邻位置的素材尽量来自不同回合。
    #    这样一旦后面做了分段取样, 也不会整段都是同一个回合。
    buckets: dict[int, list[HighlightCard]] = {}
    for c in pool:
        buckets.setdefault(c.round_num, []).append(c)
    interleaved: list[HighlightCard] = []
    bi = 0
    while any(buckets.values()):
        keys = [k for k, v in buckets.items() if v]
        keys.sort()
        for k in keys:
            interleaved.append(buckets[k].pop(0))
        bi += 1
        if bi > 10000:
            break
    pool = interleaved

    # 3. 每个段落的候选条数 = _expand 实际会用的条数 (同一个函数算出来),
    #    这样候选永远够用。再乘一个富余系数, 让 _pick_evenly 有挑选空间
    #    (避免"候选刚好等于所需"导致只能连续用相邻素材)。
    HEADROOM = 1.6
    raw_need = [
        _clips_for_span(s.duration, s.arousal, MAX_CLIPS_PER_SEGMENT, pacing)
        for s in segs
    ]
    need = [max(1, min(int(math.ceil(n * HEADROOM)), MAX_CLIPS_PER_SEGMENT)) for n in raw_need]

    # 候选总量超过池子时, 按 arousal 从低到高削减 (平静段落少给几条)
    while sum(need) > len(pool):
        cand = [i for i, n in enumerate(need) if n > 1]
        if not cand:
            break
        i = min(cand, key=lambda k: (segs[k].arousal, -need[k]))
        need[i] -= 1

    # 4. 给每个段落分配池子里的一个"切片" (互不重叠, 保证素材不复用)。
    #    切片按**未经削减的时长需求**加权, 保证每段实际拿到的候选数 >= 它
    #    在 _expand 里会用到的条数, 否则该段会因候选不足只放 1 条镜头,
    #    剩下的音乐变成空镜头。
    assign: dict[int, list[HighlightCard]] = {i: [] for i in range(n_segs)}
    weight_total = sum(raw_need) or 1
    pos = 0.0
    for i in range(n_segs):
        share = (raw_need[i] / weight_total) * len(pool)
        start = int(round(pos))
        pos += share
        end = int(round(pos)) if i < n_segs - 1 else len(pool)
        end = max(end, start + 1) if start < len(pool) else len(pool)
        slice_ = pool[start:end]
        if not slice_:
            slice_ = pool[max(start - 1, 0) :][:1]
        want = min(max(need[i], raw_need[i]), len(slice_))
        assign[i] = _pick_evenly(slice_, want) if slice_ and want > 0 else []

    # 5. 转成 picks 交给 _expand
    picks: list[dict[str, Any]] = []
    for i in range(n_segs):
        seg = segs[i]
        for card in assign[i]:
            if seg.arousal >= 0.66 and ({"ace", "quad", "triple", "multi"} & set(card.tags)):
                emph = "fast"
            elif seg.arousal <= 0.33:
                emph = "slow"
            else:
                emph = "normal"
            picks.append(
                {
                    "segment": i,
                    "highlight_id": card.id,
                    "emphasis": emph,
                    "speed": 1.0,
                    "transition": "dissolve" if seg.arousal <= 0.4 else "fade",
                    "reason": f"{seg.label} 段取评分 {card.score:.0f} 的素材",
                }
            )

    clips = _expand(
        picks, analysis, by_id,
        music_duration=dur, fps=fps, snap_beats=True, max_clips=max_clips,
        pacing=pacing,
    )

    # 与 LLM 路径同序: 先闭合缝隙 -> 再拆超长 -> 再收敛
    clips, notes = validate_edl(clips, by_id, dur, fps=fps)

    used_ids = {c.highlight_id for c in clips}
    spare_global = [c for c in sorted(cards, key=lambda x: -x.score)
                    if c.id not in used_ids]
    clips, split_notes = _split_overlong_clips(
        clips, analysis.segments, by_id, spare_by_seg={}, spare_global=spare_global,
        max_clips=max_clips, pacing=pacing,
    )
    notes.extend(split_notes)
    clips, vnotes = validate_edl(clips, by_id, dur, fps=fps)
    notes.extend(vnotes)

    # 确定性编排也可能因为素材池太小而铺不满, 同样补齐
    hot = bool(segs) and segs[-1].arousal >= 0.66
    last_seg = analysis.segments[-1] if analysis.segments else None
    hi_limit = _segment_limits(
        last_seg.arousal if last_seg else 0.5,
        getattr(last_seg, 'rel_energy', None) if last_seg else None,
        pacing,
    )[2]
    clips, top_notes = _top_up_tail(
        clips, by_id, cards, music_duration=dur, fps=fps, hot=hot,
        max_clips=max_clips, hi_limit=hi_limit,
    )
    notes.extend(top_notes)
    # 补齐镜头是按"剩余尾巴"倒推声明时长的, 素材不够长时同样可能声明过长的
    # 输出时长 -> 再收敛一次, 保证 src 区间一定撑得住声明的时长。
    clips, top_vnotes = validate_edl(clips, by_id, dur, fps=fps)
    notes.extend(top_vnotes)
    edl = EDL(
        music_path=analysis.path,
        music_duration=dur,
        fps=fps,
        clips=clips,
        meta={
            "planner": "fallback",
            "notes": notes,
            "candidates_per_segment": {str(k): len(v) for k, v in assign.items()},
            "used_highlights": len({c.highlight_id for c in clips}),
        },
    )
    # 段数上限设小、或素材池太小时, 音乐尾部可能没画面 —— 必须明确报出来,
    # 否则用户只会看到"成片比歌短"却不知道原因。
    cov = coverage_note(edl)
    if cov:
        edl.meta["coverage_warning"] = cov
    return edl


# ------------------------------------------------------------------
# LLM 编排 (失败自动回落)
# ------------------------------------------------------------------
def llm_plan(
    analysis: MusicAnalysis,
    cards: Sequence[HighlightCard],
    *,
    max_clips: int = 18,
    music_duration: float | None = None,
    fps: int = config.FPS,
    temperature: float = 0.4,
    verbose: bool = False,
    prefs: dict[str, Any] | None = None,
    profile: dict[str, str] | None = None,
) -> EDL:
    """用 LLM 做编排; 任何环节失败都回落到 fallback_plan.

    `prefs` 是合并后的用户偏好: 一部分进提示词 (节奏/段数上限/单条时长区间),
    一部分传给展开器 (同一套时长区间), 保证 LLM 与代码口径一致。
    `profile` 是从历史对话推断的用户画像 (带不确定性), 只进提示词当口味参考。
    """
    dur = music_duration if music_duration is not None else analysis.duration
    by_id = {c.id: c for c in cards}
    prefs = prefs or {}
    pacing = str(prefs.get("pacing", "balanced") or "balanced")

    if not cards or not analysis.segments:
        return fallback_plan(analysis, cards, max_clips=max_clips,
                             music_duration=dur, fps=fps, pacing=pacing)

    raw_text = ""
    try:
        import model as model_mod

        llm = model_mod.build_model(temperature=temperature)
        prompt = _build_user_prompt(analysis, cards, max_clips=max_clips,
                                    prefs=prefs, profile=profile)
        if verbose:
            brief = build_preference_brief(prefs, analysis.segments, max_clips=max_clips)
            print(f"[planner] prompt 长度 {len(prompt)} 字符 | "
                  f"节奏={brief['pacing']} 段数上限={brief['max_clips']} "
                  f"时长区间={brief['clip_duration_range']}")
        resp = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        raw_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        if verbose:
            print(f"[planner] LLM 调用失败, 回落 fallback: {type(exc).__name__}: {exc}")
        edl = fallback_plan(analysis, cards, max_clips=max_clips,
                            music_duration=dur, fps=fps, pacing=pacing)
        edl.meta["planner"] = "fallback(llm_error)"
        edl.meta["llm_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return edl

    data = _extract_json(raw_text)
    picks = data.get("clips")
    if not isinstance(picks, list) or not picks:
        if verbose:
            print("[planner] LLM 输出无法解析, 回落 fallback")
        edl = fallback_plan(analysis, cards, max_clips=max_clips,
                            music_duration=dur, fps=fps, pacing=pacing)
        edl.meta["planner"] = "fallback(bad_json)"
        edl.meta["llm_raw"] = raw_text[:1000]
        return edl

    # 去重: 同一素材只用一次 (LLM 常犯)
    # 同时防止元素不是 dict (`{"clips": ["x"]}` 这类退化输出) —— 否则
    # `p.get(...)` 抛 AttributeError 会把整条出片流程打断。
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    dropped = 0
    for p in picks:
        if not isinstance(p, dict):
            dropped += 1
            continue
        hid = str(p.get("highlight_id", ""))
        if hid in seen or hid not in by_id:
            dropped += 1
            continue
        seen.add(hid)
        clean.append(p)

    if not clean:
        if verbose:
            print(f"[planner] LLM 的 {len(picks)} 条编排全部无效, 回落 fallback")
        edl = fallback_plan(analysis, cards, max_clips=max_clips,
                            music_duration=dur, fps=fps, pacing=pacing)
        edl.meta["planner"] = "fallback(no_valid_picks)"
        edl.meta["llm_raw"] = raw_text[:1000]
        return edl

    clips = _expand(clean, analysis, by_id, music_duration=dur, fps=fps,
                    snap_beats=True, max_clips=max_clips, pacing=pacing)

    # 顺序很重要: 先 validate (它会把段落间的缝隙顺延给前一条, 因而制造出
    # 新的超长镜头), 再 split 把超长的切开, 最后再 validate 一次收敛。
    # 反过来的话, 缝隙闭合会把刚拆好的镜头又拉长, 拆分等于白做。
    clips, notes = validate_edl(clips, by_id, dur, fps=fps)

    spare_global = [c for c in sorted(cards, key=lambda x: -x.score) if c.id not in seen]
    clips, split_notes = _split_overlong_clips(
        clips, analysis.segments, by_id, spare_by_seg={}, spare_global=spare_global,
        max_clips=max_clips, pacing=pacing,
    )
    notes.extend(split_notes)
    clips, vnotes = validate_edl(clips, by_id, dur, fps=fps)
    notes.extend(vnotes)

    if not clips:
        if verbose:
            print("[planner] LLM 编排展开后为空, 回落 fallback")
        edl = fallback_plan(analysis, cards, max_clips=max_clips,
                            music_duration=dur, fps=fps, pacing=pacing)
        edl.meta["planner"] = "fallback(empty_expand)"
        return edl

    # LLM 常只排到音乐的一部分 -> 用剩余素材补齐结尾
    hot = bool(analysis.segments) and analysis.segments[-1].arousal >= 0.66
    last_seg = analysis.segments[-1] if analysis.segments else None
    hi_limit = _segment_limits(
        last_seg.arousal if last_seg else 0.5,
        getattr(last_seg, 'rel_energy', None) if last_seg else None,
        pacing,
    )[2]
    clips, top_notes = _top_up_tail(
        clips, by_id, cards, music_duration=dur, fps=fps, hot=hot,
        max_clips=max_clips, hi_limit=hi_limit,
    )
    notes.extend(top_notes)
    # 同 fallback: 补齐镜头后必须再收敛一次时长与源区间
    clips, top_vnotes = validate_edl(clips, by_id, dur, fps=fps)
    notes.extend(top_vnotes)

    edl = EDL(
        music_path=analysis.path,
        music_duration=dur,
        fps=fps,
        clips=clips,
        meta={
            "planner": "llm",
            "model_picks": len(picks),
            "used": len(clean),
            "dropped_dupes_or_unknown": dropped,
            "notes": notes,
            "llm_excerpt": raw_text[:600],
        },
    )
    cov = coverage_note(edl)
    if cov:
        edl.meta["coverage_warning"] = cov
        if verbose:
            print(f"[planner] 注意: {cov}")
    return edl


def _split_overlong_clips(
    clips: list[EDLClip],
    segs: list,
    by_id: dict[str, HighlightCard],
    spare_by_seg: dict[int, list[HighlightCard]] | None = None,
    spare_global: list[HighlightCard] | None = None,
    max_clips: int | None = None,
    pacing: str | None = None,
) -> tuple[list[EDLClip], list[str]]:
    """把超过该段落时长上限的镜头拆成多条.

    返回 (新的镜头列表, 说明). **注意必须用返回值替换原列表** —— 早先这里只
    返回说明, 拆分结果被丢掉, 调用方写成 `notes = _split(...)` 就等于什么都没做
    (节奏倒挂的 bug 因此一直没修掉)。

    为什么需要: `_expand` 优先尊重音乐段落时长, 当 LLM 给某个高激烈段落只
    排了 1~2 条镜头时 (很常见), 分配器只能把那 17.5 秒摊成两条 8 秒镜头 ——
    结果**最激烈的段落反而镜头最长**, 节奏与音乐完全相反 (实测 LLM 路径高激烈
    段平均 10.41s, 其余 7.66s, 比值 1.36 —— 本该 < 1)。

    素材来源依次为: 该段富余素材 -> 全局富余素材 -> 同一素材的不同源片段。
    第三档保证"素材不够也能拆快", 否则素材池小时拆分形同虚设。
    """
    notes: list[str] = []
    out: list[EDLClip] = []
    used: set[str] = {c.highlight_id for c in clips}
    spare_by_seg = spare_by_seg or {}
    pool_global = list(spare_global or [])

    def take_spare(si: int) -> HighlightCard | None:
        """先取本段富余, 再取全局富余."""
        for src in (spare_by_seg.get(si, []), pool_global):
            while src:
                card = src.pop(0)
                if card.id not in used:
                    used.add(card.id)
                    return card
        return None

    # 按段落归组, 保持原有时间顺序
    order: list[int] = []
    grouped: dict[int, list[EDLClip]] = {}
    for c in clips:
        if c.music_segment not in grouped:
            order.append(c.music_segment)
            grouped[c.music_segment] = []
        grouped[c.music_segment].append(c)

    # 已处理/未处理的段数 (用于判断拆分后是否还装得下全局上限)
    total_input = len(clips)
    processed = 0

    for si in order:
        seg = segs[si] if 0 <= si < len(segs) else None
        if seg is None:
            out.extend(grouped[si])
            processed += len(grouped[si])
            continue
        _, lo, hi = _segment_limits(seg.arousal, getattr(seg, "rel_energy", None),
                                    pacing)

        for c in grouped[si]:
            processed += 1
            if c.duration <= hi * 1.15:
                out.append(c)
                continue

            parts = max(2, int(math.ceil(c.duration / hi)))
            # 拆分会让总段数变多, 必须尊重全局上限 `max_clips`:
            # 拆分后总数 = 已产出 + parts + 剩余未处理。超了就少拆几条,
            # 实在装不下就保持原样 (宁可镜头长一点, 也不超出用户设的段数)。
            if max_clips and max_clips > 0:
                remaining = total_input - processed
                allowed = max_clips - len(out) - remaining
                if allowed < parts:
                    parts = allowed
                if parts < 2:
                    out.append(c)
                    continue
            part_dur = c.duration / parts
            src_total = c.src_end_tick - c.src_start_tick
            src_step = max(src_total // parts, 8)
            swapped = 0

            for k in range(parts):
                sub = EDLClip(
                    index=0,                       # 稍后重排
                    music_segment=si,
                    out_start=round(c.out_start + k * part_dur, 3),
                    out_end=round(c.out_start + (k + 1) * part_dur, 3),
                    highlight_id=c.highlight_id,
                    src_start_tick=0,
                    src_end_tick=0,
                    speed=c.speed,
                    transition=c.transition if k == 0 else "fade",
                    transition_duration=c.transition_duration,
                    effects=dict(c.effects),
                    reason=c.reason,
                )
                card = take_spare(si)
                need = int(round(part_dur * sub.speed * TICKRATE))
                if card is not None:
                    # 换一条不同素材, 真正做出快切感
                    sub.highlight_id = card.id
                    sub.src_start_tick = card.start_tick
                    sub.src_end_tick = min(card.start_tick + max(need, 8), card.end_tick)
                    swapped += 1
                else:
                    # 素材用尽: 同一素材取不同的源片段
                    s0 = c.src_start_tick + k * src_step
                    sub.src_start_tick = int(s0)
                    sub.src_end_tick = int(min(s0 + src_step, c.src_end_tick))
                    if sub.src_end_tick - sub.src_start_tick < 8:
                        sub.src_start_tick = c.src_start_tick
                        sub.src_end_tick = c.src_end_tick
                if k > 0:
                    sub.reason = (c.reason + " ·自动拆分为快切").strip(" ·")
                out.append(sub)
            notes.append(
                f"seg{si}: 镜头 {c.duration:.2f}s 超出该段上限 {hi:.1f}s, "
                f"拆成 {parts} 条 (换素材 {swapped})"
            )

    out.sort(key=lambda x: x.out_start)
    for i, c in enumerate(out):
        c.index = i
    return out, notes


def _top_up_tail(
    clips: list[EDLClip],
    by_id: dict[str, HighlightCard],
    cards: Sequence[HighlightCard],
    *,
    music_duration: float,
    fps: int,
    hot: bool = False,
    max_clips: int | None = None,
    hi_limit: float = 9.0,
) -> tuple[list[EDLClip], list[str]]:
    """用没用过的素材补齐结尾空白.

    LLM 经常只排到音乐的三分之二就收工 (例如 72 秒的歌只排 58 秒), 结果最后
    十几秒音乐没有画面。这里自动从**未被使用**的素材里继续排, 直到铺满或素材
    用尽。这是修复"覆盖不足"的关键一步 —— 只警告不修的话成片照样是断的。

    `max_clips` 是最终段数的硬上限: 已经用满就不再补。

    单条时长必须同时受**素材本身长度**与 `hi_limit` 约束 —— 否则会声明一个
    源素材填不满的时长, 渲染时帧数不够, 由 ffmpeg 的 tpad 克隆末帧补齐,
    成片里就是一段**冻结画面**。实测最初版本的 4 条补充镜头全部中招
    (声明 4.5~9.0s, 源只有 3.8s)。触发条件是素材很短: 单杀素材
    (PRE_PAD 1.6s + POST_PAD 2.2s ≈ 3.8s) 比默认 target=4.5s 还短。
    """
    notes: list[str] = []
    if max_clips and max_clips > 0 and len(clips) >= max_clips:
        return clips, notes
    used = {c.highlight_id for c in clips}
    spare = [c for c in cards if c.id not in used]
    if not spare:
        return clips, notes

    tail = music_duration - (clips[-1].out_end if clips else 0.0)
    if tail <= 0.3:
        return clips, notes

    # 目标单条时长: 默认 4 秒; 高激烈段落用更短的镜头补
    target = 3.0 if hot else 4.5
    added = 0
    frozen = 0
    for card in spare:
        if tail <= 0.3:
            break
        if max_clips and max_clips > 0 and len(clips) >= max_clips:
            break
        speed = 1.15 if hot else 1.0
        avail = card.end_tick - card.start_tick
        # 素材能撑多长 (秒): 这是声明时长的硬上限之一
        cap_sec = avail / TICKRATE / max(speed, 1e-6)
        dur = min(target, tail, hi_limit, cap_sec)
        if dur < 0.8:
            continue
        start = music_duration - tail
        end = start + dur
        ticks = int(round(dur * speed * TICKRATE))
        take = min(ticks, avail)
        if take < avail and take / TICKRATE / max(speed, 1e-6) + 0.05 < dur:
            frozen += 1
        clips.append(
            EDLClip(
                index=len(clips),
                music_segment=clips[-1].music_segment if clips else 0,
                out_start=round(start, 3),
                out_end=round(end, 3),
                highlight_id=card.id,
                src_start_tick=int(card.start_tick),
                src_end_tick=int(card.start_tick + take),
                speed=round(speed, 3),
                transition="fade",
                transition_duration=0.3,
                effects={},
                reason="自动补齐结尾 (原始编排未铺满音乐)",
            )
        )
        tail -= dur
        added += 1

    # 只剩小尾巴时, 直接让最后一条镜头吃掉它, 避免留个 1 秒的尴尬空档。
    # 但**不能让单条镜头无限变长** —— 那会把"避开内部空隙"变成"超长镜头"。
    # 实测 max_clips=6 时出现过一条 111 秒的镜头, 完全没法看。
    # 上限三重: hi_limit、剩余尾巴、以及**源素材还能撑多久** (否则末帧冻结)。
    if clips and tail > 0:
        last = clips[-1]
        card = by_id.get(last.highlight_id)
        src_room_ticks = (card.end_tick - last.src_end_tick) if card else 0
        src_room_sec = src_room_ticks / TICKRATE / max(last.speed, 1e-6)
        room = max(min(hi_limit - last.duration, src_room_sec), 0.0)
        absorb = min(tail, room)
        if absorb > 0.05:
            extra_ticks = int(round(absorb * last.speed * TICKRATE))
            cap = card.end_tick if card else last.src_end_tick
            last.src_end_tick = int(min(last.src_end_tick + extra_ticks, cap))
            last.out_end = round(last.out_end + absorb, 3)
            tail -= absorb

    if added:
        notes.append(f"结尾空白已用 {added} 条未使用素材补齐")
    if frozen:
        notes.append(
            f"其中 {frozen} 条受素材长度限制 (源不足以填满声明时长, 末帧会静止)"
        )
    if tail > 0.5:
        notes.append(
            f"结尾仍有 {tail:.1f}s 没有画面: 段数上限已用满, 且单条镜头不允许"
            f"超过 {hi_limit:.1f}s"
        )
    return clips, notes


def coverage_note(edl: EDL) -> str | None:
    """检查 EDL 是否铺满了整首音乐; 没铺满时返回说明.

    LLM 有时会给出比音乐短的编排 (例如 72 秒的歌只排到 67 秒)。成片本身
    没问题 (就是提前结束), 但用户会看到音乐被截断, 所以要明确提示。
    """
    total = edl.total_duration
    dur = edl.music_duration
    if dur <= 0:
        return None
    gap = dur - total
    if gap <= 0.5:
        return None
    pct = gap / dur * 100
    return (
        f"编排只覆盖到 {total:.1f}s / 音乐 {dur:.1f}s, 结尾 {gap:.1f}s "
        f"({pct:.0f}%) 没有画面。可调大 max_clips 或让 LLM 多排几条。"
    )


def plan_edit(
    analysis: MusicAnalysis,
    cards: Sequence[HighlightCard],
    *,
    use_llm: bool = True,
    max_clips: int = 18,
    music_duration: float | None = None,
    fps: int = config.FPS,
    verbose: bool = False,
    prefs: dict[str, Any] | None = None,
    profile: dict[str, str] | None = None,
) -> EDL:
    """对外统一入口."""
    prefs = prefs or {}
    pacing = str(prefs.get("pacing", "balanced") or "balanced")
    if use_llm:
        return llm_plan(
            analysis, cards, max_clips=max_clips,
            music_duration=music_duration, fps=fps, verbose=verbose,
            prefs=prefs, profile=profile,
        )
    return fallback_plan(
        analysis, cards, max_clips=max_clips, music_duration=music_duration,
        fps=fps, pacing=pacing,
    )


if __name__ == "__main__":  # pragma: no cover
    from . import demo as D
    from . import music as M

    track = config.WORK_DIR / "test_track.wav"
    if not track.is_file():
        M.make_test_track(track)
    a = M.analyze_music(track)
    res = D.analyze_demo(str(config.DEFAULT_DEMO), max_cards=30)
    cards = D.cards_from_dicts(res["highlights"])

    edl = plan_edit(a, cards, use_llm=False, max_clips=10, verbose=True)
    print(f"段落 {len(a.segments)} | 候选素材 {len(cards)} | 剪辑 {len(edl.clips)} | 总时长 {edl.total_duration:.1f}s")
    for c in edl.clips:
        print(
            f"  seg{c.music_segment} [{c.out_start:6.2f}-{c.out_end:6.2f}] "
            f"{c.duration:5.2f}s {c.highlight_id[:28]:<28} "
            f"src[{c.src_start_tick},{c.src_end_tick}] x{c.speed} {c.transition}"
        )
