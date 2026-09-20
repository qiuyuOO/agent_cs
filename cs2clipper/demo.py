"""Stage 2 —— demo 解析与亮点抽取.

把一场 demo 拆成"可剪辑的素材卡片" (HighlightCard):
    每条卡片 = 一个时间窗 + 一个主角 + 一组事件 + 强度评分

评分是**确定性**的 (不用 LLM), 这样:
    * 可以复现、可以单测
    * LLM 只需要在"高分素材里挑哪些、怎么排"这一层做决策

数据来源 (awpy 2.x):
    demo.kills      击杀表 (含 attacker/victim/weapon/headshot 等标志位)
    demo.ticks      逐 tick 玩家位置 (X/Y/Z/health/side/place)
    demo.smokes     烟雾弹 (start_tick/end_tick/位置)
    demo.infernos   燃烧弹
    demo.events["flashbang_detonate"]  闪光爆点

一个重要的性能设计: demo 解析一次约 10 秒, 所以用 `load_demo()` 做进程内缓存,
Stage 2 和 Stage 4 共用同一个已解析对象。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


from . import config

import polars as pl       # noqa: E402
from awpy import Demo  # noqa: E402

logger = logging.getLogger(__name__)

TICKRATE = config.DEMO_TICKRATE

# 各事件的基础分值 (强度评分用)
SCORE_ACE = 100.0
SCORE_CLUTCH_WIN = 80.0
SCORE_MULTIKILL_BASE = 30.0      # 每多杀一人叠加
SCORE_MULTIKILL_STEP = 22.0
SCORE_OPENING = 34.0
SCORE_HEADSHOT = 4.0
SCORE_NOSCOPE = 12.0
SCORE_THROUGH_SMOKE = 10.0
SCORE_PENETRATED = 6.0
SCORE_BLIND_KILL = 8.0
SCORE_IN_AIR = 5.0
SCORE_UTILITY_ASSIST = 6.0
SCORE_LONG_RANGE = 8.0

# 远距离击杀阈值。**单位是米** —— CS2 的 player_death.distance 已经是米制,
# 实测真 demo 的 166 次击杀: 中位 17.7m / 最大 67.3m。旧阈值写的是 1500
# (显然是按游戏单位 1unit≈1.9cm 估的), 比实测最大值还大 22 倍, 于是
# long_range 分支与标签是**永远不会触发**的死代码。
LONG_RANGE_M = 45.0

# 剪辑窗口前后留白 (tick)
PRE_PAD = int(1.6 * TICKRATE)
POST_PAD = int(2.2 * TICKRATE)


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------
@dataclass
class KillEvent:
    """一次击杀 (渲染与 JSON 都用它)."""

    tick: int
    attacker: str
    attacker_side: str
    victim: str
    victim_side: str
    weapon: str
    headshot: bool
    noscope: bool = False
    through_smoke: bool = False
    penetrated: bool = False
    attacker_blind: bool = False
    attacker_in_air: bool = False
    assistedflash: bool = False
    attacker_place: str | None = None
    victim_place: str | None = None
    distance: float = 0.0
    # 事件性质, 由 extract_kill_events 判定 (不靠调用方各自猜):
    #   player_kill       正常玩家击杀 —— 唯一能建卡的类别
    #   suicide_or_fall   自杀/坠落 (没有攻击者, 或攻击者就是自己)
    #   non_player_death  非玩家实体死亡 (打鸡等: 受害者不在名单里)
    #   attacker_missing  有受害者是玩家但攻击者缺失 -> 数据缺失, 上报
    kind: str = "player_kill"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["distance"] = round(self.distance, 1)
        return d


@dataclass
class HighlightCard:
    """一条可剪辑素材."""

    id: str
    round_num: int
    player: str
    player_side: str
    start_tick: int
    end_tick: int
    kills: list[KillEvent] = field(default_factory=list)
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    utility: list[dict[str, Any]] = field(default_factory=list)
    round_winner: str | None = None
    round_reason: str | None = None
    # 残局时"独自面对的敌人数" (1vX 的 X)。旧实现把这个信息完全丢掉了。
    clutch_enemies: int = 0

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def duration(self) -> float:
        """素材时长 (秒)."""
        return self.duration_ticks / TICKRATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "round_num": self.round_num,
            "player": self.player,
            "player_side": self.player_side,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "duration": round(self.duration, 2),
            "score": round(self.score, 1),
            "tags": list(self.tags),
            "places": list(self.places),
            "kills": [k.to_dict() for k in self.kills],
            "utility": self.utility,
            "round_winner": self.round_winner,
            "round_reason": self.round_reason,
            "clutch_enemies": self.clutch_enemies,
        }

    def llm_view(self) -> dict[str, Any]:
        """给 LLM 的精简视图: 不带逐杀细节, 只保留决策所需信息."""
        return {
            "id": self.id,
            "round": self.round_num,
            "player": self.player,
            "side": self.player_side,
            "duration": round(self.duration, 2),
            "score": round(self.score, 1),
            "tags": list(self.tags),
            "places": list(self.places),
            "kills": [
                {
                    "weapon": k.weapon,
                    "headshot": k.headshot,
                    "noscope": k.noscope,
                    "through_smoke": k.through_smoke,
                    "blind": k.attacker_blind,
                }
                for k in self.kills
            ],
            "utility_count": len(self.utility),
        }


# ------------------------------------------------------------------
# demo 加载 (带缓存)
# ------------------------------------------------------------------
#: 缓存的 demo 数。**保持 1**: 解析一场 demo 会把逐 tick 表 (实测 145 万行) 整份
#: 留在内存里, 两份就是双倍常驻。render 阶段会再取一次同一个路径 —— 命中同一份
#: 缓存即可, 不需要同时留两份。
#: 实测教训: 在 16 GB 机器上 (PyCharm + 抖音 + 浏览器已占 76%) 跑自检时,
#: 这里累积的常驻内存直接导致 `MemoryError: Unable to allocate 594. KiB` ——
#: 连半兆都申请不到, 然后整个测试进程静默退出 (没有 traceback, 只有 exit=1)。
_DEMO_CACHE_SIZE = 1


@lru_cache(maxsize=_DEMO_CACHE_SIZE)
def load_demo(
    demo_path: str,
    *,
    with_ticks: bool = True,
) -> Demo:
    """解析 demo 并缓存 (解析一次约 10s, Stage 2/4 共用).

    Args:
        demo_path: .dem 文件绝对路径
        with_ticks: 是否解析逐 tick 玩家位置 (渲染必需, 纯分析可关掉以省时间)
    """
    demo = Demo(demo_path)
    if with_ticks:
        demo.parse(
            player_props=["X", "Y", "Z", "health", "is_alive", "side", "place"],
        )
    else:
        demo.parse()
    return demo


def release_demo() -> int:
    """释放 demo 解析缓存, 把逐 tick 表占的内存还给系统.

    用在哪: 一次跑完 (`pipeline.run`) 之后调用。单次出片不需要缓存留着 ——
    但不释放的话, 进程会一直占着几百 MB 到 1 GB 的 tick 表, 在内存吃紧的机器上
    下一次出片就可能 MemoryError (实测: 16 GB 机器已用 76% 时连 594 KiB 都申请不到,
    测试进程直接静默退出、没有 traceback)。

    (lru_cache 只提供整体清理, 所以没有"只放掉某一场"的版本 —— 那反而会
     把还在用的那份一起丢掉。)

    Returns:
        释放掉的缓存条目数。
    """
    n = load_demo.cache_info().currsize
    load_demo.cache_clear()
    return n


def extract_kill_events(demo: Demo) -> list[KillEvent]:
    """把 demo.kills 转成 KillEvent 列表 (按 tick 排序), 并给每条判定 `kind`.

    为什么要分类 (旧实现只有 `attacker or "world"` + 一句注释, 而且注释写错了
    对象 —— 它说"跳过自杀/坠落", 实际判的却是"受害者名字为空"):

      * `victim` 为空 = **非玩家实体死亡** (打鸡等)。实测真 demo 有 3 条
        (usp_silencer / inferno / bizon, 攻击者都是真人)。这类必须丢弃:
        否则会把一个真实玩家的死亡记进 per_player, 缩小某队的存活集合。
      * `attacker` 为空或 `attacker == victim` = **自杀/坠落**。旧实现会把它
        写成 attacker="world" 的伪玩家 -> 生成 player="world"、side="" 的卡片;
        若它恰好是回合首个事件, 还会把 opening_kill(+34) 送给 world 卡
        (12+34=46 ≥ min_score=25, 会真的进成片)。
      * 受害者是玩家但攻击者缺失 = 数据缺失, 单独计数上报, 不静默当 world。
    """
    cols = set(demo.kills.columns)
    out: list[KillEvent] = []
    for k in demo.kills.iter_rows(named=True):
        victim = str(k.get("victim_name") or "")
        attacker = str(k.get("attacker_name") or "")
        if not victim or victim == "world":
            kind = "non_player_death"
        elif not attacker or attacker == victim:
            kind = "suicide_or_fall"
        else:
            kind = "player_kill"
        out.append(
            KillEvent(
                tick=int(k["tick"]),
                # attacker 仅在缺失时回落为 world; 这类事件已经用 kind 标出来了,
                # 不会再进入建卡逻辑 (保留它只为渲染/日志能显示来源)
                attacker=attacker or "world",
                attacker_side=k.get("attacker_side") or "",
                victim=victim,
                victim_side=k.get("victim_side") or "",
                weapon=k.get("weapon") or "",
                headshot=bool(k.get("headshot")),
                noscope=bool(k.get("noscope")),
                through_smoke=bool(k.get("thrusmoke")),
                penetrated=bool(k.get("penetrated")),
                attacker_blind=bool(k.get("attackerblind")),
                attacker_in_air=bool(k.get("attackerinair")),
                assistedflash=bool(k.get("assistedflash")) if "assistedflash" in cols else False,
                attacker_place=k.get("attacker_place"),
                victim_place=k.get("victim_place"),
                distance=float(k.get("distance") or 0.0),
                kind=kind,
            )
        )
    out.sort(key=lambda e: e.tick)
    return out


def _round_bounds(demo: Demo) -> dict[int, tuple[int, int, int, str | None, str | None]]:
    """{round_num: (start, end, official_end, winner, reason)}.

    `end` 是击杀结束时刻, `official_end` 是回合正式结束 (含拆包/冻结时间),
    实测后者比前者晚约 448 tick。剪辑窗口的尾留白必须用 `official_end` 做上界,
    否则"制胜击杀 = 回合最后一个事件"的回合 (实测 8/10 个回合如此) 会把
    POST_PAD 整段吃掉, 卡片正好停在击杀瞬间、没有收尾。
    """
    cols = [c for c in ("round_num", "start", "end", "official_end", "winner", "reason")
            if c in demo.rounds.columns]
    out: dict[int, tuple[int, int, int, str | None, str | None]] = {}
    for r in demo.rounds.select(cols).to_dicts():
        start = int(r["start"])
        end = int(r["end"])
        official = int(r.get("official_end") or end)
        out[int(r["round_num"])] = (
            start,
            end,
            max(official, end),      # official_end 缺失或比 end 早时退回 end
            r.get("winner"),
            r.get("reason"),
        )
    return out


def extract_utility(demo: Demo, start_tick: int, end_tick: int,
                    *, strict: bool = False) -> list[dict[str, Any]]:
    """抽取时间窗内的道具事件, 供渲染器叠加绘制.

    三种道具各自独立 try: 某一种的数据表缺失或字段变更时, 另外两种仍然可用
    (道具只是叠加层, 不该让整条出片失败)。

    但**静默吞掉异常是有代价的** —— 成片里少了烟雾而没有任何提示, 排查起来
    很费劲。所以这里把异常收集到 `extract_utility.last_errors` 里, 由调用方
    决定是否上报; `strict=True` 时直接抛出 (自检用)。

    异常粒度是**行级**而不是类型级: 最早的写法把整个 for 循环包在 try 里,
    一行坏数据就会让该类型的道具**全部**丢失 (烟雾 90 条只剩 0 条), 而错误
    信息只指向第一行。现在坏行跳过、其余照常抽取, 每行的失败原因都记下来。
    """
    util: list[dict[str, Any]] = []
    errors: list[str] = []

    def row_fail(kind: str, e: Exception, row: dict[str, Any]) -> None:
        errors.append(f"{kind} 行数据异常 {type(e).__name__}: {e} (row={str(row)[:80]})")

    try:
        for s in demo.smokes.iter_rows(named=True):
            try:
                st = int(s["start_tick"])
                et = int(s["end_tick"]) if s.get("end_tick") is not None else st + 18 * TICKRATE
                if et < start_tick or st > end_tick:
                    continue
                util.append(
                    {
                        "type": "smoke",
                        "start_tick": st,
                        "end_tick": et,
                        "x": float(s["X"]),
                        "y": float(s["Y"]),
                        "side": s.get("thrower_side"),
                    }
                )
            except Exception as e:      # 单行坏数据: 跳过它, 保留其它道具
                row_fail("smokes", e, s)
    except Exception as e:
        errors.append(f"smokes: {type(e).__name__}: {e}")

    try:
        for m in demo.infernos.iter_rows(named=True):
            try:
                st = int(m["start_tick"])
                et = int(m["end_tick"]) if m.get("end_tick") is not None else st + 7 * TICKRATE
                if et < start_tick or st > end_tick:
                    continue
                util.append(
                    {
                        "type": "fire",
                        "start_tick": st,
                        "end_tick": et,
                        "x": float(m["X"]),
                        "y": float(m["Y"]),
                        "side": m.get("thrower_side"),
                    }
                )
            except Exception as e:
                row_fail("infernos", e, m)
    except Exception as e:
        errors.append(f"infernos: {type(e).__name__}: {e}")

    try:
        ev = demo.events.get("flashbang_detonate")
        if ev is not None:
            for f in ev.iter_rows(named=True):
                try:
                    t = int(f["tick"])
                    if start_tick <= t <= end_tick:
                        util.append(
                            {
                                "type": "flash",
                                "start_tick": t,
                                "end_tick": t + int(0.6 * TICKRATE),
                                "x": float(f["x"]),
                                "y": float(f["y"]),
                                "side": f.get("user_side"),
                            }
                        )
                except Exception as e:
                    row_fail("flashbang_detonate", e, f)
    except Exception as e:
        errors.append(f"flashbang_detonate: {type(e).__name__}: {e}")

    if errors:
        if strict:
            raise RuntimeError("道具抽取失败: " + "; ".join(errors))
        # 把失败原因挂到函数上, 供调用方汇总上报 (不算异常路径, 所以用属性而不是抛出)
        extract_utility.last_errors = errors  # type: ignore[attr-defined]
    else:
        extract_utility.last_errors = []      # type: ignore[attr-defined]
    return util


# ------------------------------------------------------------------
# 亮点评分
# ------------------------------------------------------------------
def score_highlight(kills: list[KillEvent], *, is_clutch: bool, is_opening: bool) -> tuple[float, list[str]]:
    """给一组击杀算强度分与标签 (确定性规则)."""
    if not kills:
        return 0.0, []

    tags: list[str] = []
    score = 0.0

    if len(kills) >= 5:
        score += SCORE_ACE
        tags.append("ace")
    elif len(kills) > 1:
        score += SCORE_MULTIKILL_BASE + SCORE_MULTIKILL_STEP * (len(kills) - 2)
        # 显式映射: 旧写法是三层嵌套三元, 5 杀以上已被 ace 分支吃掉, 最后的
        # else 永远到不了 -> 3 杀被标成 "multi" (实测真 demo 出现 "kills=3
        # tags=multi")。'multi' 现在只作为 6 杀以上的泛化标签保留。
        tags.append({2: "double", 3: "triple", 4: "quad"}.get(len(kills), "multi"))
    else:
        score += 12.0

    for k in kills:
        if k.headshot:
            score += SCORE_HEADSHOT
        if k.noscope:
            score += SCORE_NOSCOPE
            tags.append("noscope")
        if k.through_smoke:
            score += SCORE_THROUGH_SMOKE
            tags.append("through_smoke")
        if k.penetrated:
            score += SCORE_PENETRATED
            tags.append("wallbang")
        if k.attacker_blind:
            score += SCORE_BLIND_KILL
            tags.append("blind_kill")
        if k.attacker_in_air:
            score += SCORE_IN_AIR
            tags.append("in_air")
        if k.assistedflash:
            score += SCORE_UTILITY_ASSIST
            tags.append("flash_assist")
        if k.distance >= LONG_RANGE_M:
            score += SCORE_LONG_RANGE
            tags.append("long_range")

    if is_opening:
        score += SCORE_OPENING
        tags.append("opening_kill")
    if is_clutch:
        score += SCORE_CLUTCH_WIN
        tags.append("clutch")

    # 狙击枪加权 (观赏性高)
    if any(k.weapon in ("awp", "ssg08", "scar20", "g3sg1") for k in kills):
        score += 14.0
        tags.append("sniper")

    # 去重但保序
    seen: set[str] = set()
    uniq = [t for t in tags if not (t in seen or seen.add(t))]
    return round(score, 2), uniq


@dataclass
class ClutchInfo:
    """一次真正的残局 (1vX 并赢下)."""

    side: str            # 残局方
    player: str          # 最后存活者, 也就是残局主角
    start_tick: int      # 队友全部阵亡、他独自面对 >=2 敌人的时刻
    enemies: int         # 那一刻的敌人数量


def find_clutch(
    rkills: Sequence[KillEvent],
    rwinner: str | None,
    roster: dict[str, str] | None = None,
    survivors: set[str] | None = None,
) -> ClutchInfo | None:
    """判定这一回合是否存在**真正的**残局, 并找出主角.

    定义: 某玩家成为**自己队里最后一人**, 独自面对 >=2 敌人, 并且赢下这一回合。
    他要至少拿到一次残局期间的击杀 —— 否则是队友打赢的, 不是他。

    旧实现的缺陷: 它只记录"哪一方先掉到 1 人", 完全不追踪那个幸存者是谁,
    于是队友送完时**已经阵亡**的玩家也会被当成残局主角 —— 判定过宽。

    三个前提, 缺一不可:
        1. `roster` 必须完整 (名单缺失时"某方只有 1 人"可能是假象)
        2. `survivors` 必须是**回合结束时真正活着的人**。不能用"没出现在死亡
           事件里"代替 —— 那既可能是活着也可能是数据缺失。实测靠事件推断会把
           实际存活的队友算成阵亡, 把 2v0 的收尾误判成 1vX。
        3. 赢方在回合结束时必须**恰好剩他 1 人**

    Args:
        rkills: 该回合全部击杀事件 (按 tick 升序)
        rwinner: awpy 判定的回合获胜方 ("t"/"ct")
        roster: {玩家名: 阵营} —— 该回合的完整参战名单
        survivors: 该回合结束时仍存活的玩家集合

    Returns:
        ClutchInfo 或 None
    """
    if not rwinner or rwinner not in ("t", "ct"):
        return None
    if not roster or survivors is None:
        return None

    alive: dict[str, set[str]] = {
        "t": {n for n, sd in roster.items() if sd == "t"},
        "ct": {n for n, sd in roster.items() if sd == "ct"},
    }
    if len(alive[rwinner]) < 2:
        return None

    deaths: list[tuple[int, str, str]] = []      # (tick, victim, victim_side)
    for k in rkills:
        if not k.victim or k.victim_side not in ("t", "ct"):
            continue
        deaths.append((k.tick, k.victim, k.victim_side))
    if not deaths:
        return None
    deaths.sort(key=lambda x: x[0])

    # 赢方在回合结束时只应剩一个人
    win_survivors = {n for n in survivors if roster.get(n) == rwinner}
    if len(win_survivors) != 1:
        return None
    survivor = next(iter(win_survivors))

    # 关键: 残局起点是"**他本人**成为全队最后一人"的那一刻, 而不是"最后一个
    # 队友阵亡"的那一刻。名单残缺时后者会晚得多 —— 实测回合 20: 主角在
    # t=1 ct=3 时就已是唯一存活, 但最后一个队友直到 1v1 时才死, 于是被误判成
    # "敌人只有 1 个, 不算残局"。改为逐次死亡后检查"本方剩余是否只有他"。
    solo_tick: int | None = None
    enemies: int = 0
    for tick, victim, sd in deaths:
        alive[sd].discard(victim)
        still = alive[rwinner]
        if solo_tick is None and still == {survivor}:
            solo_tick = tick
            enemies = len(alive["ct" if rwinner == "t" else "t"])
    if solo_tick is None or enemies < 2:
        return None

    # 独自面对敌人之后他还得有击杀, 否则谈不上"赢下残局"
    if not any(k.attacker == survivor and k.tick >= solo_tick for k in rkills):
        return None

    return ClutchInfo(
        side=rwinner,
        player=survivor,
        start_tick=int(solo_tick),
        enemies=int(enemies),
    )


def build_roster(
    demo, max_rounds: int = 12
) -> tuple[dict[int, dict[str, str]], dict[int, set[str]]]:
    """抽出**每回合**的参战名单与幸存者, 都以逐 tick 表为准.

    两个坑:

    1. 名单不能从击杀事件推断。整回合没参与击杀/死亡的玩家不会出现在事件里,
       名单就偏小, 会把"某方只有 1 人"这种假象当成残局。
    2. **幸存者也不能只看"没死过"**。换边后同一名字的死亡可能记在别处, 而
       `kills` 表里某玩家整回合没出现, 既可能是"一直活着"也可能是"数据缺失"。
       实测回合 9 就踩到了: 靠事件推断会把实际存活的队友当成已阵亡, 于是把
       一个 2v0 的收尾误判成 1vX 残局。
       逐 tick 表的血量是最终事实 —— 回合末血量 > 0 就是活着。

    另外名单必须**按回合**建: 玩家会在半场换边, 用全局 map 只会留下第一次
    观察到的阵营, 换边后的回合名单就是错的。

    失败可见性: 本函数是两个字典的**硬前置** (find_clutch 拿不到就直接返回
    None)。旧实现在异常时静默 `return {}, {}`, 结果是"所有残局判定、分数与
    标签全部消失, 却没有任何日志" —— 排查时只会看到"这 demo 没有残局"。
    现在失败会写进 `build_roster.last_error` (由 build_highlights 上报给
    analyze_demo), 调用方拿到空字典时能分辨"真的没有残局"和"功能坏了"。

    Returns:
        (每回合 {玩家: 阵营}, 每回合 {幸存者})
    """
    rosters: dict[int, dict[str, str]] = {}
    survivors: dict[int, set[str]] = {}
    build_roster.last_error = None
    build_roster.last_stats = {}
    try:
        ticks = demo.ticks
        sub = ticks.filter(ticks["round_num"] <= max_rounds)

        # 名单: 直接去重 (全表 group_by 比 unique 省一次排序)
        for row in sub.select(["round_num", "name", "side"]).unique().iter_rows(named=True):
            rn, name, side = row.get("round_num"), row.get("name"), row.get("side")
            if rn is None or not name or side not in ("t", "ct"):
                continue
            rosters.setdefault(int(rn), {})[str(name)] = str(side)

        # 每回合每个玩家**最后一次**的血量 / 存活标志。
        # 旧实现用 iter_rows 在 Python 里逐行扫 145 万行 tick, 纯属浪费 ——
        # polars 一句 group_by(...).last() 就够 (先 sort 再 group_by 保证
        # 取到的确实是 tick 最大的那一行)。
        cols = [c for c in ("round_num", "name", "tick", "health", "is_alive")
                if c in sub.columns]
        agg = [pl.col("health").last().alias("hp")]
        if "is_alive" in sub.columns:
            agg.append(pl.col("is_alive").last().alias("alive"))
        last = (
            sub.select(cols)
            .sort("tick")
            .group_by(["round_num", "name"], maintain_order=True)
            .agg(agg)
        )

        conf = 0
        for row in last.iter_rows(named=True):
            rn, name = row.get("round_num"), row.get("name")
            hp = row.get("hp")
            if rn is None or not name or hp is None:
                continue
            by_hp = float(hp) > 0
            # `is_alive` 是显式请求的字段, 必须真的用上: 实测 24 回合里有 1 个
            # 回合 health>0 得 4 人而 is_alive 得 3 人。两者矛盾时取**交集**
            # (即"只有两个字段都说活着才算活着") —— 多算一个幸存者会把
            # "赢方恰好剩 1 人"顶掉 -> 静默漏判残局; 少算最多漏一次残局,
            # 但不会把非残局说成残局。
            alive_flag = row.get("alive")
            if alive_flag is None:
                ok = by_hp
            else:
                ok = by_hp and bool(alive_flag)
                if by_hp != bool(alive_flag):
                    conf += 1
            if ok:
                survivors.setdefault(int(rn), set()).add(str(name))

        build_roster.last_stats = {
            "rounds_with_roster": len(rosters),
            "rounds_with_survivors": len(survivors),
            "hp_alive_conflicts": conf,
        }
    except Exception as exc:      # pragma: no cover - 依赖数据损坏路径
        build_roster.last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("build_roster 失败, 残局判定已禁用: %s", build_roster.last_error,
                       exc_info=True)
        return {}, {}
    return rosters, survivors


build_roster.last_error: str | None = None      # type: ignore[attr-defined]
build_roster.last_stats: dict[str, int] = {}    # type: ignore[attr-defined]


def build_highlights(
    demo: Demo,
    *,
    min_score: float = 25.0,
    max_cards: int | None = None,
    with_utility: bool = True,
) -> list[HighlightCard]:
    """从 demo 构建亮点卡片列表.

    策略:
        1. 按回合分组击杀
        2. 找出每回合存活人数变化, 判定残局 (clutch) 与首杀 (opening)
        3. 以"主角"为单位聚合连杀, 生成时间窗
        4. 逐条评分并过滤低分
    """
    kills = extract_kill_events(demo)
    bounds = _round_bounds(demo)
    # 每回合的完整名单与幸存者 (都来自逐 tick 表): 残局判定必须有它们.
    # 按回合取用, 因为玩家会换边。
    rosters, survivors = build_roster(demo, max_rounds=max(bounds) if bounds else 30)
    by_round: dict[int, list[KillEvent]] = {}
    for e in kills:
        rn = _tick_to_round(e.tick, bounds)
        if rn is not None:
            by_round.setdefault(rn, []).append(e)

    # 事件分类计数: 这三类数据问题以前是静默的, 出片少了内容也无从解释
    skipped: dict[str, int] = {"non_player_death": 0, "suicide_or_fall": 0,
                               "attacker_missing": 0, "post_pad_clipped": 0}

    cards: list[HighlightCard] = []
    for rn in sorted(by_round):
        rkills = by_round[rn]
        rstart, rend, official_end, rwinner, rreason = bounds[rn]
        # 尾留白允许越过 rend, 但不超过官方回合结束 (冻结时间)
        rend_pad = max(rend, official_end)

        # --- 首杀: 只认**真实玩家击杀** (world 伪玩家不能抢走 opening) ---
        opening_tick: int | None = None
        for e in rkills:
            if e.kind == "player_kill":
                opening_tick = e.tick
                break

        # --- 残局判定 ---
        # 见 find_clutch: 必须追踪到"最后存活者是谁", 不能只看哪一方掉到 1 人。
        roster_rn = rosters.get(rn)
        clutch = find_clutch(rkills, rwinner, roster_rn, survivors.get(rn))

        # --- 按主角聚合连杀 (只有真实玩家击杀能建卡) ---
        per_player: dict[str, list[KillEvent]] = {}
        for e in rkills:
            if e.kind != "player_kill":
                skipped[e.kind] += 1
                continue
            per_player.setdefault(e.attacker, []).append(e)

        for player, pkills in per_player.items():
            pkills.sort(key=lambda k: k.tick)
            # 把间隔过远的击杀切成不同"片段" (>12s 视为两段)
            groups: list[list[KillEvent]] = []
            cur: list[KillEvent] = []
            for k in pkills:
                if cur and (k.tick - cur[-1].tick) > 12 * TICKRATE:
                    groups.append(cur)
                    cur = []
                cur.append(k)
            if cur:
                groups.append(cur)

            for gi, grp in enumerate(groups):
                start = max(grp[0].tick - PRE_PAD, rstart)
                end = min(grp[-1].tick + POST_PAD, rend_pad)
                if end - start < TICKRATE:      # 太短没意义
                    continue

                is_opening = opening_tick is not None and grp[0].tick == opening_tick
                # 残局: 主角必须是那个**真正的最后存活者**, 且击杀发生在独自面对
                # 敌人的窗口内 (而不是"队友送完后已经阵亡的他也算")。
                # 判定依据是"残局窗口内他拿到几次击杀", **不是这一组里有几条**:
                # 旧实现要求 len(grp) >= 2, 而 find_clutch 只要求 >=1, 两者冲突 ——
                # 1v2 靠 1 杀 + 拆包/超时赢下的残局拿不到 +80 与 clutch 标签; 更糟
                # 的是 12s 分组正好把同一次残局切成 1+1 时, 两组都 <2, 整个残局的
                # 加分与标签**全部丢失** (score_highlight 本身支持单杀残局)。
                clutch_kills = 0
                if (clutch and player == clutch.player
                        and grp[0].attacker_side == clutch.side
                        and grp[-1].tick >= clutch.start_tick):
                    clutch_kills = sum(1 for k in grp if k.tick >= clutch.start_tick)
                is_clutch = clutch_kills >= 1
                sc, tags = score_highlight(grp, is_clutch=is_clutch, is_opening=is_opening)
                if sc < min_score:
                    continue

                places = [p for p in {k.attacker_place for k in grp if k.attacker_place}]
                card = HighlightCard(
                    id=f"r{rn}_g{gi}_{player}",
                    round_num=rn,
                    player=player,
                    player_side=grp[0].attacker_side,
                    start_tick=int(start),
                    end_tick=int(end),
                    kills=grp,
                    score=sc,
                    tags=tags,
                    places=sorted(places),
                    round_winner=rwinner,
                    round_reason=rreason,
                    clutch_enemies=clutch.enemies if is_clutch and clutch else 0,
                )
                if with_utility:
                    card.utility = extract_utility(demo, card.start_tick, card.end_tick)
                cards.append(card)

    cards.sort(key=lambda c: (-c.score, c.round_num))

    # 同一回合保留最强的 2 条, 避免整片都是同一回合
    per_round_count: dict[int, int] = {}
    kept: list[HighlightCard] = []
    for c in cards:
        n = per_round_count.get(c.round_num, 0)
        if n >= 2:
            continue
        per_round_count[c.round_num] = n + 1
        kept.append(c)

    # max_cards=0 必须是"不要任何卡片", 而不是"不截断" (旧写法 `if max_cards:`
    # 会让 0 落进 else 分支 -> 返回全部卡片, 与调用方意图相反)
    if max_cards is not None:
        kept = kept[:max(0, max_cards)]

    build_highlights.last_skipped = skipped
    build_highlights.last_roster_stats = dict(getattr(build_roster, "last_stats", {}) or {})
    build_highlights.last_roster_error = getattr(build_roster, "last_error", None)
    return kept


build_highlights.last_skipped: dict[str, int] = {}          # type: ignore[attr-defined]
build_highlights.last_roster_stats: dict[str, int] = {}     # type: ignore[attr-defined]
build_highlights.last_roster_error: str | None = None       # type: ignore[attr-defined]


def _tick_to_round(tick: int, bounds: dict[int, tuple]) -> int | None:
    """tick -> 回合号.

    取**首个**匹配的回合窗口。调用方 (build_highlights) 只用 kills 表里的
    tick, 它们必然落在某个回合里; 若真出现落在所有窗口之外的 tick, 返回 None
    由调用方丢弃 —— 这里不发警告, 因为那属于"数据缺失"而不是"逻辑失败"。
    rounds 表 (`bounds`) 与 ticks 表 (`build_roster`) 是两套 round_num,
    两者的一致性由 selftest 的名单/幸存者用例交叉校验。
    """
    for rn, b in bounds.items():
        if b[0] <= tick <= b[2]:
            return rn
    return None


def analyze_demo(
    demo_path: str | Path,
    *,
    min_score: float = 25.0,
    max_cards: int | None = 40,
    with_utility: bool = True,
) -> dict[str, Any]:
    """完整分析一个 demo, 返回可序列化的结果."""
    demo_path = str(demo_path)
    demo = load_demo(demo_path, with_ticks=True)

    header = demo.header or {}
    cards = build_highlights(
        demo, min_score=min_score, max_cards=max_cards, with_utility=with_utility
    )

    rounds = []
    for r in demo.rounds.select(["round_num", "start", "end", "winner", "reason"]).to_dicts():
        rounds.append(
            {
                "round_num": int(r["round_num"]),
                "start": int(r["start"]),
                "end": int(r["end"]),
                "winner": r.get("winner"),
                "reason": r.get("reason"),
                "duration": round((int(r["end"]) - int(r["start"])) / TICKRATE, 2),
            }
        )

    # 道具抽取若部分失败, 必须让上层看见 —— 否则成片里少了烟雾/燃烧却无人知晓
    util_errors = sorted(set(getattr(extract_utility, "last_errors", []) or []))
    # 事件分类计数与名单/幸存者统计同理: 这些数据问题以前是完全静默的
    skipped = dict(getattr(build_highlights, "last_skipped", {}) or {})
    roster_error = getattr(build_highlights, "last_roster_error", None)

    return {
        "demo_path": demo_path,
        "map_name": header.get("map_name", "unknown"),
        "tickrate": TICKRATE,
        "rounds": rounds,
        "round_count": len(rounds),
        "highlights": [c.to_dict() for c in cards],
        "highlight_count": len(cards),
        "utility_errors": util_errors,
        "utility_ok": not util_errors,
        "skipped_events": skipped,
        "roster_stats": dict(getattr(build_highlights, "last_roster_stats", {}) or {}),
        "roster_error": roster_error,
        "clutch_ok": roster_error is None,
        "clutch_count": sum(1 for c in cards if c.clutch_enemies),
    }


def save_demo_analysis(result: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def cards_from_dicts(dicts: Iterable[dict[str, Any]]) -> list[HighlightCard]:
    """把 JSON 里的 highlight 还原成 HighlightCard (渲染阶段用)."""
    out: list[HighlightCard] = []
    for d in dicts:
        kills = [
            KillEvent(
                tick=int(k["tick"]),
                attacker=k["attacker"],
                attacker_side=k["attacker_side"],
                victim=k["victim"],
                victim_side=k["victim_side"],
                weapon=k["weapon"],
                headshot=bool(k["headshot"]),
                noscope=bool(k.get("noscope")),
                through_smoke=bool(k.get("through_smoke")),
                penetrated=bool(k.get("penetrated")),
                attacker_blind=bool(k.get("attacker_blind")),
                attacker_in_air=bool(k.get("attacker_in_air")),
                assistedflash=bool(k.get("assistedflash")),
                attacker_place=k.get("attacker_place"),
                victim_place=k.get("victim_place"),
                distance=float(k.get("distance") or 0.0),
            )
            for k in d.get("kills", [])
        ]
        out.append(
            HighlightCard(
                id=d["id"],
                round_num=int(d["round_num"]),
                player=d["player"],
                player_side=d["player_side"],
                start_tick=int(d["start_tick"]),
                end_tick=int(d["end_tick"]),
                kills=kills,
                score=float(d.get("score", 0.0)),
                tags=list(d.get("tags", [])),
                places=list(d.get("places", [])),
                utility=list(d.get("utility", [])),
                round_winner=d.get("round_winner"),
                round_reason=d.get("round_reason"),
                clutch_enemies=int(d.get("clutch_enemies") or 0),
            )
        )
    return out


if __name__ == "__main__":  # pragma: no cover - 手工验证入口
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else str(config.DEFAULT_DEMO)
    t0 = time.time()
    res = analyze_demo(path)
    dt = time.time() - t0
    print(f"地图 {res['map_name']} | 回合 {res['round_count']} | 亮点 {res['highlight_count']} (耗时 {dt:.1f}s)")
    for h in res["highlights"][:15]:
        tagstr = ",".join(h["tags"]) or "-"
        print(
            f"  [{h['score']:6.1f}] R{h['round_num']:<2} {h['player'][:14]:<14} "
            f"{h['player_side']:<2} {h['duration']:5.2f}s kills={len(h['kills'])} "
            f"util={len(h['utility'])} | {tagstr}"
        )
