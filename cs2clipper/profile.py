"""从对话与行为中推断用户偏好.

与 `store.PREFS` 的区别很重要:
    * **显式偏好** (`preferences` 表) 是用户直接设定的**行为开关**, 会直接影响出片。
    * **用户画像** (`user_profile` 表) 是**推断出来的倾向**, 带可信度, 只进提示词,
      不会偷偷改变出片行为。

为什么要分开: 推断一定会有错的时候。如果推断直接写进行为开关, 用户会发现
"我什么都没设, 它却自己改了画幅" —— 那比不推断更糟。画像只作为上下文喂给 LLM,
用户随时能看到、能推翻。

推断逻辑: 每条证据记 (维度, 取值, 来源, 原话)。同一维度下按"证据条数 × 来源权重"
加权投票, 胜出者成为该维度的结论, 可信度 = 胜出权重 / 总权重 × 饱和系数。
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from . import store

# ------------------------------------------------------------------
# 维度定义
# ------------------------------------------------------------------
#: 维度 -> 中文说明。新增维度只需在这里加一行, 并在下面的规则里给出识别方式。
DIMENSIONS: dict[str, str] = {
    "pacing": "剪辑节奏倾向",
    "aspect": "成片画幅倾向",
    "clip_length": "单条镜头长度倾向",
    "llm_usage": "是否偏好 LLM 编排",
    "camera": "镜头取景倾向",
    "motion_fx": "运动特效倾向",
    "workflow": "工作方式偏好",
    "communication": "沟通偏好",
    "verification": "验证标准偏好",
    "scope": "交付范围偏好",
    "platform": "目标发布平台",
    "tech_stack": "技术栈偏好",
}

#: 每种取值的中文解释, 用于展示
VALUE_LABELS: dict[tuple[str, str], str] = {
    ("pacing", "cinematic"): "偏好长镜头、电影感（不赶节奏）",
    ("pacing", "fast"): "偏好快切碎剪",
    ("pacing", "balanced"): "均衡",
    ("aspect", "tall"): "竖屏（抖音/Shorts）",
    ("aspect", "wide"): "横屏（B站/YouTube）",
    ("aspect", "square"): "方形（调试用）",
    ("clip_length", "long"): "单条镜头偏长",
    ("clip_length", "medium"): "单条镜头适中",
    ("clip_length", "short"): "单条镜头偏短",
    ("llm_usage", "off"): "不用 LLM，走确定性编排",
    ("llm_usage", "on"): "用 LLM 编排",
    ("camera", "follow_action"): "要跟拍高光人物（不要俯瞰全图）",
    ("camera", "whole_map"): "要全景俯瞰",
    ("motion_fx", "off"): "不要运动模糊",
    ("motion_fx", "on"): "要运动模糊",
    ("workflow", "defer_vcs"): "暂不处理版本控制，先做功能",
    ("workflow", "commit"): "重视版本控制提交",
    ("communication", "wants_progress"): "要求汇报进度与完成度",
    ("communication", "wants_evidence"): "要求用实测数据说话，不接受空口结论",
    ("communication", "direct"): "回复要简洁直接",
    ("verification", "real_data"): "要用真实数据/真实素材验证，不认合成样本",
    ("verification", "e2e"): "要端到端出片验证，不只看单测",
    ("verification", "self_correct"): "要求主动纠错并承认错误",
    ("scope", "verify_first"): "先验证再优化（不建在未验证假设上）",
    ("scope", "feature_first"): "先做功能，工程化后排",
    ("platform", "douyin"): "抖音",
    ("platform", "bilibili"): "B站",
    ("tech_stack", "sqlite_local"): "偏好本地 sqlite 存储，不要外部依赖",
    ("tech_stack", "no_orm"): "不要为简单存储引入 ORM",
    ("project", "cs2_music_clip"): "项目目标：CS2 demo + 音乐情绪自动剪辑",
}


def _label(dimension: str, value: str) -> str:
    return VALUE_LABELS.get((dimension, value), value)


# ------------------------------------------------------------------
# 对话文本 -> 证据
# ------------------------------------------------------------------
#: (维度, 取值, 正则, 说明)。匹配到就把该片段作为证据记下。
#: 规则宁可少而准: 误判会让画像失真, 而画像会进提示词影响 LLM。
_RULES: list[tuple[str, str, str]] = [
    # --- 沟通方式 ---
    ("communication", "wants_progress",
     r"(完成到什么程度|做到哪|进度|完成度|完成了什么|告诉我.*完成)"),
    ("communication", "wants_evidence",
     r"(用数据|实测|证据|不要空口|怎么验证|凭什么|有没有验证)"),
    ("communication", "direct",
     r"(别废话|直接说|简单点|简洁|先不管|不重要)"),
    # --- 验证标准 ---
    ("verification", "real_data",
     r"(真实音乐|真歌|真实数据|真实素材|别用合成|合成.*掩盖|本机已有)"),
    ("verification", "e2e",
     r"(端到端|出片|完整测试|全量测试|实际跑|跑一遍)"),
    ("verification", "self_correct",
     r"(是不是.*没实现|你确定|真的吗|错了吧|重新核对|别猜)"),
    # --- 交付范围 ---
    ("scope", "verify_first",
     r"(先验证|验证优先|不确定.*先测|先做验证)"),
    ("scope", "feature_first",
     r"(先做其他|先不管|功能优先|先做功能|后面再说)"),
    # --- 工作方式 ---
    ("workflow", "defer_vcs",
     r"(先不管\s*git|不用管\s*git|git.*后面|先不提交|不着急提交)"),
    ("workflow", "commit",
     r"(提交|commit|版本控制|git\s*init)"),
    # --- 画面 ---
    ("pacing", "cinematic",
     r"(电影感|长镜头|慢一点|不要.*太快|余韵|留白多)"),
    ("pacing", "fast",
     r"(快切|快节奏|碎剪|密集|燃一点|更炸)"),
    ("camera", "follow_action",
     r"(跟拍|跟随|镜头.*近|看不清|主角.*点|像集锦|不要.*复盘)"),
    ("camera", "whole_map",
     r"(全景|俯瞰|整张地图|战术复盘)"),
    ("motion_fx", "off",
     r"(关掉.*模糊|不要.*模糊|运动模糊.*慢|模糊.*太贵)"),
    ("motion_fx", "on",
     r"(要.*运动模糊|开启.*模糊|残影|速度感)"),
    # --- 平台 ---
    ("platform", "douyin", r"(抖音|竖屏|9:16|短视频平台)"),
    ("platform", "bilibili", r"(b站|B站|哔哩|横屏|16:9)"),
    # --- 技术栈 ---
    ("tech_stack", "sqlite_local",
     r"(sqlite|本地.*存储|本地目录|落盘|持久化)"),
    ("tech_stack", "no_orm",
     r"(不要.*orm|别.*orm|轻量.*存储)"),
    # --- 项目背景 (来自需求陈述) ---
    ("project", "cs2_music_clip",
     r"(cs2|demo).{0,20}(音乐|剪辑)|音乐.{0,20}(自动)?剪辑|情绪.{0,10}剪辑"),
]

#: 助手提出方案后用户表示接受的说法 —— 这类证据 = "用户认可该方向"。
#: 为什么需要: 很多偏好是通过"用户同意了我的建议"体现的 (例如接受"运动模糊
#: 默认关闭因为太贵"), 这些话里并不含偏好关键词, 但确实是偏好证据。
ACCEPT_RE = re.compile(
    # 纯确认词
    r"(^\s*(好|行|可以|好的|嗯|ok|OK|是的|对|没问题|继续|开始)\s*[!！,，。.~～]?\s*$"
    # 明确表示按建议执行
    r"|按你(说的|给出|的)|就按这个|同意|可以这样|你决定|都行"
    # 选项式回复 (来自选择型提问), 含"（推荐）"这类标注
    r"|^[^。！？!?]{0,40}?(推荐|可以[，,]|现在就|先用|就用)"
    r"|用\S{0,30}(里的|的)(歌|音乐|素材|方案))"
)

#: 助手建议里可被"接受"的要点 -> (维度, 取值)。用户接受时记为证据。
ADVICE_POINTS: list[tuple[str, str, str]] = [
    ("pacing", "cinematic", r"(电影感|长镜头|慢一点|不要.*太快)"),
    ("pacing", "fast", r"(快切|快节奏|碎剪)"),
    ("camera", "follow_action", r"(跟拍|镜头.*近|不要.*复盘)"),
    ("motion_fx", "off", r"(模糊.*(太贵|慢|默认关闭)|运动模糊默认)"),
    ("motion_fx", "on", r"(开启.*模糊|要.*残影)"),
    ("tech_stack", "sqlite_local", r"(sqlite|本地存储|落盘)"),
    ("scope", "verify_first", r"(先验证|验证优先|先做真实音乐)"),
    ("verification", "real_data", r"(真实(音乐|数据|素材)|真歌)"),
]

_SPLIT_RE = re.compile(r"[。！？!?\n；;]+")


def extract_observations(text: str) -> list[tuple[str, str, str]]:
    """从一段文本里抽出 (维度, 取值, 原话) 证据.

    以句子为单位匹配并保留原话, 便于日后回溯"为什么这么推断"。
    """
    out: list[tuple[str, str, str]] = []
    if not text:
        return out
    for raw in _SPLIT_RE.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        for dim, val, pattern in _RULES:
            if re.search(pattern, sentence):
                out.append((dim, val, sentence[:200]))
    return out


def ingest_conversation(
    turns: Iterable[str],
    *,
    speaker: str = "user",
    path=None,
) -> dict[str, int]:
    """把一段对话 (每行/每条消息) 抽成偏好证据并落库.

    只对用户说的话做偏好推断 —— 助手自己的措辞不代表用户偏好。
    """
    added: dict[str, int] = {}
    for text in turns:
        if not text:
            continue
        for dim, val, quote in extract_observations(text):
            store.add_observation(dim, val, source=speaker, quote=quote, path=path)
            added[dim] = added.get(dim, 0) + 1
    return added


def ingest_exchange(agent_text: str, user_reply: str, *, path=None) -> dict[str, int]:
    """处理"助手建议 -> 用户接受"这一对消息.

    很多偏好是通过**用户认可助手的方向**体现的, 而不是用户自己说出来的。
    例如助手说"运动模糊太贵, 建议默认关闭", 用户回"好" —— 这就是一条
    "用户接受该取舍"的证据, 但它不含任何偏好关键词。

    做法: 先看用户回复是否属于接受/认可; 是的话, 再把助手那段话按
    ADVICE_POINTS 解析成要点, 每个要点记一条证据。
    """
    added: dict[str, int] = {}
    if not agent_text or not user_reply:
        return added
    if not ACCEPT_RE.search(user_reply.strip()):
        return added
    for dim, val, pattern in ADVICE_POINTS:
        m = re.search(pattern, agent_text)
        if not m:
            continue
        # 证据原话取助手建议里命中的那一小段, 便于回溯
        start = max(m.start() - 20, 0)
        quote = agent_text[start:m.end() + 20].strip()
        store.add_observation(dim, val, source="accepted_advice",
                              quote=quote, path=path)
        added[dim] = added.get(dim, 0) + 1
    return added


def ingest_run_history(*, limit: int = 50, path=None) -> dict[str, int]:
    """从实际出片记录里推断偏好 (弱证据).

    用户反复用同一组参数出片, 本身就说明倾向。权重低于对话, 因为参数可能
    只是沿用上次的偏好而非主动选择。
    """
    added: dict[str, int] = {}
    runs = [r for r in store.list_runs(limit=limit, path=path)
            if r.get("finished_at")]
    if not runs:
        return added

    def tally(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in runs:
            v = r.get(key)
            if v is None:
                continue
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    n = len(runs)
    for dim, key, thr in (("aspect", "aspect", 0.6), ("llm_usage", "use_llm", 0.6)):
        counter = tally(key)
        if not counter:
            continue
        val, cnt = max(counter.items(), key=lambda kv: kv[1])
        if cnt / n < thr:
            continue        # 没有明显多数就不下结论
        if dim == "llm_usage":
            val = "on" if str(val) in ("1", "True", "true") else "off"
        store.add_observation(
            dim, val, source="run_history",
            quote=f"最近 {n} 次出片中有 {cnt} 次使用 {key}={val}", path=path,
        )
        added[dim] = added.get(dim, 0) + 1

    # 单条镜头长度倾向: 用实际产出的平均镜头时长
    avg = [r["duration_sec"] / r["clips"] for r in runs
           if r.get("clips") and r.get("duration_sec")]
    if avg:
        mean_len = sum(avg) / len(avg)
        val = "long" if mean_len >= 6.0 else ("short" if mean_len <= 3.5 else "medium")
        store.add_observation(
            "clip_length", val, source="run_history",
            quote=f"最近 {len(avg)} 次出片的平均单条镜头 {mean_len:.2f}s", path=path,
        )
        added["clip_length"] = added.get("clip_length", 0) + 1
    return added


# ------------------------------------------------------------------
# 证据 -> 画像
# ------------------------------------------------------------------
def build_profile(path=None) -> list[dict[str, Any]]:
    """把证据汇总成画像并落库, 返回画像行.

    每个维度独立投票:
        score(value) = Σ 该取值的证据权重
        confidence   = score(胜出) / score(全部) × 饱和度

    饱和度 = 胜出权重 / 0.7 —— **不是 3.0**。
    实测教训: 最初把饱和度分母设成 3.0 (要求"权重累到 3 才算证据充分"),
    结果用户明确下达的指令 ("用 SQLite 在本地目录存储") 也只得到 0.9/3×1 = 0.3 分,
    而 `profile_brief` 的阈值恰好是 0.3 —— 于是**一条结论都进不了提示词**,
    整个功能形同虚设。显式诉求本来就该一次就够, 0.7 才是合理门槛。
    """
    obs = store.list_observations(limit=100000, path=path)
    by_dim: dict[str, dict[str, dict[str, Any]]] = {}
    for o in obs:
        dim, val = o["dimension"], o["value"]
        slot = by_dim.setdefault(dim, {}).setdefault(
            val, {"score": 0.0, "n": 0, "quote": "", "best": -1.0}
        )
        slot["score"] += float(o["confidence"])
        slot["n"] += 1
        # 原话保留权重最高的那条
        if float(o["confidence"]) > slot["best"]:
            slot["best"] = float(o["confidence"])
            slot["quote"] = o.get("quote") or ""

    rows: list[dict[str, Any]] = []
    for dim, values in by_dim.items():
        total = sum(v["score"] for v in values.values()) or 1e-9
        val, slot = max(values.items(), key=lambda kv: kv[1]["score"])
        ratio = slot["score"] / total
        # 饱和度: 权重累到 0.7 就算"证据充分" (见 docstring 里为什么不是 3.0)
        saturation = min(slot["score"] / 0.7, 1.0)
        # 证据数量的天花板: 只有一条证据时最高 0.85 —— 单次观察不该等于完全确定,
        # 但也不能低到让结论进不了提示词 (阈值 0.3)。
        cap = min(0.85 + 0.05 * (slot["n"] - 1), 1.0) if slot["n"] >= 1 else 0.0
        conf = round(min(ratio * saturation, cap), 3)
        rows.append({
            "dimension": dim,
            "value": val,
            "confidence": conf,
            "evidence": slot["n"],
            "quote": slot["quote"],
        })

    rows.sort(key=lambda r: (-r["confidence"], r["dimension"]))
    in_store = [r for r in rows if r["confidence"] > 0]
    store.save_profile(in_store, path=path)
    return in_store


def profile_brief(path=None, *, min_confidence: float = 0.3) -> dict[str, str]:
    """给 LLM 用的画像摘要: 只保留可信度达标的结论."""
    out: dict[str, str] = {}
    for r in store.load_profile(path):
        if r["confidence"] >= min_confidence:
            out[r["dimension"]] = _label(r["dimension"], r["value"])
    return out


def describe_profile(path=None, *, show_quotes: bool = True) -> str:
    """人类可读的画像清单."""
    rows = store.load_profile(path)
    if not rows:
        return "  (暂无画像 —— 先用 --learn-from 或 --observe 收集证据)"
    lines = []
    for r in rows:
        bar = "█" * int(round(r["confidence"] * 10)) + "░" * (10 - int(round(r["confidence"] * 10)))
        lines.append(
            f"  [{bar}] {r['confidence']:.2f}  {r['dimension']:<14} "
            f"= {r['value']:<16} {_label(r['dimension'], r['value'])}"
            f"  (证据 {r['evidence']} 条)"
        )
        if show_quotes and r.get("quote"):
            lines.append(f"          依据: 「{r['quote'][:70]}」")
    return "\n".join(lines)


def format_observations(path=None, limit: int = 40) -> str:
    """最近收集到的证据 (便于核对推断是否合理)."""
    rows = store.list_observations(limit=limit, path=path)
    if not rows:
        return "  (暂无证据)"
    lines = []
    for r in rows:
        lines.append(
            f"  #{r['id']:<4} {r['dimension']:<14} = {r['value']:<16} "
            f"[{r['source']}, w={r['confidence']:.2f}]"
            + (f"  「{r['quote'][:50]}」" if r.get("quote") else "")
        )
    return "\n".join(lines)
