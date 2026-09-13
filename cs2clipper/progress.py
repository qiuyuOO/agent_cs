"""流水线进度事件 —— 给 CLI / Web 界面 / 测试共用的一个极薄通道.

为什么要单独一层: 从命令行跑的时候只需要 `print`, 但 Web 界面需要**结构化**的
进度 (阶段 / 百分比 / 文字), 而且流水线的两个入口节点是 LangGraph 并发执行的
(两条线程), 事件必须线程安全。

设计上刻意保持"零依赖、可失败": 回调抛异常绝不能让出片失败 —— 进度只是附加
信息, 用户宁可没有进度条也不愿意因为一个 UI 的 bug 丢掉一次 200 秒的渲染。

用法:
    def on_event(ev):
        print(ev.stage, ev.percent)

    state = pipeline.run(music, demo, on_event=on_event)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# 阶段的显示名与权重 (进度条按权重估算总进度)
STAGES: dict[str, tuple[str, float]] = {
    "analyze_music": ("分析音乐", 0.08),
    "analyze_demo": ("解析 demo 与抽取亮点", 0.22),
    "plan": ("编排剪辑 (EDL)", 0.15),
    "save_artifacts": ("保存中间产物", 0.02),
    "render": ("渲染画面", 0.48),
    "compose": ("拼接并铺音乐", 0.05),
}
STAGE_ORDER = list(STAGES)

# 累计权重 (用于把"第 N 阶段"翻译成百分比)
_CUM: dict[str, float] = {}
_acc = 0.0
for _k in STAGE_ORDER:
    _CUM[_k] = _acc
    _acc += STAGES[_k][1]
_TOTAL_W = _acc or 1.0


def stage_percent(stage: str, fraction: float = 0.0) -> float:
    """阶段内进度 -> 全局百分比 (0~100)."""
    base = _CUM.get(stage, 0.0)
    w = STAGES.get(stage, ("", 0.0))[1]
    frac = min(max(float(fraction), 0.0), 1.0)
    return round(min((base + w * frac) / _TOTAL_W, 1.0) * 100, 1)


@dataclass
class Event:
    """一条进度事件 (可直接序列化成 JSON 发给前端)."""

    kind: str                      # stage_start / stage_done / log / progress / done / error
    stage: str = ""                # 见 STAGES
    message: str = ""
    percent: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)

    @property
    def stage_label(self) -> str:
        return STAGES.get(self.stage, (self.stage, 0.0))[0]

    @property
    def affects_percent(self) -> bool:
        """这条事件是否代表"进度前进".

        `log` **不算**: 它可能在任何时刻从任何阶段发出 (例如 plan 阶段的
        告警在 analyze_music 还没结束时就打出来了), 拿它推进度条会让百分比
        倒退。前端只按 stage_start / stage_done / progress / done 画进度条,
        log 里带的 percent 只当参考值。
        """
        return self.kind in ("stage_start", "stage_done", "progress", "done", "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "message": self.message,
            "percent": self.percent,
            "affects_percent": self.affects_percent,
            "detail": self.detail,
            "seq": self.seq,
            "ts": round(self.ts, 3),
        }


Emitter = Callable[[Event], None]


class Bus:
    """把发射器包成"线程安全 + 不抛异常 + 自动补 seq/percent"的小工具.

    流水线节点里只写 `bus.stage_done("plan", "18 段")`, 不用关心 seq、
    百分比、异常处理这些事。
    """

    def __init__(self, emit: Emitter | None) -> None:
        self._emit = emit
        self._lock = threading.Lock()
        self._seq = 0
        self._high = 0.0
        self.dropped = 0        # 被丢弃的事件数 (回调抛异常时 +1)

    def _send(self, ev: Event) -> None:
        with self._lock:
            self._seq += 1
            ev.seq = self._seq
            if ev.affects_percent:
                # 百分比必须**单调不减**: analyze_music 与 analyze_demo 在图里
                # 是并发的, 两个阶段的 stage_start/stage_done 会交错到达。
                # 各自按自己的权重算出来的百分比必然来回跳 (实测序列
                # 8.0 → 0.0 → 30.0), 进度条看起来像在倒退。
                # 这里统一取"历史最高值", 让进度只增不减。
                ev.percent = round(max(ev.percent, self._high), 1)
                self._high = ev.percent
        if self._emit is None:
            return
        try:
            self._emit(ev)
        except Exception:
            # 进度上报失败绝不能影响出片 —— 但也**不能完全静默**:
            # 计数挂在 bus.dropped 上, Web 端会报出来。
            with self._lock:
                self.dropped += 1

    def stage_start(self, stage: str, message: str = "", **detail: Any) -> None:
        self._send(Event(kind="stage_start", stage=stage, message=message or
                         f"开始{STAGES.get(stage, (stage, 0))[0]}",
                         percent=stage_percent(stage, 0.0), detail=detail))

    def stage_done(self, stage: str, message: str = "", **detail: Any) -> None:
        self._send(Event(kind="stage_done", stage=stage, message=message,
                         percent=stage_percent(stage, 1.0), detail=detail))

    def progress(self, stage: str, fraction: float, message: str = "",
                 **detail: Any) -> None:
        self._send(Event(kind="progress", stage=stage, message=message,
                         percent=stage_percent(stage, fraction), detail=detail))

    def log(self, message: str, stage: str = "", **detail: Any) -> None:
        self._send(Event(kind="log", stage=stage, message=message,
                         percent=stage_percent(stage, 1.0) if stage else 0.0,
                         detail=detail))

    def done(self, message: str = "完成", **detail: Any) -> None:
        self._send(Event(kind="done", message=message, percent=100.0, detail=detail))

    def error(self, message: str, stage: str = "", **detail: Any) -> None:
        self._send(Event(kind="error", stage=stage, message=message,
                         percent=stage_percent(stage, 1.0) if stage else 0.0,
                         detail=detail))

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq


def from_state(state: dict[str, Any]) -> Bus:
    """从流水线 state 里取出总线 (没有 on_event 时是空总线)."""
    bus = state.get("_bus")
    return bus if isinstance(bus, Bus) else Bus(None)
