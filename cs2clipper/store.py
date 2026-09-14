"""用户偏好与运行记录的本地 SQLite 存储.

数据库位置: `data/clipper.db` (工作区内, 已加入 .gitignore —— 它是用户本地状态,
不该进版本库)。首次使用自动建表, 无需手动初始化。

三张表:
    preferences  用户偏好 (键值对, 带更新时间和来源)
    runs         每次出片的记录 (输入/设置/产物/耗时)
    clips        每次运行里每一段剪辑的明细

设计原则:
    * **只用标准库 sqlite3** —— 不为一个键值存储引入 ORM 依赖
    * **列定义集中在 SCHEMA 里** —— 建表与迁移共用, 改一处即可
    * **偏好读取永不抛错** —— 存储坏了也不该让出片失败, 一律回落默认值
    * **写操作用事务** —— 批量设置偏好时要么全成要么全不成

为什么需要它: 出片涉及十几个参数 (画幅/帧率/段数/是否用 LLM/镜头远近/特效开关),
每次都手打一遍命令行不现实, 而且同样的偏好会被反复使用。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config

# ------------------------------------------------------------------
# 表结构
# ------------------------------------------------------------------
SCHEMA: dict[str, str] = {
    "preferences": """
        CREATE TABLE IF NOT EXISTS preferences (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,           -- 统一按 JSON 存, 读回时按类型还原
            updated_at TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'user'   -- user=用户显式设置
        )
    """,
    "runs": """
        CREATE TABLE IF NOT EXISTS runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at     TEXT NOT NULL,
            finished_at    TEXT,
            demo_path      TEXT,
            music_path     TEXT,
            out_dir        TEXT,
            map_name       TEXT,
            aspect         TEXT,
            fps            INTEGER,
            use_llm        INTEGER,            -- 0/1
            planner        TEXT,               -- llm / fallback
            clips          INTEGER,
            duration_sec   REAL,
            video_path     TEXT,
            video_bytes    INTEGER,
            elapsed_sec    REAL,
            params_json    TEXT                -- 完整参数字典, 便于回溯
        )
    """,
    "clips": """
        CREATE TABLE IF NOT EXISTS clips (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            idx           INTEGER NOT NULL,      -- 第几段 (从 0 开始)
            music_segment INTEGER,
            out_start     REAL,
            out_end       REAL,
            highlight_id  TEXT,
            player        TEXT,
            round_num     INTEGER,
            score         REAL,
            tags          TEXT,
            speed         REAL
        )
    """,
    # --- 从对话/行为中观察到的偏好证据 ---
    "pref_observations": """
        CREATE TABLE IF NOT EXISTS pref_observations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dimension  TEXT NOT NULL,     -- 偏好维度, 如 pacing / aspect / workflow
            value      TEXT NOT NULL,     -- 观察到的倾向
            confidence REAL NOT NULL,     -- 0~1, 由证据类型决定
            source     TEXT NOT NULL,     -- conversation / explicit_set / run_history / manual
            quote      TEXT,              -- 原话或事实依据 (便于回溯为什么这么推断)
            run_id     INTEGER            -- 若来自某次运行
        )
    """,
    # --- 推断出的画像 (由证据汇总而来, 可随时重建) ---
    "user_profile": """
        CREATE TABLE IF NOT EXISTS user_profile (
            dimension  TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            confidence REAL NOT NULL,     -- 0~1
            evidence   INTEGER NOT NULL,  -- 支持该结论的证据条数
            quote      TEXT,              -- 最有代表性的一句依据
            updated_at TEXT NOT NULL
        )
    """,
}

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_clips_run ON clips(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_obs_dim ON pref_observations(dimension)",
)


# ------------------------------------------------------------------
# 偏好定义
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PrefSpec:
    """一条偏好的元信息."""

    default: Any
    cast: type
    desc: str
    choices: tuple[str, ...] | None = None

    def coerce(self, raw: Any) -> Any:
        """把外部输入 (字符串/任意) 转成该偏好的类型."""
        if self.cast is bool:
            if isinstance(raw, bool):
                return raw
            s = str(raw).strip().lower()
            if s in ("1", "true", "yes", "y", "on", "开", "是"):
                return True
            if s in ("0", "false", "no", "n", "off", "关", "否"):
                return False
            raise ValueError(f"无法解析布尔值: {raw!r}")
        if self.cast is int:
            return int(float(str(raw).strip()))
        if self.cast is float:
            return float(str(raw).strip())
        val = str(raw).strip()
        if self.choices and val not in self.choices:
            raise ValueError(f"取值必须是 {list(self.choices)} 之一, 收到 {val!r}")
        return val


#: 所有可持久化的偏好。新增偏好只需在这里加一行。
PREFS: dict[str, PrefSpec] = {
    # --- 输出 ---
    "aspect": PrefSpec(config.DEFAULT_ASPECT, str,
                       "画幅", choices=tuple(sorted(config.ASPECT_PRESETS))),
    "fps": PrefSpec(config.FPS, int, "输出帧率"),
    "out_root": PrefSpec("out", str, "成片输出根目录 (相对工作区)"),
    # --- 输入 ---
    # 常用 demo 的路径。存下来就不必每次在命令行敲一长串绝对路径;
    # 命令行 --demo 仍然可以覆盖它。
    "default_demo": PrefSpec("", str, "默认 demo 路径 (留空则用项目内置示例)"),
    # --- 画面来源 ---
    # radar = 2D 雷达动画 (全自动, 不需要游戏)
    # hlae  = HLAE + CS2 游戏内录制 (真实游戏画面, 需要装 HLAE 且手动跑一次录制)
    "record_source": PrefSpec("radar", str,
                              "画面来源: radar=2D 雷达动画 / hlae=HLAE 游戏内录制",
                              choices=("radar", "hlae")),
    "hlae_dir": PrefSpec("", str, "HLAE 安装目录 (留空则用 tools/hlae)"),
    "hlae_output_dir": PrefSpec("", str,
                                "HLAE 录制输出目录 (留空则用 work/hlae_record)"),
    "hlae_fps": PrefSpec(60, int, "HLAE 录制帧率 (越高越吃磁盘, 60 够用)"),
    "cs2_exe": PrefSpec("", str, "cs2.exe 路径 (留空则自动在 Steam 库里找)"),
    # --- 编排 ---
    "use_llm": PrefSpec(True, bool, "是否用 LLM 编排 (关掉则走确定性兜底)"),
    "pacing": PrefSpec("balanced", str,
                       "剪辑节奏: fast=快切碎剪 / balanced=跟随情绪 / "
                       "cinematic=长镜头电影感",
                       choices=("fast", "balanced", "cinematic")),
    "max_clips": PrefSpec(22, int, "剪辑段数上限"),
    "max_cards": PrefSpec(40, int, "亮点素材池大小"),
    "min_score": PrefSpec(25.0, float, "素材入选的最低评分"),
    # --- 画面 ---
    "zoom_base": PrefSpec(1.0, float, "镜头基础缩放 (>1 拉近)"),
    "zoom_per_arousal": PrefSpec(0.45, float, "激烈度每增加 1.0 额外拉近的比例"),
    "fade_in": PrefSpec(0.18, float, "段首黑场淡入时长 (秒, 0=关闭)"),
    "fade_out": PrefSpec(0.22, float, "段尾黑场淡出时长 (秒)"),
    "shake_px": PrefSpec(9.0, float, "击杀震屏幅度 (像素, 0=关闭)"),
    "trail_blur": PrefSpec(0.0, float, "拖尾模糊半径 (0=关闭, >0 会明显变慢)"),
    "vignette": PrefSpec(True, bool, "四周压暗"),
    "show_places": PrefSpec(True, bool, "在雷达上标注点位名"),
    "show_hud": PrefSpec(True, bool, "显示 HUD"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------
# 连接
# ------------------------------------------------------------------
def db_path() -> Path:
    """数据库文件路径 (工作区内的 data/clipper.db)."""
    return config.DATA_DIR / "clipper.db"


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """打开数据库连接 (自动建目录/建表), 退出时提交并关闭."""
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_db(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    """建表 + 建索引 (幂等)."""
    for ddl in SCHEMA.values():
        conn.execute(ddl)
    for ddl in INDEXES:
        conn.execute(ddl)


# ------------------------------------------------------------------
# 偏好读写
# ------------------------------------------------------------------
def defaults() -> dict[str, Any]:
    """所有偏好的内置默认值."""
    return {k: v.default for k, v in PREFS.items()}


def load_prefs(path: Path | None = None) -> dict[str, Any]:
    """读出全部偏好, 与默认值合并.

    **永不抛错**: 数据库损坏/读不到时返回默认值, 保证出片流程不被存储问题拖垮。
    """
    merged = defaults()
    try:
        with connect(path) as conn:
            for row in conn.execute("SELECT key, value FROM preferences"):
                key = row["key"]
                if key not in PREFS:
                    continue        # 忽略已废弃的键
                try:
                    merged[key] = PREFS[key].coerce(json.loads(row["value"]))
                except Exception:
                    continue        # 单条坏了就跳过, 用默认值
    except Exception:
        return defaults()
    return merged


def get_pref(key: str, path: Path | None = None) -> Any:
    """读单个偏好 (未知键抛 KeyError, 便于尽早发现拼写错误)."""
    if key not in PREFS:
        raise KeyError(f"未知偏好: {key} (可用: {sorted(PREFS)})")
    return load_prefs(path).get(key, PREFS[key].default)


def set_prefs(values: dict[str, Any], path: Path | None = None,
              *, source: str = "user") -> dict[str, Any]:
    """写入若干偏好, 返回写入后的完整偏好.

    值会先按 PrefSpec 校验与转型 —— 非法值直接报错, 不会污染数据库。
    """
    cleaned: dict[str, Any] = {}
    for key, raw in values.items():
        if key not in PREFS:
            raise KeyError(f"未知偏好: {key} (可用: {sorted(PREFS)})")
        cleaned[key] = PREFS[key].coerce(raw)

    if cleaned:
        ts = _now()
        with connect(path) as conn:
            conn.executemany(
                "INSERT INTO preferences(key, value, updated_at, source) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "  value=excluded.value, updated_at=excluded.updated_at, "
                "  source=excluded.source",
                [(k, json.dumps(v, ensure_ascii=False), ts, source)
                 for k, v in cleaned.items()],
            )
    return load_prefs(path)


def clear_prefs(keys: list[str] | None = None, path: Path | None = None) -> int:
    """删除偏好 (keys 为空则清空全部). 返回删除条数."""
    with connect(path) as conn:
        if keys:
            marks = ",".join("?" * len(keys))
            cur = conn.execute(f"DELETE FROM preferences WHERE key IN ({marks})", keys)
        else:
            cur = conn.execute("DELETE FROM preferences")
        return cur.rowcount


def describe_prefs(prefs: dict[str, Any]) -> str:
    """人类可读的偏好清单 (含默认值对比)."""
    lines = []
    for key, spec in PREFS.items():
        cur = prefs.get(key, spec.default)
        mark = "" if cur == spec.default else "  (默认 %s)" % (spec.default,)
        lines.append(f"  {key:<18} = {str(cur):<10}{mark}   {spec.desc}")
    return "\n".join(lines)


def effect_overrides(prefs: dict[str, Any]) -> dict[str, Any]:
    """把画面类偏好转成 RenderStyle 能吃的 effects 字段."""
    return {
        "fade_in": prefs.get("fade_in", PREFS["fade_in"].default),
        "fade_out": prefs.get("fade_out", PREFS["fade_out"].default),
        "shake_px": prefs.get("shake_px", PREFS["shake_px"].default),
        "trail_blur": prefs.get("trail_blur", PREFS["trail_blur"].default),
        "vignette": prefs.get("vignette", PREFS["vignette"].default),
        "show_places": prefs.get("show_places", PREFS["show_places"].default),
        "show_hud": prefs.get("show_hud", PREFS["show_hud"].default),
    }


# ------------------------------------------------------------------
# 运行记录
# ------------------------------------------------------------------
def start_run(demo_path: str, music_path: str, out_dir: str,
              params: dict[str, Any], path: Path | None = None) -> int:
    """登记一次运行, 返回 run_id."""
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO runs(started_at, demo_path, music_path, out_dir, params_json) "
            "VALUES(?, ?, ?, ?, ?)",
            (_now(), str(demo_path), str(music_path), str(out_dir),
             json.dumps(params, ensure_ascii=False, default=str)),
        )
        return int(cur.lastrowid)


def finish_run(run_id: int, *, map_name: str | None = None, aspect: str | None = None,
               fps: int | None = None, use_llm: bool | None = None,
               planner: str | None = None, clips: int | None = None,
               duration_sec: float | None = None, video_path: str | None = None,
               video_bytes: int | None = None, elapsed_sec: float | None = None,
               path: Path | None = None) -> None:
    """补全一次运行的结果."""
    with connect(path) as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, map_name=?, aspect=?, fps=?, use_llm=?, "
            " planner=?, clips=?, duration_sec=?, video_path=?, video_bytes=?, "
            " elapsed_sec=? WHERE id=?",
            (_now(), map_name, aspect, fps,
             None if use_llm is None else int(use_llm),
             planner, clips, duration_sec, video_path, video_bytes, elapsed_sec, run_id),
        )


def record_clips(run_id: int, clips: list[dict[str, Any]],
                 path: Path | None = None) -> int:
    """写入某次运行的逐段明细."""
    if not clips:
        return 0
    rows = [
        (run_id, i, c.get("music_segment"), c.get("out_start"), c.get("out_end"),
         c.get("highlight_id"), c.get("player"), c.get("round_num"),
         c.get("score"), json.dumps(c.get("tags", []), ensure_ascii=False),
         c.get("speed"))
        for i, c in enumerate(clips)
    ]
    with connect(path) as conn:
        conn.executemany(
            "INSERT INTO clips(run_id, idx, music_segment, out_start, out_end, "
            " highlight_id, player, round_num, score, tags, speed) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def list_runs(limit: int = 10, path: Path | None = None) -> list[dict[str, Any]]:
    """最近的运行记录 (新的在前)."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id, started_at, finished_at, map_name, aspect, fps, use_llm, "
            "       planner, clips, duration_sec, video_path, elapsed_sec "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def last_run(path: Path | None = None) -> dict[str, Any] | None:
    """最近一次成功出片的运行记录."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE finished_at IS NOT NULL AND video_path IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def stats(path: Path | None = None) -> dict[str, Any]:
    """汇总统计, 供 `--show-stats` 展示."""
    with connect(path) as conn:
        n_runs = conn.execute(
            "SELECT COUNT(*) c FROM runs WHERE finished_at IS NOT NULL"
        ).fetchone()["c"]
        n_clips = conn.execute("SELECT COUNT(*) c FROM clips").fetchone()["c"]
        n_prefs = conn.execute("SELECT COUNT(*) c FROM preferences").fetchone()["c"]
        n_obs = conn.execute("SELECT COUNT(*) c FROM pref_observations").fetchone()["c"]
        n_prof = conn.execute("SELECT COUNT(*) c FROM user_profile").fetchone()["c"]
        row = conn.execute(
            "SELECT AVG(elapsed_sec) e, SUM(video_bytes) b FROM runs "
            "WHERE finished_at IS NOT NULL"
        ).fetchone()
    return {
        "runs": int(n_runs),
        "clips": int(n_clips),
        "prefs": int(n_prefs),
        "observations": int(n_obs),
        "profile_dims": int(n_prof),
        "avg_elapsed_sec": float(row["e"]) if row and row["e"] else 0.0,
        "total_video_bytes": int(row["b"]) if row and row["b"] else 0,
        "db_path": str(path or db_path()),
        "db_bytes": (path or db_path()).stat().st_size
        if (path or db_path()).is_file() else 0,
    }


# ------------------------------------------------------------------
# 偏好证据 (从对话/行为观察而来)
# ------------------------------------------------------------------
#: 证据来源 -> 可信度。用户显式设置最高, 助手推断最低。
SOURCE_WEIGHT: dict[str, float] = {
    "explicit_set": 1.00,      # 用户明确 --set 或直接下达的指令
    "conversation": 0.90,      # 对话里明确说出的偏好
    "accepted_advice": 0.70,   # 用户认可了助手提出的方向/取舍
    "run_history": 0.55,       # 从实际出片参数推断
    "assistant_inference": 0.45,  # 助手从行为模式推断 (最弱, 需要更多佐证)
}


def add_observation(
    dimension: str,
    value: str,
    *,
    source: str = "conversation",
    quote: str | None = None,
    confidence: float | None = None,
    run_id: int | None = None,
    path: Path | None = None,
) -> int:
    """记录一条偏好证据, 返回 observation id.

    confidence 不传则按来源取默认权重。同一 (dimension, value) 重复出现会累积
    多条证据 —— 画像的可信度正是由"证据条数 × 来源权重"决定的。
    """
    conf = SOURCE_WEIGHT.get(source, 0.5) if confidence is None else float(confidence)
    conf = max(0.0, min(1.0, conf))
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO pref_observations"
            "(created_at, dimension, value, confidence, source, quote, run_id) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (_now(), str(dimension), str(value), conf, str(source),
             (quote or "")[:400] or None, run_id),
        )
        return int(cur.lastrowid)


def list_observations(dimension: str | None = None, limit: int = 200,
                      path: Path | None = None) -> list[dict[str, Any]]:
    """列出证据 (可按维度过滤)."""
    with connect(path) as conn:
        if dimension:
            rows = conn.execute(
                "SELECT * FROM pref_observations WHERE dimension=? "
                "ORDER BY id DESC LIMIT ?", (dimension, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pref_observations ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    return [dict(r) for r in rows]


def clear_observations(dimension: str | None = None, path: Path | None = None) -> int:
    with connect(path) as conn:
        if dimension:
            cur = conn.execute("DELETE FROM pref_observations WHERE dimension=?",
                               (dimension,))
        else:
            cur = conn.execute("DELETE FROM pref_observations")
        return cur.rowcount


# ------------------------------------------------------------------
# 推断出的用户画像
# ------------------------------------------------------------------
def save_profile(rows: list[dict[str, Any]], path: Path | None = None) -> int:
    """整表替换用户画像 (画像是由证据推导的, 每次重算)."""
    ts = _now()
    with connect(path) as conn:
        conn.execute("DELETE FROM user_profile")
        conn.executemany(
            "INSERT INTO user_profile"
            "(dimension, value, confidence, evidence, quote, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            [(r["dimension"], str(r["value"]), float(r["confidence"]),
              int(r["evidence"]), (r.get("quote") or "")[:400] or None, ts)
             for r in rows],
        )
    return len(rows)


def load_profile(path: Path | None = None) -> list[dict[str, Any]]:
    """读用户画像 (按可信度降序)."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT dimension, value, confidence, evidence, quote, updated_at "
            "FROM user_profile ORDER BY confidence DESC, dimension"
        ).fetchall()
    return [dict(r) for r in rows]


def profile_as_dict(path: Path | None = None) -> dict[str, Any]:
    """画像转成 {维度: 值} 的扁平字典, 便于注入提示词."""
    return {r["dimension"]: r["value"] for r in load_profile(path)}
