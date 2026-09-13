"""CS2 自动剪辑的本地 Web 界面 (Starlette + 原生 JS, 零新依赖).

启动:
    .venv\\Scripts\\python.exe -m cs2clipper.webapp            # 127.0.0.1:8760
    .venv\\Scripts\\python.exe -m cs2clipper.webapp --port 9000 --no-browser

为什么用 Starlette 而不是 FastAPI/Flask: 本机 pip 源上没有 fastapi/flask,
而 starlette 与 uvicorn 是 langchain/mcp 的既有依赖, 已经在环境里 —— 加界面
不需要新增任何依赖, 这在"安装要靠代理"的环境里很重要。

界面本身是**一个 HTML + 一个 JS + 一个 CSS**, 没有构建链 (没有 node/npm),
浏览器直接打开; 所有数据走 /api/*, 长任务进度走 SSE。

安全边界 (这是本地工具, 但默认只监听 127.0.0.1):
    * 素材路径: 允许任意绝对路径 (音乐库常在别的盘), 但必须是存在的常规文件
    * 产物路径: 只允许工作区内, 由 media.safe_under 做 realpath 前缀校验
    * 渲染是 CPU/磁盘重活: 同一时刻只跑一个任务, 其余排队
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from . import config, media, profile as profile_mod, progress, store

STATIC_DIR = Path(__file__).parent / "web"
MAX_EVENT_HISTORY = 800          # 每个任务保留的事件条数 (含 SSE 重放)
JOB_KEEP = 24                    # 进程内保留的最近任务数


# ------------------------------------------------------------------
# 任务
# ------------------------------------------------------------------
@dataclass
class Job:
    """一次出片任务 (在独立线程里跑流水线)."""

    id: str
    params: dict[str, Any]
    status: str = "queued"          # queued|running|done|error|cancelled
    percent: float = 0.0
    stage: str = ""
    stage_label: str = ""
    message: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    out_dir: str = ""
    video_path: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    dropped_events: int = 0         # 因订阅队列满而丢弃的事件数 (可观测)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _subs: list[queue.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # --- 事件分发 -------------------------------------------------
    def publish(self, ev: progress.Event) -> None:
        """线程安全地记录事件并广播给所有 SSE 订阅者."""
        d = ev.to_dict()
        with self._lock:
            self.events.append(d)
            if len(self.events) > MAX_EVENT_HISTORY:
                del self.events[: len(self.events) - MAX_EVENT_HISTORY]
            self.percent = d["percent"] or self.percent
            if d["kind"] == "stage_start":
                self.stage, self.stage_label = d["stage"], d["stage_label"]
                self.status = "running"
            elif d["kind"] == "stage_done":
                self.stage, self.stage_label = d["stage"], d["stage_label"]
            if d["message"]:
                self.message = d["message"]
            if d["kind"] == "error":
                self.error = d["message"]
            if d["kind"] == "done":
                self.result = d["detail"]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(d)
            except queue.Full:
                # 订阅者（浏览器）消费不过来: 丢这一条而不是阻塞渲染线程。
                # 事件是增量的, 丢中间几条只影响日志完整度, 不影响最终状态 ——
                # 前端重连时会按 seq 重放历史。
                self.dropped_events += 1

    def subscribe(self) -> tuple[queue.Queue, list[dict[str, Any]]]:
        """订阅事件流; 返回 (队列, 已有事件快照) 以便断线重连不丢历史."""
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            replay = list(self.events)
            self._subs.append(q)
        return q, replay

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def snapshot(self) -> dict[str, Any]:
        """给前端的状态快照 (不含事件流)."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "percent": self.percent,
                "stage": self.stage,
                "stage_label": self.stage_label,
                "message": self.message,
                "error": self.error,
                "params": self.params,
                "out_dir": self.out_dir,
                "video_path": self.video_path,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed": round((self.finished_at or time.time())
                                 - (self.started_at or self.created_at), 1),
                "result": self.result,
                "dropped_events": self.dropped_events,
            }


class JobManager:
    """任务队列: 同一时刻只跑一个渲染 (CPU/磁盘都吃满, 并发只会互相拖慢)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._runner: threading.Thread | None = None

    def create(self, params: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > JOB_KEEP:
                old = self._order.pop(0)
                if self._jobs.get(old) and self._jobs[old].status in ("done", "error", "cancelled"):
                    self._jobs.pop(old, None)
            if self._runner is None or not self._runner.is_alive():
                self._runner = threading.Thread(target=self._pump, daemon=True,
                                                name="cs2clipper-jobs")
                self._runner.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order if i in self._jobs]

    # --- 执行 -----------------------------------------------------
    def _pump(self) -> None:
        while True:
            with self._lock:
                nxt = next((self._jobs[i] for i in self._order
                            if self._jobs.get(i) and self._jobs[i].status == "queued"), None)
            if nxt is None:
                return
            try:
                self._run(nxt)
            except Exception:
                traceback.print_exc()

    def _run(self, job: Job) -> None:
        from . import pipeline

        job.status = "running"
        job.started_at = time.time()
        p = job.params
        job.publish(progress.Event(kind="log", message="任务已开始",
                                   percent=0.0, detail={"params": p}))
        try:
            state = pipeline.run(
                p["music_path"],
                p.get("demo_path") or None,
                out_dir=p.get("out_dir") or None,
                aspect=p.get("aspect"),
                fps=p.get("fps"),
                max_clips=p.get("max_clips"),
                max_cards=p.get("max_cards"),
                use_llm=p.get("use_llm"),
                pacing=p.get("pacing"),
                verbose=False,
                use_prefs=bool(p.get("use_prefs", True)),
                record=bool(p.get("record", True)),
                on_event=job.publish,
                should_cancel=job._cancel.is_set,
            )
            job.out_dir = str(state.get("out_dir", ""))
            job.video_path = state.get("video_path")
            job.result.setdefault("timings", {
                k: round(v, 1) for k, v in (state.get("timings") or {}).items()
            })
            job.status = "cancelled" if job._cancel.is_set() else "done"
            job.percent = job.percent if job.status == "cancelled" else 100.0
            job.publish(progress.Event(
                kind="log",
                message="已按请求停止 (已完成的片段仍会出片)"
                if job.status == "cancelled" else "任务结束",
                detail={"status": job.status},
            ))
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.publish(progress.Event(kind="error", message=job.error,
                                       detail={"traceback": traceback.format_exc()[-2000:]}))
        finally:
            job.finished_at = time.time()
            # 关闭所有订阅, 让 SSE 循环退出
            with job._lock:
                subs = list(job._subs)
            for q in subs:
                try:
                    q.put_nowait({"kind": "__close__"})
                except queue.Full:
                    # 订阅队列满: 直接标记该订阅者可以退出, 由前端按
                    # job_id 重新拉一次快照兜底 (快照里有最终状态)
                    q.queue.clear()


JOBS = JobManager()


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


def _fail(message: str, status: int = 400, **extra: Any) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message, **extra}, status_code=status)


async def api_health(request):
    return _json({
        "ok": True,
        "version": "1.0",
        "root": str(config.ROOT),
        "out_dir": str(config.OUT_DIR),
        "data_dir": str(config.DATA_DIR),
        "ffmpeg": bool(config.FFMPEG),
        "jobs": [j.snapshot() for j in JOBS.all()][-5:],
    })


async def api_meta(request):
    """表单需要的全部元信息 (偏好定义 / 画幅 / 内置 demo / 队列状态)."""
    prefs = store.load_prefs()
    specs = {
        k: {"default": s.default, "type": s.cast.__name__, "desc": s.desc,
            "choices": list(s.choices) if s.choices else None}
        for k, s in store.PREFS.items()
    }
    return _json({
        "ok": True,
        "pref_specs": specs,
        "prefs": prefs,
        "defaults": store.defaults(),
        "aspects": sorted(config.ASPECT_PRESETS),
        "default_demo": str(config.DEFAULT_DEMO),
        "default_demo_exists": config.DEFAULT_DEMO.is_file(),
        "music_dir": media.default_music_dir(),
        "busy": any(j.status in ("queued", "running") for j in JOBS.all()),
        "stats": store.stats(),
    })


async def api_music(request):
    root = request.query_params.get("dir") or media.default_music_dir()
    try:
        files, dirs, used = media.list_music_files(root)
    except ValueError as exc:
        return _fail(str(exc))
    parent = str(Path(used).parent) if Path(used).parent != Path(used) else None
    return _json({"ok": True, "dir": used, "parent": parent,
                  "files": files, "dirs": dirs})


async def api_demos(request):
    extra = request.query_params.get("dir")
    roots = []
    if extra:
        roots.append(Path(extra))
    roots += [config.DEFAULT_DEMO.parent, config.ROOT, Path(r"D:\5E_cs2_demo")]
    return _json({"ok": True, "files": media.list_demos(roots)})


async def api_validate(request):
    """出片前先校验素材, 让错误在点击的瞬间就报出来而不是等 10 秒。"""
    data = await _body(request)
    try:
        music = media.resolve_media(data.get("music_path", ""), media.MUSIC_EXT, what="音乐")
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        return _fail(str(exc))
    demo_raw = (data.get("demo_path") or "").strip()
    if not demo_raw:
        return _json({"ok": True, "music": str(music), "demo": str(config.DEFAULT_DEMO),
                      "demo_from_pref": False})
    try:
        demo = media.resolve_media(demo_raw, media.DEMO_EXT, what="demo")
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        return _fail(str(exc))
    return _json({"ok": True, "music": str(music), "demo": str(demo),
                  "music_size": music.stat().st_size, "demo_size": demo.stat().st_size})


async def api_jobs_create(request):
    data = await _body(request)
    try:
        music = media.resolve_media(data.get("music_path", ""), media.MUSIC_EXT, what="音乐")
    except Exception as exc:
        return _fail(str(exc))
    demo_raw = (data.get("demo_path") or "").strip()
    demo: Path | None = None
    if demo_raw:
        try:
            demo = media.resolve_media(demo_raw, media.DEMO_EXT, what="demo")
        except Exception as exc:
            return _fail(str(exc))

    def _num(key: str, cast, lo, hi):
        raw = data.get(key)
        if raw is None or raw == "":
            return None
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {key} 不是合法数值: {raw!r}")
        if not (lo <= val <= hi):
            raise ValueError(f"参数 {key} 必须在 {lo}~{hi} 之间, 收到 {val}")
        return val

    try:
        params: dict[str, Any] = {
            "music_path": str(music),
            "demo_path": str(demo) if demo else "",
            "aspect": (data.get("aspect") or None) if (data.get("aspect") or None) in config.ASPECT_PRESETS else None,
            "fps": _num("fps", int, 10, 120),
            "max_clips": _num("max_clips", int, 1, 200),
            "max_cards": _num("max_cards", int, 1, 400),
            "pacing": (data.get("pacing") or None)
            if (data.get("pacing") or None) in ("fast", "balanced", "cinematic") else None,
            "use_llm": _bool(data.get("use_llm")),
            "use_prefs": bool(data.get("use_prefs", True)),
            "record": bool(data.get("record", True)),
            "out_dir": (str(data.get("out_dir") or "").strip() or None),
        }
    except ValueError as exc:
        return _fail(str(exc))
    if params["out_dir"]:
        # 输出目录: 允许工作区内的相对路径, 也允许任意绝对路径, 但必须可创建
        od = Path(params["out_dir"]).expanduser()
        if not od.is_absolute():
            od = (config.ROOT / od)
        try:
            od.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return _fail(f"输出目录不可用: {od} ({exc})")
        params["out_dir"] = str(od.resolve())

    job = JOBS.create(params)
    return _json({"ok": True, "job": job.snapshot()}, status=201)


async def api_jobs_list(request):
    return _json({"ok": True, "jobs": [j.snapshot() for j in JOBS.all()]})


async def api_job_get(request):
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return _fail("任务不存在", status=404)
    return _json({"ok": True, "job": job.snapshot(), "events": job.events[-200:]})


async def api_job_cancel(request):
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return _fail("任务不存在", status=404)
    if job.status in ("done", "error", "cancelled"):
        return _json({"ok": True, "job": job.snapshot(), "note": "任务已结束"})
    job._cancel.set()
    job.publish(progress.Event(kind="log", message="收到停止请求, 将在当前片段渲染完后中止"))
    return _json({"ok": True, "job": job.snapshot()})


async def api_job_events(request):
    """SSE 事件流 (支持断线重连: 先重放历史再续播)."""
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return _fail("任务不存在", status=404)
    since = int(request.query_params.get("since") or 0)
    q, replay = job.subscribe()
    done = job.status in ("done", "error", "cancelled")

    async def gen():
        try:
            yield ": connected\n\n"
            sent = 0
            for d in replay:
                if d.get("seq", 0) > since:
                    sent += 1
                    yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
            if done and sent == 0 and len(replay) <= since:
                # 任务已经结束且这条连接不会再有新事件
                yield f"data: {json.dumps({'kind': '__close__'})}\n\n"
                return
            idle = 0.0
            while True:
                if await request.is_disconnected():
                    return
                try:
                    d = q.get_nowait()
                except queue.Empty:
                    idle += 0.5
                    if idle >= 15:
                        idle = 0.0
                        yield ": ping\n\n"          # 心跳, 防代理超时断流
                    await asyncio.sleep(0.5)
                    continue
                idle = 0.0
                if d.get("kind") == "__close__":
                    yield f"data: {json.dumps({'kind': '__close__'})}\n\n"
                    return
                yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
        finally:
            job.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


# ------------------------------------------------------------------
# 产物 / 素材读取
# ------------------------------------------------------------------
async def api_artifact(request):
    """读取工作区内的产物文件 (视频 / 缩略图 / JSON)."""
    rel = request.query_params.get("path") or ""
    try:
        p = media.safe_under(rel, config.ROOT)
    except media.PathNotAllowed as exc:
        return _fail(str(exc), status=403)
    if not p.is_file():
        return _fail(f"文件不存在: {p}", status=404)
    if p.suffix.lower() == ".json":
        return _json(json.loads(p.read_text(encoding="utf-8")))
    media_type = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    }.get(p.suffix.lower(), "application/octet-stream")
    # FileResponse 自带 Range 支持, 视频可以拖动进度条
    return FileResponse(p, media_type=media_type, filename=p.name)


async def api_artifact_text(request):
    """读文本类产物 (日志/EDL 原文), 便于排查."""
    rel = request.query_params.get("path") or ""
    try:
        p = media.safe_under(rel, config.ROOT)
    except media.PathNotAllowed as exc:
        return _fail(str(exc), status=403)
    if not p.is_file():
        return _fail(f"文件不存在: {p}", status=404)
    if p.stat().st_size > 8 * 1024 * 1024:
        return _fail("文件过大, 请直接打开文件", status=413)
    return JSONResponse({"ok": True, "path": str(p),
                         "text": p.read_text(encoding="utf-8", errors="replace")})


async def api_download(request):
    """下载最终成片 (任意路径 -> 文件名), 供历史记录里的旧产物使用."""
    rel = request.query_params.get("path") or ""
    try:
        p = media.safe_under(rel, config.ROOT)
    except media.PathNotAllowed as exc:
        return _fail(str(exc), status=403)
    if not p.is_file():
        return _fail(f"文件不存在: {p}", status=404)
    return FileResponse(p, media_type="video/mp4", filename=p.name)


async def api_outputs(request):
    """列出某个输出目录里的产物 (成片/EDL/分析 JSON/片段)."""
    rel = request.query_params.get("dir") or ""
    try:
        d = media.safe_under(rel, config.ROOT)
    except media.PathNotAllowed as exc:
        return _fail(str(exc), status=403)
    if not d.is_dir():
        return _fail(f"目录不存在: {d}", status=404)
    items = []
    for p in sorted(d.iterdir()):
        try:
            st = p.stat()
        except OSError:
            continue
        items.append({"name": p.name, "path": str(p), "is_dir": p.is_dir(),
                      "size": 0 if p.is_dir() else st.st_size,
                      "mtime": st.st_mtime})
    clips = d / "clips"
    n_clips = len(list(clips.glob("*.mp4"))) if clips.is_dir() else 0
    return _json({"ok": True, "dir": str(d), "items": items, "clips": n_clips})


# ------------------------------------------------------------------
# 偏好 / 画像 / 历史
# ------------------------------------------------------------------
async def api_prefs_get(request):
    prefs = store.load_prefs()
    return _json({"ok": True, "prefs": prefs, "defaults": store.defaults(),
                  "effective": {k: prefs.get(k, s.default) for k, s in store.PREFS.items()},
                  "specs": {k: {"type": s.cast.__name__, "desc": s.desc,
                                "choices": list(s.choices) if s.choices else None,
                                "default": s.default}
                            for k, s in store.PREFS.items()}})


async def api_prefs_set(request):
    data = await _body(request)
    values = data.get("values") or {}
    if not isinstance(values, dict) or not values:
        return _fail("没有要写入的偏好")
    try:
        store.set_prefs(values)
    except (ValueError, KeyError) as exc:
        return _fail(f"偏好写入失败: {exc}")
    return _json({"ok": True, "prefs": store.load_prefs(), "applied": values})


async def api_prefs_reset(request):
    try:
        n = store.clear_prefs()
    except Exception as exc:
        return _fail(f"清空偏好失败: {exc}", status=500)
    return _json({"ok": True, "cleared": n, "prefs": store.load_prefs()})


async def api_profile(request):
    rows = store.load_profile()
    obs = store.list_observations(limit=200)
    by_dim: dict[str, list[dict]] = {}
    for o in obs:
        by_dim.setdefault(o["dimension"], []).append(o)
    return _json({
        "ok": True,
        "dimensions": profile_mod.DIMENSIONS,
        "value_labels": {f"{k[0]}|{k[1]}": v for k, v in profile_mod.VALUE_LABELS.items()},
        # 每个维度的**已知取值** (前端做下拉, 避免手打出画像里查不到的取值)
        "dimension_values": _dimension_values(),
        "profile": rows,
        "profile_dict": store.profile_as_dict(),
        "brief": profile_mod.profile_brief(),
        # describe_* 只接受 db 路径, 不接受 dict —— 传 dict 会 TypeError
        "describe": profile_mod.describe_profile(),
        "observations": obs,
        "by_dimension": by_dim,
        "source_weight": store.SOURCE_WEIGHT,
        "text": profile_mod.format_observations(limit=200),
    })


async def api_profile_rebuild(request):
    try:
        rows = profile_mod.build_profile()
    except Exception as exc:
        return _fail(f"重建画像失败: {exc}", status=500)
    return _json({"ok": True, "rows": len(rows), "profile": store.load_profile()})


async def api_observe(request):
    """手工补一条偏好证据 (相当于 CLI 的 --observe)."""
    data = await _body(request)
    dim = (data.get("dimension") or "").strip()
    val = (data.get("value") or "").strip()
    if dim not in profile_mod.DIMENSIONS:
        return _fail(f"未知维度: {dim} (可用: {list(profile_mod.DIMENSIONS)})")
    if not val:
        return _fail("value 不能为空")
    try:
        store.add_observation(dim, val, source="explicit_set",
                              quote=(data.get("quote") or "").strip() or "由界面手工记录")
        rows = profile_mod.build_profile()
    except Exception as exc:
        return _fail(f"写入证据失败: {exc}", status=500)
    return _json({"ok": True, "rows": len(rows), "profile": store.load_profile()})


async def api_runs(request):
    limit = int(request.query_params.get("limit") or 30)
    runs = store.list_runs(limit=max(1, min(limit, 200)))
    with store.connect() as conn:
        for r in runs:
            row = conn.execute("SELECT COUNT(*) c FROM clips WHERE run_id = ?",
                               (r["id"],)).fetchone()
            r["clip_count"] = int(row["c"]) if row else 0
    return _json({"ok": True, "runs": runs, "stats": store.stats(),
                  "last": store.last_run()})


async def api_run_clips(request):
    run_id = int(request.path_params["run_id"])
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM clips WHERE run_id = ? ORDER BY out_start", (run_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("effects", "tags"):
            if isinstance(d.get(k), str) and d[k]:
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        out.append(d)
    return _json({"ok": True, "clips": out})


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------
async def _body(request) -> dict[str, Any]:
    try:
        raw = await request.body()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _dimension_values() -> dict[str, list[str]]:
    """每个画像维度的已知取值 (来自 VALUE_LABELS), 供前端做下拉."""
    out: dict[str, list[str]] = {}
    for (dim, val) in profile_mod.VALUE_LABELS:
        out.setdefault(dim, []).append(val)
    return out


def _bool(raw: Any) -> bool | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on", "开", "是"):
        return True
    if s in ("0", "false", "no", "off", "关", "否"):
        return False
    raise ValueError(f"无法解析布尔值: {raw!r}")


async def index(request):
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse("<h1>缺少前端资源 cs2clipper/web/index.html</h1>",
                            status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


async def favicon(request):
    return Response(status_code=204)


def create_app() -> Starlette:
    routes = [
        Route("/", index),
        Route("/favicon.ico", favicon),
        Route("/api/health", api_health),
        Route("/api/meta", api_meta),
        Route("/api/music", api_music),
        Route("/api/demos", api_demos),
        Route("/api/validate", api_validate, methods=["POST"]),
        Route("/api/jobs", api_jobs_create, methods=["POST"]),
        Route("/api/jobs", api_jobs_list, methods=["GET"]),
        Route("/api/jobs/{job_id}", api_job_get),
        Route("/api/jobs/{job_id}/events", api_job_events),
        Route("/api/jobs/{job_id}/cancel", api_job_cancel, methods=["POST"]),
        Route("/api/artifact", api_artifact),
        Route("/api/artifact/text", api_artifact_text),
        Route("/api/download", api_download),
        Route("/api/outputs", api_outputs),
        Route("/api/prefs", api_prefs_get),
        Route("/api/prefs/set", api_prefs_set, methods=["POST"]),
        Route("/api/prefs/reset", api_prefs_reset, methods=["POST"]),
        Route("/api/profile", api_profile),
        Route("/api/profile/rebuild", api_profile_rebuild, methods=["POST"]),
        Route("/api/observe", api_observe, methods=["POST"]),
        Route("/api/runs", api_runs),
        Route("/api/runs/{run_id}/clips", api_run_clips),
    ]
    app = Starlette(routes=routes)
    # 前端只认 JSON: 把"请求体不合法"这类解析错误翻成 400, 而不是一个 HTML 500
    async def on_value_error(request, exc):
        return _fail(f"请求无法处理: {exc}", status=400)

    async def on_error(request, exc):
        traceback.print_exc()
        return _fail(f"服务器内部错误: {type(exc).__name__}: {exc}", status=500)

    app.add_exception_handler(ValueError, on_value_error)
    app.add_exception_handler(Exception, on_error)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CS2 自动剪辑 Web 界面 (本地服务)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址 (默认只监听本机; 改成 0.0.0.0 会暴露到局域网)")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    ap.add_argument("--reload", action="store_true", help="改代码自动重启 (开发用)")
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}/"
    print("=" * 64)
    print("CS2 自动剪辑 · Web 界面")
    print(f"  地址: {url}")
    print(f"  工作区: {config.ROOT}")
    print(f"  ffmpeg: {config.FFMPEG or '未找到 (出片会失败)'}")
    print("  按 Ctrl+C 停止")
    print("=" * 64)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    if args.reload:
        uvicorn.run("cs2clipper.webapp:create_app", factory=True,
                    host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
