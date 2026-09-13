"""完整流水线 (LangGraph 编排).

    analyze_music ──┐
                    ├─→ plan ─→ render ─→ compose ─→ END
    analyze_demo  ──┘

为什么用 LangGraph 而不是一个普通函数串下来:
    * 每个阶段是可以单独重跑的节点, 中间产物落盘, 调试时不用从头再来
    * 后续要加"人审"环节 (比如让用户挑片段) 时, 直接插入 interrupt 即可
    * 与项目原有的 agent 技术栈一致 (langgraph / langchain)

使用:
    python -m cs2clipper.pipeline --music song.mp3 --demo match.dem
    python -m cs2clipper.pipeline --music song.mp3          # 用默认 demo
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import compose, config, demo as demo_mod, music as music_mod, planner as planner_mod, profile as profile_mod, progress, radar as radar_mod, store
from . import mapview as mv


# ------------------------------------------------------------------
# 状态
# ------------------------------------------------------------------
def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    """timings 的 reducer.

    analyze_music 与 analyze_demo 在图里是**并行**的, 两个节点都会写
    timings。LangGraph 的默认通道只允许"一步一个写入", 所以必须给这个键
    声明一个 reducer 来合并, 否则会抛 InvalidUpdateError。
    """
    return {**(left or {}), **(right or {})}


class PipelineState(TypedDict, total=False):
    # 输入
    demo_path: str
    music_path: str
    out_dir: str
    fps: int
    aspect: str
    max_cards: int
    max_clips: int
    use_llm: bool
    verbose: bool
    prefs: dict[str, Any]          # 合并后的完整偏好 (含画面类, 供渲染使用)
    profile: dict[str, str]        # 从对话推断的用户画像 (进提示词)

    # 阶段产物
    music_analysis: Any          # MusicAnalysis
    demo_analysis: dict          # analyze_demo 的返回值
    highlights: Any              # list[HighlightCard]
    edl: Any                     # EDL
    video_path: str
    # 渲染阶段产出的待拼接段落 (LangGraph 只保存 schema 里声明过的键,
    # 用未声明的私有键会被静默丢掉)
    concat_items: list[Any]
    timings: Annotated[dict[str, float], _merge_dicts]
    # 进度事件总线 (progress.Bus). 带下划线前缀, 只在进程内用, 不进任何产物。
    _bus: Any
    # 取消判定回调 (Web 界面"停止"按钮用): 返回 True 时在下一段之前中止。
    _should_cancel: Any


# ------------------------------------------------------------------
# 节点
# ------------------------------------------------------------------
def node_analyze_music(state: PipelineState) -> dict[str, Any]:
    t0 = time.time()
    bus = progress.from_state(state)
    bus.stage_start("analyze_music", f"分析音乐: {Path(state['music_path']).name}")
    a = music_mod.analyze_music(state["music_path"])
    msg = (f"音乐分析完成 {time.time()-t0:.1f}s: BPM {a.bpm:.1f}, "
           f"{len(a.segments)} 个段落, {len(a.beat_times)} 个拍点")
    if state.get("verbose"):
        print(f"[1/5] {msg}")
    bus.stage_done(
        "analyze_music", msg,
        bpm=round(a.bpm, 1), segments=len(a.segments), beats=len(a.beat_times),
        duration=round(a.duration, 1),
        has_climax=bool(a.extra.get("has_climax")),
    )
    return {
        "music_analysis": a,
        "timings": {**state.get("timings", {}), "analyze_music": time.time() - t0},
    }


def node_analyze_demo(state: PipelineState) -> dict[str, Any]:
    t0 = time.time()
    prefs = state.get("prefs") or store.defaults()
    bus = progress.from_state(state)
    bus.stage_start("analyze_demo", f"解析 demo: {Path(state['demo_path']).name}")
    res = demo_mod.analyze_demo(
        state["demo_path"],
        min_score=float(prefs.get("min_score", 25.0)),
        max_cards=state.get("max_cards", 40),
        with_utility=True,
    )
    cards = demo_mod.cards_from_dicts(res["highlights"])
    # 道具抽取部分失败时必须出声 —— 否则成片里少了烟雾/燃烧而无人知晓
    util_errors = res.get("utility_errors") or []
    if util_errors:
        warn = f"道具抽取部分失败 (成片会缺少对应叠加层): {'; '.join(util_errors)}"
        if state.get("verbose"):
            print(f"      [warn] {warn}")
        bus.log(warn, stage="analyze_demo")
    msg = (f"demo 分析完成 {time.time()-t0:.1f}s: 地图 {res['map_name']}, "
           f"{res['round_count']} 回合, {len(cards)} 条亮点素材")
    if state.get("verbose"):
        print(f"[2/5] {msg}")
    bus.stage_done(
        "analyze_demo", msg,
        map_name=res["map_name"], rounds=res["round_count"], cards=len(cards),
        clutches=res.get("clutch_count", 0),
        skipped=res.get("skipped_events", {}),
        roster_error=res.get("roster_error"),
        tickrate=res.get("tickrate"),
    )
    return {
        "demo_analysis": {k: v for k, v in res.items() if k != "highlights"},
        "highlights": cards,
        "timings": {**state.get("timings", {}), "analyze_demo": time.time() - t0},
    }


def node_plan(state: PipelineState) -> dict[str, Any]:
    t0 = time.time()
    bus = progress.from_state(state)
    bus.stage_start("plan", "按音乐情绪编排剪辑点")
    edl = planner_mod.plan_edit(
        state["music_analysis"],
        state["highlights"],
        use_llm=state.get("use_llm", True),
        max_clips=state.get("max_clips", 22),
        fps=state.get("fps", config.FPS),
        verbose=state.get("verbose", False),
        # 用户偏好进提示词 (节奏/段数上限/单条时长区间), 同时贯穿展开器
        prefs=state.get("prefs") or {},
        # 从历史对话推断的画像 (带不确定性) 只进提示词当口味参考
        profile=state.get("profile") or {},
    )
    planner_name = edl.meta.get("planner", "?")
    notes = edl.meta.get("notes") or []
    msg = (f"编排完成 {time.time()-t0:.1f}s ({planner_name}): "
           f"{len(edl.clips)} 条剪辑, 覆盖 {edl.total_duration:.1f}s / "
           f"{edl.music_duration:.1f}s")
    if state.get("verbose"):
        print(f"[3/5] {msg}")
        for n in notes[:5]:
            print(f"      注: {n}")
    for n in notes:
        bus.log(f"注: {n}", stage="plan")
    cov = edl.meta.get("coverage_warning")
    if cov:
        bus.log(f"[warn] {cov}", stage="plan")
    bus.stage_done(
        "plan", msg, planner=planner_name, clips=len(edl.clips),
        covered=round(edl.total_duration, 1),
        music_duration=round(edl.music_duration, 1),
        coverage_warning=cov,
        notes=notes,
    )
    return {
        "edl": edl,
        "timings": {**state.get("timings", {}), "plan": time.time() - t0},
    }


def node_render(state: PipelineState) -> dict[str, Any]:
    """渲染每一段并编码成 clip mp4."""
    t0 = time.time()
    edl = state["edl"]
    verbose = state.get("verbose", False)
    bus = progress.from_state(state)
    bus.stage_start("render", f"准备渲染 {len(edl.clips)} 段画面")
    prefs = state.get("prefs") or store.defaults()
    out_dir = Path(state["out_dir"])
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # 复用已解析的 demo: 注意 analyze_demo 内部用的是 lru_cache, 这里再取一次
    # 通常是命中缓存 (同一路径), 不会重新解析 10 秒。
    dem = demo_mod.load_demo(state["demo_path"], with_ticks=True)
    table = radar_mod.build_tick_table(dem)
    by_id = {c.id: c for c in state["highlights"]}

    # 画幅: 地图区始终是正方形 (视口尺寸 = 地图区边长), 画布可以更大
    layout = radar_mod.Layout.from_config(state.get("aspect"))
    places = mv.collect_places(dem, [k for c in state["highlights"] for k in c.kills])
    map_name = state["demo_analysis"].get("map_name", "")

    # 全片兜底视口 (逐段取景退化时用) + 区域标签集合
    ticks_lo = min(c.src_start_tick for c in edl.clips)
    ticks_hi = max(c.src_end_tick for c in edl.clips)
    xs_all, ys_all = [], []
    for i in range(len(table.names)):
        sl = table.slice_at(i, ticks_lo, ticks_hi)
        if sl.stop > sl.start:
            xs_all.append(table.xs[i][sl])
            ys_all.append(table.ys[i][sl])
    import numpy as np

    xs = np.concatenate(xs_all) if xs_all else np.array([0.0])
    ys = np.concatenate(ys_all) if ys_all else np.array([0.0])
    vp_fallback = mv.Viewport.from_positions(xs, ys, size=layout.map_size)

    # 模块级表对象, 供 Layout 排序等复用
    segs = state["music_analysis"].segments
    if verbose:
        print(
            f"[4/5] 渲染 {len(edl.clips)} 段 "
            f"画布 {layout.width}x{layout.height} (地图 {layout.map_size}px, "
            f"面板={layout.panel}, 逐段跟拍)..."
        )

    items: list[compose.ConcatItem] = []
    zooms: list[float] = []
    should_cancel = state.get("_should_cancel")
    for ci, clip in enumerate(edl.clips, start=1):
        # 取消检查点放在**每段开始之前**: 一段的渲染是原子的, 中途打断会留下
        # 半个 mp4。停在这里最坏只损失当前这一段的几秒。
        if callable(should_cancel) and should_cancel():
            bus.log(f"已停止: 完成 {len(items)}/{len(edl.clips)} 段后中止", stage="render")
            break
        card = by_id.get(clip.highlight_id)
        if card is None:
            continue
        # 画面类偏好作为基础 effects, 再由 EDL 里逐段的 effects 覆盖
        base_effects = store.effect_overrides(prefs)
        merged_effects = {**base_effects, **(clip.effects or {})}
        style = radar_mod.RenderStyle.from_effects(merged_effects)
        renderer = radar_mod.RadarRenderer(
            dem, table, vp_fallback,
            style=style,
            layout=layout,
            title=map_name,
        )

        # --- P0 逐段跟拍: 用本段实际参与的位置重新取景 ---
        seg = segs[clip.music_segment] if 0 <= clip.music_segment < len(segs) else None
        if seg is not None:
            # 越激烈的段落越拉近; 基础缩放与斜率都可由偏好调整
            zoom = float(prefs.get("zoom_base", 1.0)) + float(
                prefs.get("zoom_per_arousal", 0.45)
            ) * max(min(seg.arousal, 1.0), 0.0)
        else:
            zoom = float(prefs.get("zoom_base", 1.0))
        zoom *= float(clip.effects.get("zoom", 1.0) or 1.0)
        clip_vp = renderer.clip_viewport(
            demo_mod.HighlightCard(
                id=card.id, round_num=card.round_num, player=card.player,
                player_side=card.player_side,
                start_tick=clip.src_start_tick, end_tick=clip.src_end_tick,
                kills=card.kills, score=card.score, tags=card.tags,
            ),
            zoom=zoom,
            clamp_to=vp_fallback,      # 镜头中心不越出全片活动范围
        )
        renderer.rebase(clip_vp, places, map_name)
        zooms.append(clip_vp.span_x)

        # 用裁剪后的源区间渲染: 直接移动 tick 窗口
        sub = demo_mod.HighlightCard(
            id=card.id,
            round_num=card.round_num,
            player=card.player,
            player_side=card.player_side,
            start_tick=clip.src_start_tick,
            end_tick=clip.src_end_tick,
            kills=[k for k in card.kills if clip.src_start_tick <= k.tick <= clip.src_end_tick],
            score=card.score,
            tags=card.tags,
            places=card.places,
            utility=[
                u for u in card.utility
                if u["end_tick"] >= clip.src_start_tick and u["start_tick"] <= clip.src_end_tick
            ],
            round_winner=card.round_winner,
            round_reason=card.round_reason,
        )

        frames = renderer.render_clip(
            sub,
            fps=state.get("fps", config.FPS),
            speed=clip.speed,
            places=places,
            map_name=state["demo_analysis"].get("map_name", ""),
            score_text=_score_text(state["demo_analysis"], card.round_num),
            progress=(ci, len(edl.clips)),
        )
        out_mp4 = clips_dir / f"clip_{ci:03d}.mp4"
        raw_mp4 = clips_dir / f"_raw_{ci:03d}.mp4"
        compose.render_clip_video(
            frames,
            raw_mp4,
            fps=state.get("fps", config.FPS),
            duration=clip.duration,
            size=(layout.width, layout.height),
        )
        # 规整到精确时长: 慢放段生成的帧数偏少, 不补的话视频轨会比音乐轨短
        compose.finalize_clip(
            raw_mp4, out_mp4,
            target_duration=clip.duration,
            size=(layout.width, layout.height),
        )
        items.append(
            compose.ConcatItem(
                path=out_mp4,
                duration=clip.duration,
                transition=clip.transition,
                transition_duration=clip.transition_duration,
            )
        )
        if verbose:
            print(f"      clip {ci:03d}/{len(edl.clips)}  {clip.duration:5.2f}s  "
                  f"取景 {clip_vp.span_x:5.0f}u  {card.id[:30]}")
        bus.progress(
            "render", ci / max(len(edl.clips), 1),
            f"clip {ci:03d}/{len(edl.clips)} {clip.duration:.2f}s {card.id[:28]}",
            clip=ci, total=len(edl.clips), clip_duration=round(clip.duration, 2),
            span=round(clip_vp.span_x), highlight_id=card.id,
            player=card.player, round_num=card.round_num,
        )

    if verbose and zooms:
        uniq = len({round(z) for z in zooms})
        print(f"      渲染总耗时 {time.time()-t0:.1f}s | "
              f"逐段取景 {min(zooms):.0f}~{max(zooms):.0f} units ({uniq} 种不同跨度)")
    bus.stage_done(
        "render",
        f"渲染完成 {time.time()-t0:.1f}s: {len(items)} 段"
        + (f", 取景 {min(zooms):.0f}~{max(zooms):.0f}u" if zooms else ""),
        clips=len(items),
        span_min=round(min(zooms)) if zooms else None,
        span_max=round(max(zooms)) if zooms else None,
    )
    return {
        "timings": {**state.get("timings", {}), "render": time.time() - t0},
        "concat_items": items,
    }


def node_compose(state: PipelineState) -> dict[str, Any]:
    t0 = time.time()
    bus = progress.from_state(state)
    items = state.get("concat_items") or []
    if not items:
        raise RuntimeError("没有可拼接的片段 (渲染阶段失败?)")
    bus.stage_start("compose", f"拼接 {len(items)} 段并铺音乐")
    out_dir = Path(state["out_dir"])
    final = out_dir / "final.mp4"
    compose.concat_clips(
        items,
        final,
        music_path=state["music_path"],
        total_duration=state["edl"].total_duration,
    )
    dur = compose.probe_duration(final)
    size_mb = final.stat().st_size / 1024 / 1024
    msg = f"合成完成 {time.time()-t0:.1f}s: {final.name} {dur:.1f}s, {size_mb:.1f}MB"
    if state.get("verbose"):
        print(f"[5/5] {msg}")
    bus.stage_done("compose", msg, video_path=str(final), duration=round(dur, 2),
                   size_mb=round(size_mb, 2))
    return {
        "video_path": str(final),
        "timings": {**state.get("timings", {}), "compose": time.time() - t0},
    }


def _score_text(demo_analysis: dict, round_num: int) -> str | None:
    """按回合号累计比分 (用于 HUD 右上角)."""
    rounds = demo_analysis.get("rounds") or []
    t = ct = 0
    for r in rounds:
        if r["round_num"] >= round_num:
            break
        if r.get("winner") == "t":
            t += 1
        elif r.get("winner") == "ct":
            ct += 1
    return f"{t} - {ct}"


def node_save_artifacts(state: PipelineState) -> dict[str, Any]:
    bus = progress.from_state(state)
    bus.stage_start("save_artifacts", "保存中间产物")
    out_dir = Path(state["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    music_mod.save_analysis(state["music_analysis"], out_dir / "music_analysis.json")
    demo_mod.save_demo_analysis(
        {**state["demo_analysis"], "highlights": [c.to_dict() for c in state["highlights"]]},
        out_dir / "demo_analysis.json",
    )
    state["edl"].save(out_dir / "edl.json")
    bus.stage_done("save_artifacts", f"中间产物已写入 {out_dir.name}/",
                   edl_json="edl.json", music_json="music_analysis.json",
                   demo_json="demo_analysis.json")
    return {}


# ------------------------------------------------------------------
# 构图
# ------------------------------------------------------------------
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("analyze_music", node_analyze_music)
    g.add_node("analyze_demo", node_analyze_demo)
    g.add_node("plan", node_plan)
    g.add_node("save_artifacts", node_save_artifacts)
    g.add_node("render", node_render)
    g.add_node("compose", node_compose)

    g.add_edge(START, "analyze_music")
    g.add_edge(START, "analyze_demo")
    g.add_edge("analyze_music", "plan")
    g.add_edge("analyze_demo", "plan")
    g.add_edge("plan", "save_artifacts")
    g.add_edge("save_artifacts", "render")
    g.add_edge("render", "compose")
    g.add_edge("compose", END)
    return g.compile()


def resolve_params(
    explicit: dict[str, Any] | None = None,
    *,
    use_prefs: bool = True,
) -> dict[str, Any]:
    """合并参数: 显式传参 > 已存偏好 > 内置默认.

    这是偏好能生效的关键: CLI 的所有偏好类参数默认值都是 `None` (而不是内置
    默认值), 这样才分得清"用户没写"和"用户写了恰好等于默认值"。用户没写的
    位置由已存偏好补上, 再没有才用内置默认。
    """
    prefs = store.load_prefs() if use_prefs else store.defaults()
    resolved = dict(prefs)
    for key, val in (explicit or {}).items():
        if key in store.PREFS and val is not None:
            resolved[key] = val
    return resolved


def run(
    music_path: str | Path,
    demo_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    fps: int | None = None,
    aspect: str | None = None,
    max_cards: int | None = None,
    max_clips: int | None = None,
    use_llm: bool | None = None,
    pacing: str | None = None,
    verbose: bool = True,
    use_prefs: bool = True,
    record: bool = True,
    on_event: progress.Emitter | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> PipelineState:
    """跑完整流水线, 返回最终 state.

    偏好类参数 (fps/aspect/max_clips/max_cards/use_llm) 传 None 表示"用已存偏好";
    显式传值则覆盖偏好。画面类偏好 (镜头缩放/特效开关) 会作为每段的基础 effects
    传给渲染器, 由 EDL 里逐段的 effects 覆盖。

    `on_event` 是可选的进度回调 (见 progress.py): Web 界面用它做实时进度条,
    CLI 不用。回调抛异常会被吞掉, 绝不影响出片。
    `should_cancel` 是可选的取消判定: 返回 True 时在渲染的下一段之前中止,
    已完成的片段会正常拼接出片 (而不是整次失败)。
    """
    bus = progress.Bus(on_event)
    params = resolve_params(
        {
            "fps": fps,
            "aspect": aspect,
            "max_cards": max_cards,
            "max_clips": max_clips,
            "use_llm": use_llm,
            "pacing": pacing,
        },
        use_prefs=use_prefs,
    )
    fps = int(params["fps"])
    aspect = str(params["aspect"])
    max_cards = int(params["max_cards"])
    max_clips = int(params["max_clips"])
    use_llm = bool(params["use_llm"])
    pacing = str(params.get("pacing", "balanced"))

    music_path = Path(music_path)
    # demo: 显式传参 > 已存偏好 default_demo > 项目内置示例
    if demo_path is None:
        prefs_demo = str(params.get("default_demo", "") or "").strip()
        demo_path = Path(prefs_demo) if prefs_demo else config.DEFAULT_DEMO
    else:
        demo_path = Path(demo_path)
    if out_dir is None:
        root = config.ROOT / str(params.get("out_root", "out"))
        out_dir = root / music_path.stem
    out_dir = Path(out_dir)
    # 立刻转成绝对路径: 下游 (渲染/拼接) 会在 work/ 下写中间文件, 而 ffmpeg 的
    # concat demuxer 以**清单文件所在目录**为相对路径基准 —— 相对 out_dir 会让
    # clip 路径被解析成 work/out/... 而打不开。另外进程内会 chdir 的地方很多,
    # 绝对路径能保证"日志里打印的路径"和"实际写入的路径"始终是同一个。
    out_dir = out_dir.resolve()

    if not music_path.is_file():
        raise FileNotFoundError(f"音乐文件不存在: {music_path}")
    if not demo_path.is_file():
        raise FileNotFoundError(f"demo 文件不存在: {demo_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 登记运行 (失败也不该影响出片)
    run_id: int | None = None
    t0 = time.time()
    if record:
        try:
            run_id = store.start_run(
                str(demo_path), str(music_path), str(out_dir),
                {**params, "fps": fps, "aspect": aspect, "use_llm": use_llm,
                 "pacing": pacing},
            )
        except Exception:
            run_id = None

    app = build_graph()
    init: PipelineState = {
        "demo_path": str(demo_path),
        "music_path": str(music_path),
        "out_dir": str(out_dir),
        "fps": fps,
        "aspect": aspect,
        "max_cards": max_cards,
        "max_clips": max_clips,
        "use_llm": use_llm,
        "verbose": verbose,
        "prefs": params,
        "profile": profile_mod.profile_brief() if use_prefs else {},
        "timings": {},
        "_bus": bus,
        "_should_cancel": should_cancel,
    }
    bus.log(
        f"开始: {music_path.name} + {demo_path.name} → {out_dir}",
        out_dir=str(out_dir),
        aspect=aspect, fps=fps, max_clips=max_clips, max_cards=max_cards,
        use_llm=use_llm, pacing=pacing,
    )
    try:
        state = app.invoke(init)
    except Exception as exc:
        bus.error(f"{type(exc).__name__}: {exc}")
        raise

    # 回填运行结果
    if record and run_id is not None:
        try:
            video = Path(state.get("video_path", ""))
            edl = state.get("edl")
            meta = getattr(edl, "meta", {}) or {}
            store.finish_run(
                run_id,
                map_name=state.get("demo_analysis", {}).get("map_name"),
                aspect=aspect, fps=fps, use_llm=use_llm,
                planner=str(meta.get("planner", "")) or None,
                clips=len(edl.clips) if edl else None,
                duration_sec=edl.total_duration if edl else None,
                video_path=str(video) if video.is_file() else None,
                video_bytes=video.stat().st_size if video.is_file() else None,
                elapsed_sec=time.time() - t0,
            )
            if edl:
                # EDLClip 本身不含 player/round_num/score/tags (那是素材的属性),
                # 直接从 to_dict() 取会全是 None, 运行记录就失去价值。这里按
                # highlight_id 回填素材信息。
                by_id = {c.id: c for c in state.get("highlights") or []}
                rows = []
                for cl in edl.clips:
                    d = cl.to_dict()
                    card = by_id.get(cl.highlight_id)
                    if card is not None:
                        d["player"] = card.player
                        d["round_num"] = card.round_num
                        d["score"] = card.score
                        d["tags"] = list(card.tags)
                    rows.append(d)
                store.record_clips(run_id, rows)
        except Exception as e:
            # 记录失败不该影响出片, 但**绝不能完全静默** —— 否则运行记录功能
            # 悄悄失效 (表为空、字段全 None), 用户以为在记录其实没有。
            state.setdefault("warnings", []).append(
                f"运行记录写入失败 (出片不受影响): {type(e).__name__}: {e}"
            )
            if verbose:
                print(f"[warn] 运行记录写入失败: {type(e).__name__}: {e}")
    bus.done(
        "出片完成",
        video_path=str(state.get("video_path", "")),
        duration=round(state["edl"].total_duration, 2) if state.get("edl") else None,
        clips=len(state["edl"].clips) if state.get("edl") else None,
        timings={k: round(v, 1) for k, v in (state.get("timings") or {}).items()},
        warnings=state.get("warnings") or [],
    )
    return state


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    prefs_now = store.load_prefs()
    ap = argparse.ArgumentParser(
        description="CS2 demo + 音乐 自动剪辑 (2D 雷达可视化)",
        epilog=(
            "偏好会存到 data/clipper.db。命令行显式传的参覆盖已存偏好。\n"
            "例如: --set max_clips 30 aspect tall   然后以后直接 --music 即可"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--music", required=False, help="音乐文件 (mp3/wav/flac...)")
    ap.add_argument("--demo", default=None,
                    help="CS2 demo 路径 (默认用已存偏好 default_demo, 再退回内置示例)")
    ap.add_argument("--out", default=None, help="输出目录")

    # --- 偏好类参数: 默认 None, 以便区分"没写"与"写了默认值" ---
    ap.add_argument("--fps", type=int, default=None,
                    help=f"输出帧率 (偏好值: {prefs_now['fps']})")
    ap.add_argument(
        "--aspect", choices=sorted(config.ASPECT_PRESETS), default=None,
        help=f"画幅: wide=1920x1080 (B站), tall=1080x1920 (抖音), square (偏好值: {prefs_now['aspect']})",
    )
    ap.add_argument("--max-clips", type=int, default=None,
                    help=f"剪辑段数上限 (偏好值: {prefs_now['max_clips']})")
    ap.add_argument("--max-cards", type=int, default=None,
                    help=f"亮点素材池大小 (偏好值: {prefs_now['max_cards']})")
    ap.add_argument(
        "--pacing", choices=("fast", "balanced", "cinematic"), default=None,
        help=f"剪辑节奏 (偏好值: {prefs_now['pacing']})",
    )
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM, 用确定性编排")
    ap.add_argument("--use-llm", action="store_true", help="强制使用 LLM 编排")

    # --- 偏好管理 ---
    ap.add_argument("--set", nargs="+", metavar="KEY VALUE",
                    help="保存偏好 (可写多组), 例如: --set aspect tall max_clips 30")
    ap.add_argument("--show-prefs", action="store_true", help="打印当前偏好后退出")
    ap.add_argument("--reset-prefs", nargs="*", metavar="KEY",
                    help="清除偏好 (不带键则清空全部) 后退出")
    ap.add_argument("--show-runs", nargs="?", type=int, const=10, default=None,
                    metavar="N", help="列出最近 N 次运行记录后退出")
    ap.add_argument("--show-stats", action="store_true", help="打印存储统计后退出")
    ap.add_argument("--show-profile", action="store_true",
                    help="打印从对话推断的用户画像后退出")
    ap.add_argument("--show-evidence", nargs="?", type=int, const=40, default=None,
                    metavar="N", help="打印最近 N 条偏好证据后退出")
    ap.add_argument("--observe", nargs=3, metavar=("DIM", "VALUE", "QUOTE"),
                    help="手工记录一条偏好证据后退出")
    ap.add_argument("--rebuild-profile", action="store_true",
                    help="由现有证据重新汇总画像后退出")
    ap.add_argument("--forget", nargs="*", metavar="DIM",
                    help="删除某些维度的证据与结论 (不带则清空全部画像) 后退出")
    ap.add_argument("--no-prefs", action="store_true", help="本次忽略已存偏好, 全用默认值")
    ap.add_argument("--no-record", action="store_true", help="本次不写入运行记录")

    ap.add_argument(
        "--demo-track",
        action="store_true",
        help="没有音乐素材时, 生成一首合成测试曲来走通全流程",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    # ---------- 偏好管理子命令 ----------
    if args.set:
        if len(args.set) % 2 != 0:
            ap.error("--set 需要成对的 KEY VALUE, 例如: --set aspect tall")
        pairs = dict(zip(args.set[0::2], args.set[1::2]))
        try:
            updated = store.set_prefs(pairs)
        except (KeyError, ValueError) as e:
            ap.error(str(e))
            return 2
        print("已保存偏好:")
        for k in pairs:
            print(f"  {k} = {updated[k]}")
        print(f"\n存储位置: {store.db_path()}")
        return 0

    if args.reset_prefs is not None:
        n = store.clear_prefs(args.reset_prefs or None)
        print(f"已清除 {n} 条偏好" + (f" ({', '.join(args.reset_prefs)})"
                                     if args.reset_prefs else " (全部)"))
        return 0

    if args.show_prefs:
        print(f"当前偏好 (存储: {store.db_path()}):\n")
        print(store.describe_prefs(store.load_prefs()))
        return 0

    if args.show_runs is not None:
        runs = store.list_runs(limit=args.show_runs)
        if not runs:
            print("还没有运行记录")
            return 0
        print(f"最近 {len(runs)} 次运行:\n")
        print(f"  {'id':<5}{'时间':<21}{'地图':<14}{'画幅':<8}{'段数':>5}"
              f"{'时长':>8}{'耗时':>8}  成片")
        for r in runs:
            started = str(r["started_at"] or "")[:19].replace("T", " ")
            dur = f"{r['duration_sec']:.0f}s" if r["duration_sec"] else "-"
            el = f"{r['elapsed_sec']:.0f}s" if r["elapsed_sec"] else "-"
            vid = Path(r["video_path"]).name if r["video_path"] else "(未完成)"
            print(f"  {r['id']:<5}{started:<21}{str(r['map_name'] or '-'):<14}"
                  f"{str(r['aspect'] or '-'):<8}{r['clips'] or 0:>5}{dur:>8}{el:>8}  {vid}")
        return 0

    if args.show_profile:
        print(f"用户画像 (由对话与行为推断, 存储: {store.db_path()}):\n")
        print(profile_mod.describe_profile())
        s = store.stats()
        print(f"\n  依据 {s['observations']} 条证据汇总出 {s['profile_dims']} 个维度")
        print(f"  重建画像: --rebuild-profile    查看证据: --show-evidence")
        print(f"  注意: 画像只作为提示词上下文, 不会自动改变出片参数。")
        return 0

    if args.show_evidence is not None:
        print(profile_mod.format_observations(limit=args.show_evidence))
        return 0

    if args.observe:
        dim, value, quote = args.observe
        try:
            store.add_observation(dim, value, source="explicit_set", quote=quote)
        except Exception as e:
            ap.error(str(e))
            return 2
        profile_mod.build_profile()
        print(f"已记录证据: {dim} = {value}")
        print(f"  依据: 「{quote[:60]}」")
        print("\n当前画像:")
        print(profile_mod.describe_profile())
        return 0

    if args.rebuild_profile:
        rows = profile_mod.build_profile()
        print(f"已由 {store.stats()['observations']} 条证据重建画像, "
              f"{len(rows)} 个维度:\n")
        print(profile_mod.describe_profile())
        return 0

    if args.forget is not None:
        dims = args.forget or None
        n_obs = store.clear_observations and store.clear_observations(
            dims[0] if dims and len(dims) == 1 else None)
        # 多维度时逐个删
        if dims and len(dims) > 1:
            n_obs = sum(store.clear_observations(d) for d in dims)
        if not dims:
            store.save_profile([])
        rows = profile_mod.build_profile()
        print(f"已删除 {n_obs} 条证据"
              + (f" (维度: {', '.join(dims)})" if dims else " (全部)"))
        print(f"剩余画像 {len(rows)} 个维度")
        return 0

    if args.show_stats:
        s = store.stats()
        print("存储统计:")
        print(f"  数据库      {s['db_path']}  ({s['db_bytes'] / 1024:.1f} KB)")
        print(f"  偏好条目    {s['prefs']} / {len(store.PREFS)} 项可配置")
        print(f"  完成运行    {s['runs']} 次")
        print(f"  剪辑记录    {s['clips']} 段")
        print(f"  平均耗时    {s['avg_elapsed_sec']:.1f}s")
        print(f"  成片总大小  {s['total_video_bytes'] / 1024 / 1024:.1f} MB")
        return 0

    # ---------- 出片 ----------
    music = args.music
    if args.demo_track and not music:
        music = str(config.WORK_DIR / "test_track.wav")
        if not Path(music).is_file():
            music_mod.make_test_track(music)
            print(f"[生成测试音轨] {music}")
    if not music:
        ap.error("必须提供 --music, 或使用 --demo-track 生成测试音轨")

    use_llm_explicit: bool | None = None
    if args.no_llm:
        use_llm_explicit = False
    elif args.use_llm:
        use_llm_explicit = True

    t0 = time.time()
    state = run(
        music,
        args.demo,
        out_dir=args.out,
        fps=args.fps,
        aspect=args.aspect,
        max_cards=args.max_cards,
        max_clips=args.max_clips,
        use_llm=use_llm_explicit,
        pacing=args.pacing,
        verbose=not args.quiet,
        use_prefs=not args.no_prefs,
        record=not args.no_record,
    )
    print(f"\n完成, 总耗时 {time.time()-t0:.1f}s")
    print(f"成片: {state.get('video_path')}")
    print(f"中间产物: {state['out_dir']}  (edl.json / music_analysis.json / demo_analysis.json)")
    print(f"运行记录已写入 {store.db_path()} (--show-runs 查看)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
