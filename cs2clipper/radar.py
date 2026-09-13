"""Stage 4 —— 雷达帧渲染.

把 demo 的逐 tick 数据画成"战术回放"风格的动画帧:
    * 玩家圆点: 半径随血量收缩, CT 蓝 / T 黄
    * 移动轨迹: 最近 ~1.5 秒的淡出拖尾
    * 击杀特效: 红色 X + 扩散圆环 + 击杀者连线
    * 道具: 烟雾 (灰雾) / 燃烧 (橙红) / 闪光 (白爆)
    * HUD: 回合、比分、当前主角、镜头信息

性能要点:
    * 底图只画一次 (PIL Image), 每帧 copy() 后叠加动态层
    * 逐 tick 数据预先转成 numpy 数组, 用 searchsorted 取时间片, 不做逐帧过滤
    * 所有几何计算都在 numpy 上批量做, 避免 Python 循环
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config
from . import mapview as mv
from .demo import HighlightCard, KillEvent, TICKRATE


# ------------------------------------------------------------------
# 画布布局 (画幅预设)
# ------------------------------------------------------------------
@dataclass
class Layout:
    """成片画布布局.

    只有方形时地图区 = 整张画布; 横幅/竖幅时地图区居中, 旁边留出信息面板
    (放标题、比分、回合数、击杀计数)。面板不是"黑边", 而是刻意留白 ——
    否则直接拿正方形视频加黑边推到 B 站/抖音会很难看。
    """

    width: int
    height: int
    map_size: int
    panel: str          # "none" | "right" | "bottom"
    hud_bar: int = 0    # 地图下沿的 HUD 条高度 (竖屏用; 必须计入面板起点)

    @property
    def map_origin(self) -> tuple[int, int]:
        """地图区左上角在画布中的坐标."""
        if self.panel == "right":
            # 地图靠左居中对齐, 右侧留面板
            return (max((self.width - self.map_size) // 2 - int(self.width * 0.02), 0),
                    (self.height - self.map_size) // 2)
        if self.panel == "bottom":
            return ((self.width - self.map_size) // 2,
                    max(int(self.height * 0.02), 0))
        return ((self.width - self.map_size) // 2, (self.height - self.map_size) // 2)

    @property
    def panel_rect(self) -> tuple[int, int, int, int] | None:
        """信息面板矩形 (x0, y0, x1, y1); 没有面板时返回 None.

        注意竖屏分支: 面板横向范围必须对齐**地图区**的实际左右边界
        (mx .. mx+map_size), 而不是地图边长本身。当画布比地图宽时
        (例如 1080 宽画布 + 1024 地图, mx=28), 用 map_size 当矩形宽度
        会让面板右边界落到地图内部, 视觉上面板与地图重叠。
        """
        mx, my = self.map_origin
        if self.panel == "right":
            x0 = mx + self.map_size
            if x0 >= self.width - 10:
                return None
            return (x0, my, self.width, my + self.map_size)
        if self.panel == "bottom":
            # 面板起点要跳过地图下沿那条 HUD 条, 否则会与 HUD 文字重叠
            y0 = my + self.map_size + self.hud_bar
            if y0 >= self.height - 10:
                return None
            return (mx, y0, mx + self.map_size, self.height)
        return None

    @classmethod
    def from_config(cls, aspect: str | None = None) -> "Layout":
        width, height = config.aspect_size(aspect or config.DEFAULT_ASPECT)
        hud_bar = 0
        if width == height:
            panel = "none"
            map_size = min(width, height)
        elif width > height:
            # 横屏: 地图靠左, 右侧留出约 1/4 宽的信息面板
            panel = "right"
            map_size = min(height, max(int(width * 0.66), 1))
        else:
            # 竖屏: 地图在上方, 地图下沿留一条 HUD 条, 其余给信息面板。
            # hud_bar 必须参与 map_size 的计算, 否则地图 + HUD 条会挤掉面板空间。
            panel = "bottom"
            hud_bar = 78
            map_size = min(width, int((height - hud_bar) * 0.60))
        return cls(width=width, height=height, map_size=int(map_size),
                   panel=panel, hud_bar=hud_bar)


# ------------------------------------------------------------------
# 预先索引的逐 tick 数据
# ------------------------------------------------------------------
@dataclass
class TickTable:
    """按玩家分组、按 tick 排序的位置表, 支持 O(log n) 时间片查询."""

    names: list[str]                 # 玩家名 (按组顺序)
    side: list[str]                  # 每个玩家的阵营
    ticks: list[np.ndarray]          # 每个玩家的 tick 数组 (已排序, int32)
    xs: list[np.ndarray]             # 世界 X
    ys: list[np.ndarray]             # 世界 Y
    health: list[np.ndarray]         # 血量
    place: list[np.ndarray]          # 区域名 (object 数组)

    @property
    def all_ticks(self) -> np.ndarray:
        return np.concatenate(self.ticks) if self.ticks else np.array([], dtype="int32")

    def index_at(self, player_i: int, tick: int, *, pre: np.ndarray | None = None) -> int:
        """返回该玩家在 <= tick 的最后一帧下标 (找不到返回 -1).

        `pre` 是预先算好的下标数组 (见 RadarRenderer._frame_indices), 传入可
        避免同一帧内反复 searchsorted。
        """
        if pre is not None:
            return int(pre[player_i])
        arr = self.ticks[player_i]
        if arr.size == 0:
            return -1
        i = int(np.searchsorted(arr, tick, side="right")) - 1
        return i if i >= 0 else -1

    def slice_at(self, player_i: int, tick_lo: int, tick_hi: int) -> slice:
        """返回 [tick_lo, tick_hi] 区间在该玩家数组里的切片."""
        arr = self.ticks[player_i]
        if arr.size == 0:
            return slice(0, 0)
        lo = int(np.searchsorted(arr, tick_lo, side="left"))
        hi = int(np.searchsorted(arr, tick_hi, side="right"))
        return slice(lo, hi)


def build_tick_table(demo, *, min_rows: int = 200) -> TickTable:
    """把 demo.ticks 转成分玩家 numpy 数组.

    注意两点真实数据的坑:
      1. `ticks` 里有 NaN (玩家死亡瞬间/观战状态), 必须丢掉, 否则坐标变换出 NaN。
      2. 同一玩家会因换边/死亡出现多个 side 分组 (含 side=None)。这里按行数
         过滤掉稀碎的 None 组, 再按名字合并同侧分组, 避免出现重复玩家点。
    """
    t = (
        demo.ticks.select(["name", "side", "tick", "X", "Y", "health", "place"])
        .drop_nulls(["X", "Y", "tick"])
        .sort(["name", "tick"])
    )
    names: list[str] = []
    side: list[str] = []
    ticks: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    hp: list[np.ndarray] = []
    pl_: list[np.ndarray] = []

    # 先按 (name, side) 分组, 收集所有有效组
    groups: list[tuple[str, str, "object"]] = []
    for key, part in t.partition_by(
        ["name", "side"], as_dict=True, maintain_order=True
    ).items():
        nm, sd = key if isinstance(key, tuple) else (key, "")
        nm_s, sd_s = str(nm), ("" if sd is None else str(sd))
        if part.height < min_rows:
            continue
        groups.append((nm_s, sd_s, part))

    # 同一 (name, side) 可能被拆成多段 -> 合并; 不同 side 保留 (换边)
    merged: dict[tuple[str, str], list] = {}
    for nm_s, sd_s, part in groups:
        merged.setdefault((nm_s, sd_s), []).append(part)

    for (nm_s, sd_s), parts in merged.items():
        if len(parts) == 1:
            cat = parts[0]
        else:
            import polars as pl

            cat = pl.concat(parts).sort("tick")
        names.append(nm_s)
        side.append(sd_s or "ct")
        ticks.append(cat["tick"].to_numpy().astype("int32"))
        xs.append(cat["X"].to_numpy().astype("float64"))
        ys.append(cat["Y"].to_numpy().astype("float64"))
        hp.append(cat["health"].to_numpy().astype("float64"))
        pl_.append(np.asarray(cat["place"].to_list(), dtype=object))

    return TickTable(names=names, side=side, ticks=ticks, xs=xs, ys=ys, health=hp, place=pl_)


# ------------------------------------------------------------------
# 渲染参数
# ------------------------------------------------------------------
@dataclass
class RenderStyle:
    """视觉风格 —— 由 EDL 的 effects 控制, 每个 clip 一份.

    尺寸一律用**世界单位** (CS2 units) 而不是像素, 渲染时按视口跨度换算成
    像素。这样不同地图/不同取景下, 圆点相对地图的大小都一致 —— 否则在
    大范围取景时圆点会糊成一片。
    """

    trail_len: int = config.TRAIL_TICKS
    trail_alpha: int = 130
    trail_width_units: float = 11.0
    # 玩家圆点半径不再用世界单位: 逐段跟拍后各段跨度不同, 按世界单位换算会
    # 导致圆点大小飘忽。改为固定像素半径 (见 RadarRenderer._dot_radius),
    # 用 dot_scale 调节。
    dot_min_frac: float = 0.62
    dot_scale: float = 1.0                 # 全局缩放 (EDL 可调)
    kill_flash: float = 0.45        # X 标记持续时间 (秒)
    kill_ring: float = 0.75         # 圆环扩散时长 (秒)
    show_places: bool = True
    show_hud: bool = True
    vignette: bool = True
    hud_alpha: int = 235
    focus_glow: bool = True

    # 道具尺寸 (世界单位)
    smoke_world: float = 144.0
    fire_world: float = 120.0

    # --- P4 画面质感 ---
    # 段首尾淡入淡出: 只对已渲染好的帧做一次黑场混合, **零额外渲染成本**,
    # 却能显著软化硬切 (换回合时的场景跳变最刺眼)。默认开启。
    fade_in: float = 0.18           # 秒, 0 表示关闭
    fade_out: float = 0.22
    fade_color: str = "#000000"
    # 拖尾高斯模糊 (残影/速度感): **默认关闭**, 因为代价实测太大。
    # PIL 没有小范围模糊原语, 全帧模糊约占单帧预算 30~45% (1080² 下 19~28ms),
    # 会把渲染从 25fps 拖到 10fps。而且 2D 雷达上玩家是匀速平移, 模糊带来的
    # 观感提升有限。想要的话把它设成 1.5~2.5 即可, 但要接受渲染时间翻倍。
    trail_blur: float = 0.0
    # 击杀瞬间震屏: 只是把地图区整体平移几个像素, 成本可忽略。默认开启。
    shake_px: float = 9.0
    shake_sec: float = 0.22

    @classmethod
    def from_effects(cls, effects: dict | None) -> "RenderStyle":
        """从 EDL 的 effects 字段构造风格 (未知字段忽略)."""
        st = cls()
        if not effects:
            return st
        simple = {
            "trail_len": ("trail_len", int),
            "trail_alpha": ("trail_alpha", int),
            "show_places": ("show_places", bool),
            "show_hud": ("show_hud", bool),
            "vignette": ("vignette", bool),
            "dot_scale": ("dot_scale", float),
            "fade_in": ("fade_in", float),
            "fade_out": ("fade_out", float),
            "trail_blur": ("trail_blur", float),
            "shake_px": ("shake_px", float),
        }
        for key, (attr, cast) in simple.items():
            if effects.get(key) is not None:
                try:
                    setattr(st, attr, cast(effects[key]))
                except Exception:
                    pass
        return st


# ------------------------------------------------------------------
# 渲染器
# ------------------------------------------------------------------
class RadarRenderer:
    """负责把一个 HighlightCard 渲染成帧序列."""

    def __init__(
        self,
        demo,
        table: TickTable,
        viewport: mv.Viewport,
        *,
        base_image: Image.Image | None = None,
        style: RenderStyle | None = None,
        layout: Layout | None = None,
        title: str = "",
    ) -> None:
        self.demo = demo
        self.table = table
        self.vp = viewport
        self.style = style or RenderStyle()
        self.layout = layout or Layout.from_config("square")
        self.title = title
        self._base = base_image
        self._kill_events: list[KillEvent] = []
        self._name_to_i = {n: i for i, n in enumerate(table.names)}
        # 地图区在画布中的偏移, 供所有绘制坐标平移
        self._ox, self._oy = self.layout.map_origin

    # ---------------- 画布 ----------------
    def _blank_canvas(self) -> Image.Image:
        """整张画布 (含信息面板背景)."""
        L = self.layout
        canvas = Image.new("RGB", (L.width, L.height), config.BG_COLOR)
        if L.panel != "none":
            d = ImageDraw.Draw(canvas)
            rect = L.panel_rect
            if rect:
                x0, y0, x1, y1 = rect
                d.rectangle([x0 - 8, y0, x1, y1], fill="#0b1017")
                d.line([(x0 - 8, y0), (x0 - 8, y1)], fill="#2b3648", width=3)
        return canvas

    def _compose(self, map_img: Image.Image, *, offset: tuple[int, int] = (0, 0)) -> Image.Image:
        """把地图区贴到画布上 (offset 用于震屏)."""
        L = self.layout
        if L.panel == "none" and map_img.size == (L.width, L.height) and offset == (0, 0):
            return map_img
        canvas = self._blank_canvas()
        ox = self._ox + offset[0]
        oy = self._oy + offset[1]
        # 震屏可能把地图挪出画布, 裁掉越界部分再贴, 否则 PIL 会报错
        src = map_img
        if ox < 0:
            src = src.crop((-ox, 0, src.width, src.height))
            ox = 0
        if oy < 0:
            src = src.crop((0, -oy, src.width, src.height))
            oy = 0
        if ox + src.width > L.width:
            src = src.crop((0, 0, L.width - ox, src.height))
        if oy + src.height > L.height:
            src = src.crop((0, 0, src.width, L.height - oy))
        if src.width > 0 and src.height > 0:
            canvas.paste(src, (ox, oy))
        return canvas

    def clip_viewport(
        self,
        card: HighlightCard,
        *,
        zoom: float = 1.0,
        clamp_to: mv.Viewport | None = None,
    ) -> mv.Viewport:
        """按该片段算取景 (逐段跟拍).

        focus 取"主角 + 该段所有受害者 + 助攻者" —— 素材讲的是这段交火, 相机
        只需框住参与交火的人。把全场 10 人都纳入会让跨度撑到整张地图, 跟拍就
        失效了 (实测所有素材都会撞上上限)。

        另外把主角的**中位位置**作为 must_include: 保证主角始终在画面内,
        否则它跑到边缘时名字标注会被切掉。
        """
        focus = {card.player}
        for k in card.kills:
            if k.victim:
                focus.add(k.victim)
            if k.attacker and k.attacker != card.player:
                focus.add(k.attacker)

        # 主角在该段的中位位置 (用中位而非末帧, 避免镜头跟着抖动)
        main_xy: tuple[float, float] | None = None
        for i, nm in enumerate(self.table.names):
            if nm != card.player:
                continue
            sl = self.table.slice_at(i, card.start_tick, card.end_tick)
            if sl.stop <= sl.start:
                continue
            xs = self.table.xs[i][sl]
            ys = self.table.ys[i][sl]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not bool(ok.any()):
                continue
            main_xy = (float(np.median(xs[ok])), float(np.median(ys[ok])))
            break

        return mv.Viewport.for_tick_window(
            self.table, card.start_tick, card.end_tick,
            focus=focus, size=self.vp.size, zoom=zoom,
            clamp_to=clamp_to, must_include=main_xy,
        )

    # ---------------- 底图 ----------------
    def _ensure_base(self, places: dict[str, tuple[float, float]] | None, map_name: str) -> Image.Image:
        if self._base is None or self._base.mode != "RGBA":
            # 外部传入的 base_image 可能是 RGB (mv.render_base 返回 RGB),
            # 必须走 _set_base 统一转 RGBA, 否则 alpha_composite 会报
            # "image has wrong mode"。
            src = self._base if self._base is not None else mv.render_base(
                self.vp, places=places, map_name=map_name
            )
            self._set_base(src)
        return self._base

    def _set_base(self, img: Image.Image) -> None:
        """缓存底图, 并做两件一次性的事:

        1. **预转 RGBA** —— 每帧 `convert("RGBA")` 实测约 8ms/帧, 纯浪费。
        2. **把暗角烘进底图** —— 原先每帧对整幅图做 `Image.composite`, 实测占
           单帧开销约 22%。暗角是静态的, 烘焙后每帧省掉一次全图合成。
        """
        base = img.convert("RGBA")
        if self.style.vignette:
            size = self.vp.size
            vig = self._get_vignette(size)
            dark = Image.new("RGBA", base.size, (0, 0, 0, 255))
            # 注意顺序: composite(image1, image2, mask) 在 **mask 为白** 处取 image1。
            # 遮罩中心是 255 (保留画面), 边缘趋暗 -> 所以 image1 必须是亮的底图,
            # 暗色层放 image2。反过来会把中心压黑、边缘留亮 (暗角反转)。
            base = Image.composite(base, dark, vig)
        self._base = base

    def _get_vignette(self, size: int) -> Image.Image:
        """暗角遮罩 (L 模式), 按尺寸缓存."""
        if getattr(self, "_vignette_cache", None) is None or self._vig_size != size:
            yy, xx = np.mgrid[0:size, 0:size]
            cx = cy = (size - 1) / 2.0
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.72)
            mask = np.clip(1.0 - np.clip(r - 0.55, 0, 1) ** 1.7 * 0.95, 0.0, 1.0)
            self._vignette_cache = Image.fromarray((mask * 255).astype("uint8"), mode="L")
            self._vig_size = size
        return self._vignette_cache

    def rebase(self, vp: mv.Viewport, places: dict[str, tuple[float, float]] | None,
               map_name: str) -> None:
        """换一个视口并重建底图 —— 逐段跟拍时每段调用一次.

        底图重建 (含暗角烘焙) 约 0.05s, 相对每段几百帧的渲染开销可以忽略。
        """
        self.vp = vp
        self._set_base(mv.render_base(vp, places=places, map_name=map_name))

    # ---------------- 主入口 ----------------
    def _kills_for_frame(self, tick: float) -> list[tuple[KillEvent, float, float, float, float]]:
        """返回本帧要画的击杀及其屏幕坐标.

        击杀位置只取决于击杀时刻 (固定 tick), 与当前帧无关, 所以在逐帧循环前
        由 `_prepare_kills` 算好缓存; 这里只做时间窗筛选。
        """
        out = []
        for k, vx, vy, ax, ay in self._kill_cache:
            dt = (tick - k.tick) / TICKRATE
            if -0.05 <= dt <= self.style.kill_ring:
                out.append((k, vx, vy, ax, ay))
        return out

    def _prepare_kills(self, card: HighlightCard) -> None:
        """预计算每条击杀的屏幕坐标 (受害者 + 击杀者)."""
        s = max(self.vp.units_to_px(26), 7.0)
        cache: list[tuple[KillEvent, float, float, float, float]] = []
        for k in self._kill_events:
            if not k.victim_side:
                continue
            vi = self._lookup(k.victim, k.victim_side)
            if vi is None:
                continue
            jv = self.table.index_at(vi, int(k.tick))
            if jv < 0:
                continue
            vx, vy = self.vp.to_px(float(self.table.xs[vi][jv]),
                                   float(self.table.ys[vi][jv]))
            ax = ay = float("nan")
            ai = self._lookup(k.attacker, k.attacker_side)
            if ai is not None:
                ja = self.table.index_at(ai, int(k.tick))
                if ja >= 0:
                    ax, ay = self.vp.to_px(float(self.table.xs[ai][ja]),
                                           float(self.table.ys[ai][ja]))
            cache.append((k, float(vx), float(vy), float(ax), float(ay)))
        self._kill_cache = cache
        self._kill_mark = s

    def render_clip(
        self,
        card: HighlightCard,
        *,
        fps: int = config.FPS,
        speed: float = 1.0,
        places: dict[str, tuple[float, float]] | None = None,
        map_name: str = "",
        score_text: str | None = None,
        progress: tuple[int, int] | None = None,
        frame_skip: int = 1,
    ) -> Iterator[Image.Image]:
        """逐帧产出 PIL Image.

        Args:
            card: 要渲染的素材
            fps: 输出帧率
            speed: 播放倍速 (>1 快放), 决定 tick 步进
            places: 底图区域标签
            map_name: 右下角水印
            score_text: HUD 顶部右侧文字 (如 "2 - 1")
            progress: (当前序号, 总数) 显示在 HUD
            frame_skip: 跳帧渲染 (调试用, 2 = 只渲染偶数帧)
        """
        base_map = self._ensure_base(places, map_name)
        st = self.style
        size = self.vp.size

        tick_step = TICKRATE / fps * speed
        n_frames = max(int(card.duration_ticks / (TICKRATE / fps) / speed), 1)

        # kill 事件转成 (tick, ...) 便于每帧查询
        self._kill_events = sorted(card.kills, key=lambda k: k.tick)
        self._prepare_kills(card)

        for fi in range(n_frames):
            if frame_skip > 1 and fi % frame_skip:
                continue
            tick = card.start_tick + fi * tick_step

            # 动态层画在地图区大小的透明层上, 之后整体贴到画布
            overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay, "RGBA")

            # 一次性算好本帧所有下标, 三层复用 (原先每层各自 searchsorted)
            lo_tick = tick - max(st.trail_len, 1)
            idx = self._frame_indices(tick, lo_tick)
            pre = np.fromiter((j for _, j, _ in idx), dtype="int64",
                              count=len(idx))

            self._draw_utility(d, card, tick)

            # 拖尾单独一层: 先模糊再合成 -> 残影/速度感, 且不会糊掉圆点和击杀标记
            if st.trail_blur > 0:
                trail_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                self._draw_trails(ImageDraw.Draw(trail_layer, "RGBA"), tick,
                                  layer=trail_layer, idx=idx)
                trail_layer = self._fast_blur(trail_layer, st.trail_blur)
                overlay = Image.alpha_composite(overlay, trail_layer)
                d = ImageDraw.Draw(overlay, "RGBA")
            else:
                self._draw_trails(d, tick, idx=idx)

            self._draw_kills(d, tick, card, pre=pre)
            self._draw_players(d, tick, card, pre=pre)
            map_img = Image.alpha_composite(base_map, overlay).convert("RGB")
            # 暗角已烘焙进底图 (见 _set_base), 这里不再逐帧合成

            # 击杀震屏: 把地图区整体平移一小段距离 (分 3 段衰减, 比连续抖动更"脆")
            offset = self._shake_offset(tick)
            img = self._compose(map_img, offset=offset)

            # 段首尾淡入淡出 (在 HUD 之前: 黑场时 HUD 仍可见, 便于观众跟随)
            # 只在中段之外才做混合 —— Image.blend 是全帧运算, 1080² 实测 18ms,
            # 每帧都调会让渲染慢 30%。判定为 1.0 时直接跳过。
            fade = self._fade_factor(fi, n_frames, fps, st)
            if fade < 0.999:
                img = Image.blend(
                    Image.new("RGB", img.size, st.fade_color), img, fade
                )

            if st.show_hud:
                d2 = ImageDraw.Draw(img, "RGBA")
                self._draw_hud(d2, card, tick, fi, n_frames, score_text, progress)

            yield img

    # ---------------- P4 效果 ----------------
    @staticmethod
    def _fast_blur(layer: Image.Image, radius: float, factor: int = 4) -> Image.Image:
        """降采样近似的模糊 —— 比全分辨率高斯快约一个数量级.

        实测: 1080x1080 RGBA 上做 `GaussianBlur`, 半径 0.8 与 2.5 的耗时**相同**
        (都约 0.1s/帧), 直接把渲染从 15fps 拖到 6fps —— 慢 2.6 倍, 完全不可接受。
        把图层缩到 1/4 线性尺寸 (1/16 像素量) 再放大回来, 得到的等效模糊半径
        约等于 `radius * factor`, 视觉上对"残影"足够, 代价降到可以忽略。
        """
        size = layer.width
        small = max(int(size / max(factor, 1)), 8)
        tiny = layer.resize((small, small), Image.BILINEAR)
        # 先在低分辨率上补一次小半径高斯, 让边缘更柔和
        r_small = max(radius / max(factor, 1), 0.4)
        tiny = tiny.filter(ImageFilter.GaussianBlur(radius=r_small))
        return tiny.resize((size, size), Image.BILINEAR)

    def _fade_factor(self, fi: int, n_frames: int, fps: int, st: RenderStyle) -> float:
        """段首尾的黑场混合系数: 0=全黑, 1=原图."""
        t = fi / max(fps, 1)
        total = n_frames / max(fps, 1)
        f = 1.0
        if st.fade_in > 0 and t < st.fade_in:
            f = min(f, t / st.fade_in)
        if st.fade_out > 0:
            tail = total - t
            if tail < st.fade_out:
                f = min(f, max(tail, 0.0) / st.fade_out)
        return max(0.0, min(1.0, f))

    def _shake_offset(self, tick: float) -> tuple[int, int]:
        """击杀瞬间的屏幕位移; 用分段衰减制造"撞击感"."""
        st = self.style
        if st.shake_px <= 0 or not self._kill_events:
            return (0, 0)
        best = 0.0
        for k in self._kill_events:
            dt = (tick - k.tick) / TICKRATE
            if 0.0 <= dt <= st.shake_sec:
                best = max(best, 1.0 - dt / st.shake_sec)
        if best <= 0:
            return (0, 0)
        # 3 段交替方向, 幅度递减
        seg = int((1.0 - best) * 3.0)
        signs = ((1, -1), (-1, 1), (1, 1))
        sx, sy = signs[min(seg, 2)]
        amp = st.shake_px * best
        return (int(sx * amp), int(sy * amp))

    # ---------------- 各图层 ----------------
    def _visible(self, tick: float, pre: np.ndarray | None = None) -> list[tuple[int, int]]:
        """返回 [(玩家下标, 该玩家在 tick 处的帧下标)], 只含活着/刚死的."""
        out: list[tuple[int, int]] = []
        for i in range(len(self.table.names)):
            j = int(pre[i]) if pre is not None else self.table.index_at(i, int(tick))
            if j < 0:
                continue
            # 只画未来 8 秒内还有数据的玩家 (避免死亡后一直画)
            arr = self.table.ticks[i]
            if arr[j] > tick + 1:
                continue
            out.append((i, j))
        return out

    def _frame_indices(self, tick: float, lo: float | None = None) -> list[tuple[int, int, slice]]:
        """一次性算出该帧需要的所有下标, 供各图层复用.

        原先是 trails / players / kills 三层各自对每个玩家调用 `searchsorted`
        (实测每帧 57 次, 占渲染时间约 17%)。这里统一算一遍传下去。

        Returns: [(玩家下标, 当前帧下标(-1 表示无), 时间片 slice)]
        """
        ti = int(tick)
        out: list[tuple[int, int, slice]] = []
        for i in range(len(self.table.names)):
            arr = self.table.ticks[i]
            if arr.size == 0:
                out.append((i, -1, slice(0, 0)))
                continue
            j = int(np.searchsorted(arr, ti, side="right")) - 1
            if lo is None:
                sl = slice(0, 0)
            else:
                a = int(np.searchsorted(arr, int(lo), side="left"))
                b = int(np.searchsorted(arr, ti, side="right"))
                sl = slice(a, b)
            out.append((i, j, sl))
        return out

    @property
    def _dot_radius(self) -> float:
        """玩家圆点半径 (像素).

        刻意**不随视口跨度变化**: 逐段跟拍后每段跨度不同, 若按世界单位换算,
        拉近的段落圆点会变得很大、拉远的又很小, 大小飘忽反而更难读。
        这里以"满屏取景"为参照算出固定像素半径, 再由 dot_scale 全局调节。
        """
        return max(config.MAP_SIZE * 0.0130 * self.style.dot_scale, 4.0)

    def _draw_trails(self, d: ImageDraw.ImageDraw, tick: float,
                     layer: Image.Image | None = None,
                     idx: list | None = None) -> None:
        """画移动拖尾.

        `layer` 非空时 (拖尾模糊模式) 走"干净折线"路径: 在独立图层上画一条
        均匀亮度的轨迹, 之后整层做模糊 —— 这样得到的是连续残影。
        非模糊模式用**降采样折线**: 把轨迹点抽稀到约 1/3, 再单次 polyline
        绘制, 比逐段画线快得多, 视觉上几乎无差别 (原实现每帧每玩家画 ~96 段)。
        """
        st = self.style
        if st.trail_len <= 0:
            return
        lo = tick - st.trail_len
        clean = layer is not None
        wmax = max(int(self.vp.units_to_px(st.trail_width_units)), 2)
        n_players = len(self.table.names)

        for i in range(n_players):
            sl = idx[i][2] if idx is not None else self.table.slice_at(i, int(lo), int(tick))
            if sl.stop - sl.start < 2:
                continue
            xs = self.table.xs[i][sl]
            ys = self.table.ys[i][sl]
            hps = self.table.health[i][sl]
            if np.all(hps <= 0):
                continue
            px, py = self.vp.to_px_array(xs, ys)
            side = self.table.side[i]
            col = config.SIDE_STYLE.get(side, config.SIDE_STYLE["ct"])
            n = len(px)

            if clean:
                pts = [(float(px[k]), float(py[k])) for k in range(n)]
                d.line(pts, fill=self._rgba(col["face"], 210), width=wmax, joint="curve")
                continue

            # 抽稀: 每 ~3 个点取 1 个 (端点必取), 再用单条 polyline 画。
            # 原实现是每帧每玩家逐段 `d.line()` (~96 次), 占单帧开销约 33%。
            step = max(n // 24, 1)
            keep = list(range(0, n, step))
            if keep[-1] != n - 1:
                keep.append(n - 1)
            if len(keep) < 2:
                continue
            pts = [(float(px[k]), float(py[k])) for k in keep]
            # 尾段单独加粗提亮, 保留"越新越亮"的观感
            d.line(pts, fill=self._rgba(col["face"], st.trail_alpha), width=wmax,
                   joint="curve")
            tail = pts[-min(len(pts), 6):]
            if len(tail) >= 2:
                d.line(tail, fill=self._rgba(col["face"], min(st.trail_alpha + 90, 255)),
                       width=wmax + 1, joint="curve")

    def _draw_players(self, d: ImageDraw.ImageDraw, tick: float, card: HighlightCard,
                      pre: np.ndarray | None = None) -> None:
        st = self.style
        r_full = self._dot_radius
        r_min = r_full * st.dot_min_frac
        for i, j in self._visible(tick):
            if pre is not None and j >= 0:
                j = int(pre[i])
            if j < 0:
                continue
            hp = float(self.table.health[i][j])
            if hp <= 0:
                continue
            x = float(self.table.xs[i][j])
            y = float(self.table.ys[i][j])
            px, py = self.vp.to_px(x, y)
            side = self.table.side[i]
            col = config.SIDE_STYLE.get(side, config.SIDE_STYLE["ct"])

            frac = max(min(hp / 100.0, 1.0), 0.0)
            r = r_min + (r_full - r_min) * frac
            is_focus = self.table.names[i] == card.player

            # 主打玩家加外发光 + 白环
            if is_focus and st.focus_glow:
                d.ellipse(
                    [px - r * 2.0, py - r * 2.0, px + r * 2.0, py + r * 2.0],
                    fill=self._rgba(col["face"], 44),
                )
                d.ellipse(
                    [px - r * 1.35, py - r * 1.35, px + r * 1.35, py + r * 1.35],
                    outline=self._rgba("#ffffff", 210),
                    width=max(int(r * 0.22), 2),
                )

            d.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=col["face"],
                outline=col["edge"],
                width=max(int(r * 0.2), 2),
            )

            # 名字: 只标主打玩家, 避免糊成一片
            if is_focus:
                font = mv.load_font(max(int(r * 1.5), 14), bold=True)
                d.text(
                    (px, py - r - r * 1.3),
                    self.table.names[i],
                    fill=config.TEXT_COLOR,
                    font=font,
                    anchor="mm",
                    stroke_width=3,
                    stroke_fill="#000000",
                )

    def _draw_kills(self, d: ImageDraw.ImageDraw, tick: float, card: HighlightCard,
                    pre: np.ndarray | None = None) -> None:
        st = self.style
        s = getattr(self, "_kill_mark", 12.0)
        for k, px, py, apx, apy in self._kills_for_frame(tick):
            dt = (tick - k.tick) / TICKRATE

            # 击杀者 -> 受害者连线 (先画, 压在标记下面)
            if apx == apx and apy == apy:      # 非 NaN
                a = max(int(170 * (1 - dt / max(st.kill_flash, 1e-6))), 0)
                if a > 0:
                    d.line([(apx, apy), (px, py)], fill=self._rgba("#ffffff", a), width=3)

            # X 标记 (白描边 + 红芯, 保证在深色底图上都醒目)
            if dt <= st.kill_flash:
                for w, c in ((max(int(s * 0.45), 3), config.ACCENT_COLOR),
                             (max(int(s * 0.18), 2), "#ffffff")):
                    d.line([(px - s, py - s), (px + s, py + s)], fill=c, width=w)
                    d.line([(px - s, py + s), (px + s, py - s)], fill=c, width=w)
            # 扩散圆环
            if dt >= 0:
                rr = s * 1.4 + s * 5.0 * (dt / st.kill_ring)
                a = int(210 * (1 - dt / st.kill_ring))
                d.ellipse(
                    [px - rr, py - rr, px + rr, py + rr],
                    outline=self._rgba(config.ACCENT_COLOR, a),
                    width=max(int(s * 0.2), 3),
                )

    def _draw_utility(self, d: ImageDraw.ImageDraw, card: HighlightCard, tick: float) -> None:
        for u in card.utility:
            if not (u["start_tick"] <= tick <= u["end_tick"]):
                continue
            px, py = self.vp.to_px(float(u["x"]), float(u["y"]))
            t = u["type"]
            if t == "smoke":
                # 烟雾: 半径随时间膨胀 (世界单位换算, 与取景联动)
                age = (tick - u["start_tick"]) / TICKRATE
                r = self.vp.units_to_px(self.style.smoke_world + min(age * 22, 34))
                d.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=self._rgba("#b9c2cf", 88),
                    outline=self._rgba("#e6ecf3", 150),
                    width=max(int(r * 0.05), 2),
                )
            elif t == "fire":
                age = (tick - u["start_tick"]) / TICKRATE
                r = self.vp.units_to_px(self.style.fire_world + min(age * 10, 30))
                d.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=self._rgba("#ff6d00", 105),
                    outline=self._rgba("#ffca28", 190),
                    width=max(int(r * 0.06), 3),
                )
            elif t == "flash":
                age = (tick - u["start_tick"]) / TICKRATE
                r = self.vp.units_to_px(60 + age * 900)
                a = int(230 * max(1 - age / 0.6, 0))
                if a > 0:
                    d.ellipse(
                        [px - r, py - r, px + r, py + r],
                        outline=self._rgba("#ffffff", a),
                        width=max(int(r * 0.06), 3),
                    )

    def _draw_hud(
        self,
        d: ImageDraw.ImageDraw,
        card: HighlightCard,
        tick: float,
        fi: int,
        n_frames: int,
        score_text: str | None,
        progress: tuple[int, int] | None,
    ) -> None:
        st = self.style
        L = self.layout
        size = self.vp.size
        W, H = L.width, L.height
        mx, my = self._ox, self._oy
        a = st.hud_alpha
        f_big = mv.load_font(38, bold=True)
        f_mid = mv.load_font(26)
        f_sm = mv.load_font(20)
        col = config.SIDE_STYLE.get(card.player_side, config.SIDE_STYLE["ct"])
        tags = " · ".join(card.tags[:5]) if card.tags else ""
        done = sum(1 for k in card.kills if k.tick <= tick)

        def pips(x0: int, y0: int) -> None:
            for n in range(len(card.kills)):
                xx = x0 + n * 26
                if n < done:
                    d.rectangle([xx, y0, xx + 18, y0 + 18], fill=config.ACCENT_COLOR)
                else:
                    d.rectangle([xx, y0, xx + 18, y0 + 18],
                                outline=self._rgba("#3d4a5c", 255), width=2)

        if L.panel == "right" and L.panel_rect:
            # 右侧面板: 信息全部搬到面板里, 地图区保持干净
            x0, y0, x1, y1 = L.panel_rect
            px = x0 + 18
            yy = y0 + 40
            if self.title:
                d.text((px, yy), self.title, fill="#8b98a9", font=f_sm, anchor="lm")
                yy += 44
            d.text((px, yy), "ROUND", fill="#8b98a9", font=f_sm, anchor="lm")
            yy += 42
            d.text((px, yy), str(card.round_num), fill=config.TEXT_COLOR, font=f_big, anchor="lm")
            yy += 70
            d.text((px, yy), col["label"], fill=col["face"], font=f_mid, anchor="lm")
            yy += 40
            for line in _wrap(card.player, 11):
                d.text((px, yy), line, fill=config.TEXT_COLOR, font=f_mid, anchor="lm")
                yy += 34
            yy += 24
            if score_text:
                d.text((px, yy), "SCORE", fill="#8b98a9", font=f_sm, anchor="lm")
                yy += 40
                d.text((px, yy), score_text, fill=config.TEXT_COLOR, font=f_big, anchor="lm")
                yy += 60
            if card.kills:
                d.text((px, yy), "KILLS", fill="#8b98a9", font=f_sm, anchor="lm")
                yy += 34
                pips(px, yy)
                yy += 40
            if tags:
                d.text((px, yy), "TAGS", fill="#8b98a9", font=f_sm, anchor="lm")
                yy += 32
                for line in _wrap(tags.upper(), 13):
                    d.text((px, yy), line, fill="#8b98a9", font=f_sm, anchor="lm")
                    yy += 26
            if progress:
                d.text((px, y1 - 30), f"CLIP {progress[0]}/{progress[1]}",
                       fill="#8b98a9", font=f_sm, anchor="lm")

        elif L.panel == "bottom" and L.panel_rect:
            # 竖屏: 地图在上, 地图下沿一条 HUD 条 + 下方信息面板
            x0, y0, x1, y1 = L.panel_rect
            d.rectangle([mx, my + size, mx + size, my + size + 74],
                        fill=self._rgba("#0b1017", a))
            d.line([(mx, my + size + 74), (mx + size, my + size + 74)],
                   fill=self._rgba("#2b3648", 255), width=2)
            cy = my + size + 37
            d.text((mx + 22, cy), f"ROUND {card.round_num}", fill=config.TEXT_COLOR,
                   font=f_big, anchor="lm")
            if score_text:
                d.text((mx + size - 22, cy), score_text, fill=config.TEXT_COLOR,
                       font=f_big, anchor="rm")
            px = x0 + 14
            yy = y0 + 46 + 74
            d.text((px, yy), col["label"], fill=col["face"], font=f_mid, anchor="lm")
            d.text((px + 70, yy), card.player, fill=config.TEXT_COLOR, font=f_mid, anchor="lm")
            yy += 46
            pips(px, yy)
            yy += 46
            if tags:
                for line in _wrap(tags.upper(), 22):
                    d.text((px, yy), line, fill="#8b98a9", font=f_sm, anchor="lm")
                    yy += 26
            if progress:
                d.text((x1 - 14, y1 - 26), f"CLIP {progress[0]}/{progress[1]}",
                       fill="#8b98a9", font=f_sm, anchor="rm")

        else:
            # 方形: 沿用上下压条
            d.rectangle([0, 0, W, 74], fill=self._rgba("#0b1017", a))
            d.line([(0, 74), (W, 74)], fill=self._rgba("#2b3648", 255), width=2)
            d.text((22, 37), f"ROUND {card.round_num}", fill=config.TEXT_COLOR,
                   font=f_big, anchor="lm")
            d.text((190, 37), col["label"], fill=col["face"], font=f_mid, anchor="lm")
            d.text((250, 37), card.player, fill=config.TEXT_COLOR, font=f_mid, anchor="lm")
            if score_text:
                d.text((W - 24, 37), score_text, fill=config.TEXT_COLOR, font=f_big, anchor="rm")
            d.rectangle([0, H - 62, W, H], fill=self._rgba("#0b1017", a))
            d.line([(0, H - 62), (W, H - 62)], fill=self._rgba("#2b3648", 255), width=2)
            if tags:
                d.text((22, H - 31), tags.upper(), fill="#8b98a9", font=f_sm, anchor="lm")
            if progress:
                d.text((W - 24, H - 31), f"CLIP {progress[0]}/{progress[1]}",
                       fill="#8b98a9", font=f_sm, anchor="rm")
            pips(470, 50)

    # ---------------- 工具 ----------------
    def _lookup(self, name: str, side: str) -> int | None:
        """按 (名字, 阵营) 找玩家下标."""
        if not name:
            return None
        # 先试 (name, side) 组合, 再退回纯名字
        for i, n in enumerate(self.table.names):
            if n == name and (not side or self.table.side[i] == side):
                return i
        for i, n in enumerate(self.table.names):
            if n == name:
                return i
        return None

    @staticmethod
    def _rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (r, g, b, max(0, min(255, int(alpha))))


def _wrap(text: str, width: int) -> list[str]:
    """按字符数折行 (中文没有空格, 不能用 textwrap 的按词折行)."""
    if not text:
        return []
    return [text[i : i + width] for i in range(0, len(text), width)]
