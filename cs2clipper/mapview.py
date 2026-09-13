"""坐标变换与雷达底图.

两条路线:
    1. 能拿到 awpy 的 MAP_DATA (需要 ~/.awpy/maps/map-data.json) 时, 用官方
       pos_x / pos_y / scale 变换, 与游戏内雷达像素对齐。
    2. 拿不到时 (本项目当前情况: awpycs.com 的 17595823 资源已 404),
       退化为"按 demo 实际活动范围自适应取景" —— 不需要任何外部素材,
       对任意地图都能工作, 且取景更紧凑、可读性更好。

底图用 PIL 直接绘制 (网格 + 区域标签), 不引入 matplotlib:
    * 纯像素操作, 渲染 1000+ 帧时快得多
    * 不受 matplotlib 字体缓存写盘限制影响
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

# 常见 Windows 中文字体, 按优先级尝试
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\arial.ttf",
)


@lru_cache(maxsize=8)
def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """加载一个支持中文的字体, 失败则退回 PIL 默认位图字体."""
    order = _FONT_CANDIDATES[1:] + _FONT_CANDIDATES[:1] if bold else _FONT_CANDIDATES
    for cand in order:
        p = Path(cand)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


# ------------------------------------------------------------------
# 视口 (世界坐标 -> 像素)
# ------------------------------------------------------------------
@dataclass
class Viewport:
    """把 CS2 世界坐标线性映射到 0..size 的方形画面.

    映射 (与 Valve 雷达一致, Y 轴翻转):
        px = (X - x_min) / (x_max - x_min) * size
        py = (y_max - Y) / (y_max - y_min) * size

    注意: 这里只描述**地图区** (正方形)。宽屏/竖屏成片中地图区是画布内的
    一块方形区域, 由 radar.Layout 负责摆放, 视口本身不关心画布尺寸。
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    size: int = config.MAP_SIZE
    margin: float = 0.06        # 半径方向留白比例

    @property
    def span_x(self) -> float:
        return max(self.x_max - self.x_min, 1e-6)

    @property
    def span_y(self) -> float:
        return max(self.y_max - self.y_min, 1e-6)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        # 保持等比缩放: 用较大的跨度, 避免地图被拉伸
        span = max(self.span_x, self.span_y)
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        hx = cx - span / 2.0
        hy = cy - span / 2.0
        px = (x - hx) / span * self.size
        py = (self.size - 1) - (y - hy) / span * self.size
        return px, py

    def to_px_array(self, xs, ys):
        """向量化版本, 渲染时批量转换 (避免逐点 python 循环)."""
        import numpy as np

        span = max(self.span_x, self.span_y)
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        hx = cx - span / 2.0
        hy = cy - span / 2.0
        px = (np.asarray(xs, dtype="float64") - hx) / span * self.size
        py = (self.size - 1) - (np.asarray(ys, dtype="float64") - hy) / span * self.size
        return px, py

    def units_to_px(self, units: float) -> float:
        """世界单位 -> 像素 (用于把圆点/道具尺寸与取景联动)."""
        span = max(self.span_x, self.span_y)
        return abs(units) / span * self.size

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "size": self.size,
        }

    @classmethod
    def for_tick_window(
        cls,
        table,
        tick_lo: int,
        tick_hi: int,
        *,
        focus: set[str] | None = None,
        size: int = config.MAP_SIZE,
        pad: int = 64,
        margin: float = 0.10,
        zoom: float = 1.0,
        min_span: float = 1150.0,
        max_span: float = 2600.0,
        radius_percentile: float = 80.0,
        fallback_players: int = 4,
        clamp_to: "Viewport | None" = None,
        must_include: tuple[float, float] | None = None,
    ) -> "Viewport":
        """按某个时间窗内的玩家位置取景 —— 逐段跟拍的核心.

        关键设计: **只框住这一段的参与者, 不是全场 10 人**。

        实测过: 用全场玩家算取景时, 到中心的 95 分位距离在 1995~6484 units,
        需要的跨度 4000~13000 —— 全部素材都撞上上限, 逐段跟拍退化成"永远最远",
        等于没做。原因很直接: 素材讲的是**一个人的高光**, 而地图另一头的队友
        会把包围圈撑到整张地图。

        所以优先用 `focus` (主角 + 受害者) 取景, 样本不足时才逐步纳入附近的
        其他玩家 (最多 `fallback_players` 个, 按到主角的距离排序)。

        Args:
            table: radar.TickTable
            focus: 优先纳入取景的玩家名集合 (通常 {主角} ∪ {受害者})
            zoom: >1 表示拉近 (跨度变小), 用于按音乐激烈度调整镜头远近
            radius_percentile: 用"到中心的距离"的哪个分位作为半径
            clamp_to: 把取景中心限制在这个范围内 —— 防止镜头漂到地图外的空白
            must_include: 必须完整出现在画面里的坐标 (通常是主角),
                若不在画面内则平移取景把它拉回来。否则主角跑到边缘时
                名字标注会被画面边界切掉。
        """
        lo, hi = int(tick_lo) - pad, int(tick_hi) + pad
        import numpy as np

        def collect(idx: int) -> tuple[np.ndarray, np.ndarray]:
            sl = table.slice_at(idx, lo, hi)
            if sl.stop <= sl.start:
                return np.empty(0), np.empty(0)
            hps = table.health[idx][sl]
            alive = hps > 0                      # 尸体停在原地会拽偏中心
            if not bool(alive.any()):
                return np.empty(0), np.empty(0)
            x = table.xs[idx][sl][alive]
            y = table.ys[idx][sl][alive]
            ok = np.isfinite(x) & np.isfinite(y)
            return x[ok], y[ok]

        # 1. focus 里的玩家 (同名可能因换边出现多组, 全部纳入)
        idx_focus = [i for i, n in enumerate(table.names) if focus and n in focus]
        # 2. 其余玩家, 按到主角的接近程度排序备用
        idx_rest = [i for i in range(len(table.names)) if i not in set(idx_focus)]

        xs_parts, ys_parts = [], []
        for i in idx_focus:
            x, y = collect(i)
            if x.size:
                xs_parts.append(x)
                ys_parts.append(y)

        # focus 样本太少时, 逐步纳入其他玩家 (保证画面里有参照, 不至于只剩一个点)
        if sum(p.size for p in xs_parts) < 30:
            for i in idx_rest[:fallback_players]:
                x, y = collect(i)
                if x.size:
                    xs_parts.append(x)
                    ys_parts.append(y)

        if not xs_parts:
            # 完全没有 focus 可用 -> 退回全场
            for i in range(len(table.names)):
                x, y = collect(i)
                if x.size:
                    xs_parts.append(x)
                    ys_parts.append(y)
            if not xs_parts:
                return cls.from_positions([], [], size=size, margin=margin,
                                          min_span=min_span, max_span=max_span)

        xs = np.concatenate(xs_parts)
        ys = np.concatenate(ys_parts)
        if xs.size == 0:
            return cls.from_positions([], [], size=size, margin=margin,
                                      min_span=min_span, max_span=max_span)

        # 镜头中心: 位置中位数 (对单个绕后的玩家稳健)
        fx, fy = float(np.median(xs)), float(np.median(ys))
        d = np.sqrt((xs - fx) ** 2 + (ys - fy) ** 2)
        radius = float(np.percentile(d, radius_percentile))
        span = max(radius * 2.0, 1.0)

        span = span / max(zoom, 0.2)
        span = min(max(span, min_span), max_span)
        half = span / 2.0

        # 把中心夹在允许范围内, 避免镜头漂到地图外的空白区
        def clamp_axis(v: float, lo_b: float, hi_b: float) -> float:
            if hi_b <= lo_b:
                return (lo_b + hi_b) / 2.0
            return min(max(v, lo_b), hi_b)

        if clamp_to is not None:
            fx = clamp_axis(fx, clamp_to.x_min, clamp_to.x_max)
            fy = clamp_axis(fy, clamp_to.y_min, clamp_to.y_max)

        # 主角必须完整在画面内, 否则名字标注会被边界切掉。
        # 留出 20% 边距: 名字标注画在圆点上方约 1.3 倍半径处, 边距太小时
        # 圆点还在画面内、文字已经被切掉 (实测 12% 仍会切)。
        if must_include is not None:
            mx, my = float(must_include[0]), float(must_include[1])
            pad_world = span * 0.20
            if mx < fx - half + pad_world:
                fx = mx + half - pad_world
            elif mx > fx + half - pad_world:
                fx = mx - half + pad_world
            if my < fy - half + pad_world:
                fy = my + half - pad_world
            elif my > fy + half - pad_world:
                fy = my - half + pad_world

        return cls(fx - half, fx + half, fy - half, fy + half, size=size, margin=margin)

    @classmethod
    def from_positions(
        cls,
        xs,
        ys,
        *,
        size: int = config.MAP_SIZE,
        margin: float = 0.06,
        min_span: float = 1200.0,
        max_span: float = 3400.0,
        pad_percentile: float = 3.0,
    ) -> "Viewport":
        """按活动范围自适应取景.

        用 3%/97% 分位而不是极值: 单个离群点 (出生点、穿图) 会把取景
        拉得很远, 画面变得又空又小。再用 min/max_span 夹住取景范围,
        保证玩家圆点在画面上的相对大小稳定。
        """
        import numpy as np

        xs = np.asarray(xs, dtype="float64").ravel()
        ys = np.asarray(ys, dtype="float64").ravel()
        # 真实 ticks 里有 NaN (死亡瞬间), 必须过滤掉
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size == 0:
            # 兜底: 用 dust2 大致范围
            return cls(-2400.0, 1800.0, -1200.0, 3000.0, size=size, margin=margin)
        lo_p, hi_p = pad_percentile, 100.0 - pad_percentile
        x_lo, x_hi = np.percentile(xs, [lo_p, hi_p])
        y_lo, y_hi = np.percentile(ys, [lo_p, hi_p])
        cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        # 保证中心点在整体活动范围内 (避免只有少数点时取景跑偏)
        cx = float(np.clip(cx, xs.min(), xs.max()))
        cy = float(np.clip(cy, ys.min(), ys.max()))
        span = max(x_hi - x_lo, y_hi - y_lo)
        span = min(max(span, min_span), max_span)
        half = span / 2.0
        # 关键: 夹取范围必须在**加留白之前**对"含留白的最终跨度"生效。
        # 否则 span 先乘 (1+2*margin) 就可能超出 max_span —— 实测 3400 * 1.12
        # = 3808, 取景比预期宽 12%。
        half = min(half * (1.0 + margin * 2), max_span / 2.0)
        return cls(cx - half, cx + half, cy - half, cy + half, size=size, margin=margin)


def try_map_data_viewport(map_name: str, size: int = config.MAP_SIZE) -> Viewport | None:
    """尝试用 awpy 的官方地图数据构造视口; 不可用则返回 None.

    官方数据给的是"雷达图左上角世界坐标 + 每像素单位数", 需要知道雷达图
    分辨率才能算右下角。这里按 Valve 标准雷达 1024px 估算, 仅作为可选路径;
    主线仍走 from_positions 自适应。
    """
    try:
        from awpy.data.map_data import MAP_DATA

        md = MAP_DATA.get(map_name)
        if not md:
            return None
        pos_x = float(md["pos_x"])
        pos_y = float(md["pos_y"])
        scale = float(md["scale"])
        radar_px = 1024.0
        return Viewport(
            x_min=pos_x,
            x_max=pos_x + radar_px * scale,
            y_min=pos_y - radar_px * scale,
            y_max=pos_y,
            size=size,
        )
    except Exception:
        return None


# ------------------------------------------------------------------
# 底图渲染
# ------------------------------------------------------------------
def render_base(
    vp: Viewport,
    *,
    places: dict[str, tuple[float, float]] | None = None,
    map_name: str = "",
    grid_step: int = 512,
) -> Image.Image:
    """画静态底图: 深色背景 + 网格 + 区域标签 + 地图名.

    Args:
        places: {区域名: (世界X, 世界Y)} —— 用 demo 里出现过的点位名标注
    """
    size = vp.size
    img = Image.new("RGB", (size, size), config.BG_COLOR)
    d = ImageDraw.Draw(img)

    import numpy as np

    # 视口无效时直接返回纯背景, 不让 np.arange 抛错
    if not all(np.isfinite([vp.x_min, vp.x_max, vp.y_min, vp.y_max])):
        return img

    # --- 网格 (按世界坐标步进, 保证与地图尺度一致) ---
    span = max(vp.span_x, vp.span_y)
    # 找一个视觉密度合理的步长
    step = float(grid_step)
    while span / step > 14:
        step *= 2
    while span / step < 5:
        step /= 2

    gx = np.arange(np.floor(vp.x_min / step) * step, vp.x_max + step, step)
    gy = np.arange(np.floor(vp.y_min / step) * step, vp.y_max + step, step)
    for x in gx:
        px, _ = vp.to_px(x, vp.y_min)
        if 0 <= px < size:
            d.line([(px, 0), (px, size)], fill=config.GRID_COLOR, width=1)
    for y in gy:
        _, py = vp.to_px(vp.x_min, y)
        if 0 <= py < size:
            d.line([(0, py), (size, py)], fill=config.GRID_COLOR, width=1)

    # --- 外边框 ---
    d.rectangle([0, 0, size - 1, size - 1], outline="#2b3648", width=3)

    # --- 区域标签 ---
    if places:
        font = load_font(20)
        for name, (wx, wy) in places.items():
            px, py = vp.to_px(wx, wy)
            if not (0 <= px < size and 0 <= py < size):
                continue
            # 点位用一个小方块 + 名称, 类似游戏内雷达标注
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=config.PLACE_COLOR)
            d.text((px + 9, py - 11), name, fill=config.PLACE_COLOR, font=font)

    # --- 地图名 (右下角水印) ---
    if map_name:
        font = load_font(26)
        d.text((size - 24, size - 46), map_name, fill="#243044", font=font, anchor="ra")

    return img


def collect_places(demo, kills=None, max_places: int = 14) -> dict[str, tuple[float, float]]:
    """统计区域名 → 平均坐标, 用于在雷达底图上标注点位.

    坐标取自逐 tick 表的 `place` 列 —— 每个区域在整局里的平均位置, 比用击杀
    位置更可靠 (后者只覆盖发生过击杀的地方)。

    失败时返回空字典: 点位标注是可选的装饰, 没有它成片依然能出。但失败原因会
    记录到 `collect_places.last_error`, 便于排查"为什么图上没有点位名"。
    """
    try:
        import polars as pl

        t = demo.ticks.select(["place", "X", "Y"]).drop_nulls()
        agg = (
            t.group_by("place")
            .agg([pl.col("X").mean().alias("mx"), pl.col("Y").mean().alias("my"), pl.len().alias("n")])
            .sort("n", descending=True)
            .head(max_places)
        )
        collect_places.last_error = None        # type: ignore[attr-defined]
        return {r["place"]: (float(r["mx"]), float(r["my"]))
                for r in agg.to_dicts() if r["place"]}
    except Exception as e:
        collect_places.last_error = f"{type(e).__name__}: {e}"   # type: ignore[attr-defined]
        return {}
