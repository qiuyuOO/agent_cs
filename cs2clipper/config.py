"""路径与环境引导.

**重要**: 这个模块必须在 import matplotlib / awpy / librosa 之前导入,
因为它负责把 HOME 和 MPLCONFIGDIR 重定向到工作区内, 否则这些库会尝试写
C:\\Users\\HP\\.matplotlib\\ 之类的工作区外路径 (在受限沙箱下会被拒绝).

用法:
    from cs2clipper import config  # noqa: F401  (side-effect import)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# ------------------------------------------------------------------
# 1. 工作区根目录与子目录
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = ROOT / "out"          # 成片输出
WORK_DIR = ROOT / "work"        # 中间产物 (帧序列、音频、EDL)
ASSET_DIR = ROOT / "assets"     # 静态素材 (雷达底图、字体)
TOOLS_DIR = ROOT / "tools"      # 本地二进制 (ffmpeg)
CACHE_DIR = ROOT / ".cache"     # 第三方库缓存 (mpl/pip/tmp)
HOME_DIR = ROOT / ".home"       # 伪造的 HOME, 供 awpy 等库写数据
DATA_DIR = ROOT / "data"        # 持久化数据 (用户偏好、运行记录的 sqlite)

for _d in (OUT_DIR, WORK_DIR, ASSET_DIR, TOOLS_DIR, CACHE_DIR, HOME_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 2. 环境重定向 (必须在重库 import 前生效)
# ------------------------------------------------------------------
os.environ.setdefault("HOME", str(HOME_DIR))
os.environ.setdefault("USERPROFILE", str(HOME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mpl"))
(CACHE_DIR / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
# 保证无界面渲染
os.environ.setdefault("MPLBACKEND", "Agg")


# ------------------------------------------------------------------
# 3. 外部工具定位
# ------------------------------------------------------------------
def find_ffmpeg() -> Path | None:
    """优先用工作区内的 ffmpeg, 其次系统 PATH, 最后常见安装位置."""
    local = TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.is_file():
        return local
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)
    for cand in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"D:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"E:\ffmpeg\bin\ffmpeg.exe"),
    ):
        if cand.is_file():
            return cand
    return None


def find_ffprobe() -> Path | None:
    ff = find_ffmpeg()
    if ff is None:
        return None
    probe = ff.with_name("ffprobe.exe")
    if probe.is_file():
        return probe
    which = shutil.which("ffprobe")
    return Path(which) if which else None


FFMPEG = find_ffmpeg()
FFPROBE = find_ffprobe()


def require_ffmpeg() -> Path:
    if FFMPEG is None:
        raise RuntimeError(
            "找不到 ffmpeg。请把 ffmpeg 放到 tools/ffmpeg/bin/, 或加入系统 PATH。"
        )
    return FFMPEG


# ------------------------------------------------------------------
# 4. 项目默认值
# ------------------------------------------------------------------
DEFAULT_DEMO = Path(
    r"D:\5E_cs2_demo\g161-20260827151717641063023_de_dust2"
    r"\g161-20260827151717641063023_de_dust2.dem"
)

# CS2 demo 固定 64 tick; awpy Demo.tickrate 一般也是 64
DEMO_TICKRATE = 64

# 渲染参数
FPS = 30                 # 输出帧率
FRAME_DPI = 100
TRAIL_TICKS = 96         # 玩家轨迹回看窗口 (tick 数), 96 tick ≈ 1.5s

# 画幅预设 —— 决定成片投到哪个平台。
#   square   1024x1024  通用/调试
#   wide     1920x1080  横屏 (B站/YouTube)
#   tall     1080x1920  竖屏 (抖音/Shorts)
ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "wide": (1920, 1080),
    "tall": (1080, 1920),
}
DEFAULT_ASPECT = "wide"

# 雷达地图区的边长 (像素, 正方形)。宽/高画面里地图按这个尺寸居中,
# 剩余空间用作信息面板。
MAP_SIZE = 1024

# 阵营配色 (CT 蓝 / T 黄, 与游戏内雷达一致) 与受击/死亡色
SIDE_STYLE: dict[str, dict[str, str]] = {
    "ct": {"face": "#4FC3F7", "edge": "#01579B", "label": "CT"},
    "t": {"face": "#FFD54F", "edge": "#E65100", "label": "T"},
}


def aspect_size(aspect: str) -> tuple[int, int]:
    """把画幅名解析成 (宽, 高). 未知名回落到默认预设."""
    if aspect in ASPECT_PRESETS:
        return ASPECT_PRESETS[aspect]
    return ASPECT_PRESETS[DEFAULT_ASPECT]

# 主题色 (深色底, 适合视频)
BG_COLOR = "#0d1117"
GRID_COLOR = "#1f2733"
PLACE_COLOR = "#3d4a5c"
TEXT_COLOR = "#e6edf3"
ACCENT_COLOR = "#ff5252"


def describe() -> str:
    """给日志/调试用的一行环境摘要."""
    return (
        f"ROOT={ROOT} | ffmpeg={FFMPEG} | out={OUT_DIR}"
    )


if __name__ == "__main__":
    print(describe())
