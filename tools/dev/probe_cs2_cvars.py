"""在 CS2 自己的二进制里核实 cvar / 命令是否存在.

背景: 慢放片段有观感问题。demo 片段 5.45s 要撑成 9.00s (0.606x), 而录像时
demo 是按 1x 播放、60fps 抓帧 -> 只有 ~327 帧, 靠 fit_clip 拉长到 9s 时
**每帧要停留 1.65 帧的时间**, 也就是靠复制帧凑时长, 画面会一顿一顿。

正确做法是录的时候就放慢 demo: `host_timescale <speed>` -> D 秒 demo 花
D/s 秒播完, 60fps 抓到 60*T 帧, 正好等于成片需要的帧数, **一帧不重不漏**
(smooth slow-mo)。

但"往脚本里写没核实过的命令"是这个项目明确拒绝过的做法, 所以先确认这些名字
在 CS2 的二进制里真的存在。字符串在 ConVar/ConCommand 注册时是 ASCII 字面量,
可以直接在文件里搜。

用法: .venv\\Scripts\\python.exe tools\\dev\\probe_cs2_cvars.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cs2clipper import hlaerec as H                            # noqa: E402

#: 想核实的名字。第二个元素只是说明用途, 方便读输出。
WANTED: list[tuple[str, str]] = [
    ("host_timescale", "录慢放用: 让 demo 按 s 倍速播, 抓到的帧数正好等于成片需要"),
    ("demo_pauseatservertick", "自动停录: 跑到该 tick 自动暂停, 省掉一次按键"),
    ("demo_timescale", "CS2 里可能改名成了这个 (Source1 的 host_timescale)"),
    ("demo_gototick", "定位备选: 按 tick 而不是秒"),
    ("demo_pause", "已在用"),
    ("demo_resume", "已在用"),
    ("spec_player", "已在用"),
    ("spec_show_xray", "引导脚本里的可选项"),
    ("demo_ui", "引导脚本里报过 Unknown command"),
    ("cl_draw_only_deathnotices", "引导脚本里的可选项"),
    ("sv_cheats", "控制台变量 (必然存在, 当搜索的对照组)"),
]


def candidate_files() -> list[Path]:
    exe = H.find_cs2_exe()
    if exe is None:
        return []
    game = exe.parents[2]                       # ...\game
    out: list[Path] = [exe]
    for sub in ("bin/win64", "csgo/bin/win64"):
        d = game / sub
        if d.is_dir():
            out += sorted(p for p in d.iterdir()
                          if p.suffix.lower() in (".dll", ".exe"))
    return [p for p in out if p.is_file()]


def scan(path: Path, needles: dict[str, bytes]) -> set[str]:
    """一次读完整个文件, 找出命中的名字 (文件不大, 不必分块)."""
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    return {name for name, b in needles.items() if b in data}


def main() -> int:
    files = candidate_files()
    if not files:
        print("找不到 CS2 安装, 无法核实")
        return 1

    needles = {n: n.encode("ascii") for n, _ in WANTED}
    found: dict[str, list[str]] = {n: [] for n in needles}
    print(f"扫描 {len(files)} 个文件 ...")
    total_mb = 0.0
    for f in files:
        total_mb += f.stat().st_size / 2**20
        for name in scan(f, needles):
            found[name].append(f.name)
    print(f"共 {total_mb:.0f} MB\n")

    print(f"{'名字':26} {'结果':6} 出现位置 / 用途")
    print("-" * 96)
    for name, why in WANTED:
        where = found[name]
        mark = "存在" if where else "**没找到**"
        loc = ", ".join(sorted(set(where))[:4]) if where else "-"
        print(f"{name:26} {mark:6} {loc}")
        print(f"{'':26} {'':6} -> {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
