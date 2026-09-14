"""从 AfxHookSource2.dll 里精确提取 mirv_cmd / mirv_skip / mirv_campath 的帮助文本。

用于确认"按 demo 时间自动执行命令"的确切语法, 好让生成的 cfg 能自动起停录制,
而不是让人手点 22 次。
"""
from __future__ import annotations

import re
from pathlib import Path

DLL = Path(r"E:\agent_cs\tools\hlae\x64\AfxHookSource2.dll")
OUT = Path(r"E:\agent_cs\work\_hlae_cmds2.txt")

data = DLL.read_bytes()
strings = [s.decode("ascii", "replace") for s in re.findall(rb"[ -~]{4,}", data)]
strings += [w.decode("utf-16-le", "replace")
            for w in re.findall(rb"(?:[ -~]\x00){4,}", data)]

KEYS = ("mirv_cmd", "mirv_skip", "mirv_campath", "mirv_camio", "demo_goto",
        "demo_pause", "demo_resume", "demo_timescale", "spec_player",
        "host_framerate", "playdemo")

lines: list[str] = []
for key in KEYS:
    hits = sorted({s for s in strings if key in s and len(s) > len(key)})
    lines.append(f"\n===== {key}  ({len(hits)} 条) =====")
    for s in hits[:60]:
        lines.append(f"  {s}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入 {OUT}")
print(OUT.read_text(encoding="utf-8")[:200])
