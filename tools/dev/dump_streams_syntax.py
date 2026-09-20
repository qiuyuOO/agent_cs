"""从 AfxHookSource2.dll 里挖 mirv_streams add / settings 的确切语法.

之前只用了 record name/fps/start/end, 漏掉了"必须先 add 一个流"这一步 ——
结果是只录到 audio.wav, 一帧画面都没有。这次要把 add 的完整参数组挖出来。
"""
from __future__ import annotations

import re
from pathlib import Path

DLL = Path(r"E:\agent_cs\tools\hlae\x64\AfxHookSource2.dll")
OUT = Path(r"E:\agent_cs\work\_streams_syntax.txt")

data = DLL.read_bytes()
strings = [s.decode("ascii", "replace") for s in re.findall(rb"[ -~]{4,}", data)]
strings += [w.decode("utf-16-le", "replace")
            for w in re.findall(rb"(?:[ -~]\x00){4,}", data)]
uniq = sorted(set(strings))

lines: list[str] = []
lines.append(f"# AfxHookSource2.dll 中与录制流相关的字符串 (共 {len(uniq)} 条里筛)")
lines.append("")

pats = [
    r"mirv_streams",
    r"stream type",
    r"type <",
    r"name <",
    r"settings <",
    r"record",
    r"screen",
    r"world",
    r"matte",
    r"ffmpeg",
    r"tga|bmp|png|jpe?g",
    r"startMovie|endMovie|host_framerate",
]
for pat in pats:
    hits = [s for s in uniq if re.search(pat, s, re.I) and 6 < len(s) < 300]
    lines.append(f"\n===== /{pat}/  ({len(hits)} 条) =====")
    for s in hits[:120]:
        lines.append(f"  {s}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入 {OUT} ({OUT.stat().st_size} 字节)")
