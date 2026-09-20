"""把 mirv_streams add / edit / settings 的完整语法挖出来.

上一轮只找到
    "%s normal|depth|hudBlack|... <sUniqueStreamName> - Adds a stream of given type."
这一条, 没找到"add 之后怎么让这个流真正录画面"。现在用户实测确认
`record screen enabled 1` **不会创建流** (print 显示 Total streams: 0),
所以必须走 add 这条路。把相关字符串全部捞出来看清。
"""
from __future__ import annotations

import re
from pathlib import Path

DLL = Path(r"E:\agent_cs\tools\hlae\x64\AfxHookSource2.dll")
OUT = Path(r"E:\agent_cs\work\_add_syntax.txt")

data = DLL.read_bytes()
strings = [s.decode("ascii", "replace") for s in re.findall(rb"[ -~]{4,}", data)]
strings += [w.decode("utf-16-le", "replace")
            for w in re.findall(rb"(?:[ -~]\x00){4,}", data)]
uniq = sorted(set(strings))

pats = [
    r"adds a stream",
    r"Adds a stream",
    r"afxFfmpeg",
    r"recording preset",
    r"recording setting",
    r"settingsName",
    r"sUniqueStreamName",
    r"preset",
    r"mirv_streams (add|edit|remove|settings)",
    r"worldAction",
    r"beforePresent",
    r"take\d",
    r"take folder",
]
lines: list[str] = [f"# AfxHookSource2.dll 里与 stream add/settings 有关的字符串"]
for pat in pats:
    hits = sorted({s for s in uniq if re.search(pat, s, re.I)})
    lines.append(f"\n===== /{pat}/  ({len(hits)} 条) =====")
    for s in hits[:80]:
        lines.append(f"  {s}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入 {OUT}")
# 顺手在 stdout 上把最关键的几条打出来
for pat in (r"adds a stream", r"afxFfmpeg", r"recording preset", r"take folder"):
    hits = sorted({s for s in uniq if re.search(pat, s, re.I)})
    print(f"\n/{pat}/:")
    for s in hits[:12]:
        print("   ", s)
