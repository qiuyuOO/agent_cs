"""从 AfxHookSource2.dll 里提取它真正实现的 mirv 命令与帮助文本.

比翻 wiki 更可靠: wiki 可能滞后, 而这里读的是本机已校验过的那份二进制。
结果写进 work/_hlae_cmds.txt (UTF-8), 避免控制台 GBK 打印乱码。
"""
from __future__ import annotations

import re
from pathlib import Path

DLL = Path(r"E:\agent_cs\tools\hlae\x64\AfxHookSource2.dll")
OUT = Path(r"E:\agent_cs\work\_hlae_cmds.txt")

data = DLL.read_bytes()
print(f"DLL: {DLL}  {len(data)} bytes")

# ASCII 字符串
strings = re.findall(rb"[ -~]{4,}", data)
text = [s.decode("ascii", "replace") for s in strings]
print(f"ASCII 字符串: {len(text)} 条")

# UTF-16LE 字符串 (C++ 里常用宽字符)
wide = re.findall(rb"(?:[ -~]\x00){4,}", data)
text += [w.decode("utf-16-le", "replace") for w in wide]
print(f"含 UTF-16 后总计: {len(text)} 条")

cmds = sorted({s for s in text if re.match(r"^mirv_[a-zA-Z0-9_]+$", s)})
tools = sorted({s for s in text if re.match(r"^mirv_[a-zA-Z0-9_]+ \w", s)})

lines: list[str] = []
lines.append(f"# AfxHookSource2.dll 命令清单  ({DLL.name})")
lines.append("")
lines.append(f"## 1. 顶层 mirv 命令 ({len(cmds)} 个)")
for c in cmds:
    lines.append(f"  {c}")

lines.append("")
lines.append("## 2. 带子命令的帮助串 (前 200 条)")
seen = set()
n = 0
for s in sorted(set(tools)):
    if s in seen:
        continue
    seen.add(s)
    lines.append(f"  {s}")
    n += 1
    if n >= 200:
        break

# 找与录制/相机/cfg 相关的帮助块
lines.append("")
lines.append("## 3. 与 streams / record / cam / cfg 相关的长帮助文本")
pat = re.compile(r"(streams|record|camimport|cam_|mirv_cmd|mat_queue|host_framerate|"
                 r"demo_(goto|togglepause|resume)|spec_|loadcfg|exec\b)", re.I)
hits = [s for s in text if len(s) > 25 and pat.search(s)]
for s in sorted(set(hits))[:260]:
    lines.append(f"  {s}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n已写入 {OUT}  ({OUT.stat().st_size} bytes, {len(lines)} 行)")
print(f"顶层 mirv 命令数量: {len(cmds)}")
