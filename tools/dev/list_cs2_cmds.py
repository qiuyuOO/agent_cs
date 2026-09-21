"""枚举 / 核对 CS2 二进制里真实存在的命令名与 cvar 名.

为什么要枚举而不是猜 —— 已经吃过两次亏:
  * `demo_ui`: 子串探测命中了 `demo_ui_mode` 里的 `demo_ui`, 于是给出一个
    **根本不存在的命令名**, 控制台只会回 Unknown command;
  * 覆盖层 (控制台 / 回放控制条) 和镜头没锁到玩家, 直接决定成片能不能用,
    靠猜会一直试错。
所以这里把二进制里的 ASCII 字符串抽出来, **用整串精确匹配**核对名字。

用法:
  # 按前缀列出名单
  python tools/dev/list_cs2_cmds.py --prefix demo_ spec_ con_
  # 精确核对若干名字 (推荐在把命令写进 cfg 之前跑)
  python tools/dev/list_cs2_cmds.py --exact demo_ui_mode spec_lock_to_accountid
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cs2clipper import hlaerec as H                            # noqa: E402

CHUNK = 1 << 22                      # 4 MB, 流式读, client.dll 有 200MB+
STR_RE = re.compile(rb"[ -~]{3,64}")
#: 命令/cvar 名的形态。**必须整串匹配** —— 子串匹配会把 `demo_ui_mode` 里的
#: `demo_ui` 也算命中, 这正是之前给出错名字的原因。
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


def strings_in(path: Path):
    """流式抽 ASCII 字符串, 不把整个 dll 读进内存."""
    tail = b""
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            data = tail + buf
            for m in STR_RE.finditer(data):
                yield m.group().decode("ascii", "replace")
            tail = data[-64:]
    if tail:
        for m in STR_RE.finditer(tail):
            yield m.group().decode("ascii", "replace")


def scan_all() -> dict[str, set[str]]:
    """-> {名字: {出现在哪些文件}}"""
    exe = H.find_cs2_exe()
    if exe is None:
        return {}
    game = exe.parents[2]
    files = [exe] + sorted(
        p for sub in ("bin/win64", "csgo/bin/win64")
        for p in (game / sub).glob("*.dll")
    )
    hits: dict[str, set[str]] = {}
    for f in files:
        if not f.is_file():
            continue
        for s in strings_in(f):
            if NAME_RE.match(s):
                hits.setdefault(s, set()).add(f.name)
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="枚举 / 核对 CS2 命令名")
    ap.add_argument("--prefix", nargs="*", default=None,
                    help="按前缀列出完整名单")
    ap.add_argument("--exact", nargs="*", default=None,
                    help="精确核对这些名字是否存在")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.prefix and not args.exact:
        args.prefix = ["demo_", "spec_", "mirv_", "con_"]

    hits = scan_all()
    print(f"共抽出 {len(hits)} 个形如命令名的字符串\n")

    if args.prefix:
        for prefix in args.prefix:
            group = sorted(n for n in hits if n.startswith(prefix))
            print(f"==== {prefix}*  ({len(group)}) ====")
            for n in group:
                print(f"  {n:38} {sorted(hits[n])[:2]}")
            print()

    if args.exact:
        print("==== 精确核对 ====")
        bad = 0
        for n in args.exact:
            if n in hits:
                print(f"  {n:38} FOUND    {sorted(hits[n])[:2]}")
            else:
                print(f"  {n:38} **不存在**")
                bad += 1
        # 顺带提示"是不是某个存在的名字的子串" —— 那正是 demo_ui 的坑
        for n in args.exact:
            if n in hits:
                continue
            near = sorted(x for x in hits if n in x)[:5]
            if near:
                print(f"  提示: {n!r} 不是命令名, 但它出现在 {near} 里面 "
                      f"(子串探测会在这里产生假阳性)")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
