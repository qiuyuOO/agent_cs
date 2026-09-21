"""核对"计划里的玩家名"和"demo 里真实的玩家名"是否**逐字节**相同.

为什么必须查: 如果名字对不上, `spec_player "<名字>"` 永远不可能生效 ——
镜头就会停在自由视角 (实测录出来的画面正是贴地的旁观视角, 没有第一人称武器),
而成片看着像"录到了", 其实根本不是要拍的那个人。

控制台对中文是乱码的, 所以这里一律输出 ASCII 转义 (repr / utf-8 hex),
不靠眼睛看。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = ROOT / "work" / "_hlae_route" / "hlae_plan.json"
ANALYSIS = ROOT / "work" / "_hlae_route" / "demo_analysis.json"
PLAYERS = ROOT / "work" / "demo_players.json"


def a(s) -> str:
    """ASCII 安全显示: 非 ASCII 一律转义."""
    return str(s).encode("unicode_escape").decode("ascii")


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    players = json.loads(PLAYERS.read_text(encoding="utf-8"))

    real = {p["name"]: p for p in players}
    print("demo 里的玩家名 (转义显示):")
    for n, p in real.items():
        print(f"  {a(n):32} len={len(n)} codepoints  account_id={p['account_id']}")

    print(f"\n计划里的片段 (共 {len(plan)}):")
    hl_by_id = {h["id"]: h for h in analysis.get("highlights") or []}
    print(f"分析里的亮点 id: {[a(i) for i in hl_by_id]}")

    rc = 0
    for seg in plan:
        name = seg.get("player") or ""
        hid = seg.get("highlight_id") or ""
        print(f"\n  片段 {seg['name']}:")
        print(f"    player        = {a(name)}  (len={len(name)})")
        print(f"    highlight_id  = {a(hid)}")
        exact = name in real
        print(f"    ** 与 demo 玩家名逐字节相同? {exact} **")
        if not exact:
            rc = 1
            near = [n for n in real if name in n or n in name]
            print(f"    相近的名字: {[a(n) for n in near]}")
        if hid in hl_by_id:
            hp = hl_by_id[hid].get("player")
            print(f"    亮点里的 player = {a(hp)}  相同? {hp == name}")
        else:
            print("    !! 计划里的 highlight_id 在分析结果里找不到")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
