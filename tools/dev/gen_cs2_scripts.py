"""生成整套 CS2 录制脚本 (真 demo + 真歌), 并直接写进 CS2 的 cfg 目录.

用途: 用户已经在游戏里等着 exec, 我们要立刻产出一套可用的脚本。
对齐口径与 CLI 的 --hlae-gen-cfg 一致 (确定性编排, 不调 LLM)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"E:\agent_cs")

from cs2clipper import config, demo as D, hlaerec as H, music as M, planner as P

DEMO = (r"D:\5E_cs2_demo\g161-n-20260912132632481838433_de_dust2"
        r"\g161-n-20260912132632481838433_de_dust2.dem")
SONG = r"D:\CloudMusic\在虚无中永存 - 英雄主义.mp3"
OUT = config.ROOT / "out" / "hlae_run"

print("=== 1) 体检 ===")
pf = H.preflight(check_running=False)
print(f"  就绪={pf.ok}  HLAE={pf.info['hlae_version']}  "
      f"cs2={pf.info['cs2_size_gb']} GB")
for p in pf.problems:
    print("  [问题]", p)

print("\n=== 2) 音乐 + demo ===")
a = M.analyze_music(SONG)
res = D.analyze_demo(DEMO, max_cards=30, with_utility=False)
cards = D.cards_from_dicts(res["highlights"])
print(f"  歌 {a.duration:.1f}s / {len(a.segments)} 段 | demo {res['round_count']} 回合 "
      f"| {len(cards)} 张卡")

print("\n=== 3) 编排 ===")
edl = P.plan_edit(a, cards, use_llm=False, max_clips=22, verbose=False)
print(f"  {len(edl.clips)} 段, 覆盖 {edl.total_duration:.1f}s / {edl.music_duration:.1f}s")

print("\n=== 4) 生成 CS2 脚本并写入 cfg 目录 ===")
OUT.mkdir(parents=True, exist_ok=True)
plan = H.plan_from_edl(edl, cards)
H.write_plan(plan, OUT / "hlae_plan.json")
scripts = H.write_cs2_scripts(res["demo_path"], plan,
                             output_dir=H.record_output_dir(),
                             fps=H.hlae_capture_fps(),
                             fallback_dir=OUT / "cs2_cfg")
print(f"  脚本目录: {scripts['cfg_dir']}")
print(f"  在 CS2 的 cfg 目录内: {scripts['in_cs2_cfg_dir']}")
for n in scripts["notes"]:
    print("  [提示]", n)
print(f"  引导脚本: {Path(scripts['bootstrap']).name}")
print(f"  停录脚本: {Path(scripts['stop']).name}")
print(f"  片段脚本: {len(scripts['clips'])} 个 "
      f"({Path(scripts['clips'][0]).name} .. {Path(scripts['clips'][-1]).name})")
print(f"  录制输出: {scripts['record_dir']}")

print("\n=== 5) 确认文件真的在 CS2 能找到的位置 ===")
cfg_dir = Path(scripts["cfg_dir"])
for f in [scripts["bootstrap"], scripts["stop"], *scripts["clips"]]:
    p = Path(f)
    ok = p.is_file()
    if not ok:
        print(f"  [缺失] {p}")
print(f"  {len(list(cfg_dir.glob('cs2clipper_*.cfg')))} 个 cs2clipper_*.cfg 已在 "
      f"{cfg_dir}")

print("\n=== 6) 前 3 段的镜头信息 (供你核对) ===")
for s in plan[:3]:
    print(f"  {s.name}: R{s.round_num} {s.player}  "
          f"demo {s.demo_start_sec:.1f}~{s.demo_end_sec:.1f}s → "
          f"成片 {s.out_duration:.1f}s")
