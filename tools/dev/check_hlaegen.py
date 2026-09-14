"""核验 HLAE 录制计划与 cfg 生成 (不启动游戏、不写 CS2 目录)。

用真 demo + 真歌走一遍: analyze_music → analyze_demo → plan_edit → 生成 cfg,
然后把生成的 cfg 打到控制台看内容是否合理。
"""
import sys
from pathlib import Path

sys.path.insert(0, r"E:\agent_cs")

from cs2clipper import config, demo as D, hlaerec as H, music as M, planner as P

DEMO = r"D:\5E_cs2_demo\g161-n-20260912132632481838433_de_dust2\g161-n-20260912132632481838433_de_dust2.dem"
SONG = r"D:\CloudMusic\在虚无中永存 - 英雄主义.mp3"

print("=== 1) 分析音乐与 demo ===")
a = M.analyze_music(SONG)
res = D.analyze_demo(DEMO, max_cards=40, with_utility=False)
cards = D.cards_from_dicts(res["highlights"])
print(f"  音乐 {a.duration:.1f}s / {len(a.segments)} 段; demo {res['round_count']} 回合, "
      f"{len(cards)} 张卡")

print("\n=== 2) 确定性编排 ===")
edl = P.plan_edit(a, cards, use_llm=False, max_clips=22, verbose=False)
print(f"  {len(edl.clips)} 段, 覆盖 {edl.total_duration:.1f}s / {edl.music_duration:.1f}s")

print("\n=== 3) EDL → 录制计划 ===")
plan = H.plan_from_edl(edl, cards)
print(f"  {len(plan)} 个片段")
for s in plan[:6]:
    print(f"    {s.name}  demo {s.demo_start_sec:8.2f}~{s.demo_end_sec:8.2f}s "
          f"(跨 {s.demo_span:5.2f}s)  成片 {s.out_duration:5.2f}s  "
          f"realtime={s.realtime_factor:5.2f}  speed={s.speed}  "
          f"R{s.round_num} {s.player[:10]}")
if len(plan) > 6:
    print(f"    ... 其余 {len(plan) - 6} 段")

rt = [s.realtime_factor for s in plan]
print(f"\n  realtime_factor 范围 {min(rt):.2f}~{max(rt):.2f}  "
      f"(<1 表示录下来的素材比成片短, 需要慢放)")

print("\n=== 4) 生成整套 CS2 录制脚本 ===")
scripts = H.write_cs2_scripts(res["demo_path"], plan,
                              output_dir=H.record_output_dir(),
                              fps=H.hlae_capture_fps(),
                              target_dir=Path(r"E:\agent_cs\work\_hlae_cfg_preview"))
print(f"  脚本目录: {scripts['cfg_dir']}")
print(f"  引导脚本: {Path(scripts['bootstrap']).name}")
print(f"  停录脚本: {Path(scripts['stop']).name}")
print(f"  片段脚本: {len(scripts['clips'])} 个 (前 3: "
      f"{[Path(c).name for c in scripts['clips'][:3]]})")

print("\n---- 引导脚本 ----")
for l in Path(scripts["bootstrap"]).read_text(encoding="utf-8").splitlines():
    print("  |", l)
print("\n---- 片段 1 脚本 ----")
for l in Path(scripts["clips"][0]).read_text(encoding="utf-8").splitlines():
    print("  |", l)
print("\n---- 片段 6 脚本 (speed=1.25 那段) ----")
for l in Path(scripts["clips"][5]).read_text(encoding="utf-8").splitlines():
    print("  |", l)
print("\n---- 停录脚本 ----")
for l in Path(scripts["stop"]).read_text(encoding="utf-8").splitlines():
    print("  |", l)

# 关键不变量: 每段的 record name 必须互不相同, 否则输出会互相覆盖
names = []
for c in scripts["clips"]:
    for l in Path(c).read_text(encoding="utf-8").splitlines():
        if l.startswith("mirv_streams record name"):
            names.append(l)
print(f"\n每段 record name 数量={len(names)} 唯一={len(set(names))}")
assert len(names) == len(set(names)), "各段的 record name 有重复 -> 输出会互相覆盖!"

print("\n=== 5) 素材探测 (录制前, 应为空) ===")
print("  ", H.discover_recordings(H.record_output_dir()))