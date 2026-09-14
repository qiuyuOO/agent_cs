"""端到端验证 HLAE 画面源: 真 demo + 真歌, 只把"录制"换成合成帧序列.

这是在不启动游戏的前提下, 能对整条 HLAE 链路做的**最强验证**:
  分析音乐 → 抽亮点 → 编排 → 生成录制计划/CS2 脚本
  → 按计划造出"看起来像 HLAE 产物"的帧序列目录
  → 走 build_from_recordings 规整 → concat_clips 铺音乐 → final.mp4

要验证的是: 成片能不能出来、时长对不对、画面尺寸对不对、片段是不是真的
来自"录制素材"(而不是退化成别的路径)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"E:\agent_cs")

from PIL import Image, ImageDraw

from cs2clipper import config, compose, demo as D, hlaerec as H, music as M, planner as P

DEMO = (r"D:\5E_cs2_demo\g161-n-20260912132632481838433_de_dust2"
        r"\g161-n-20260912132632481838433_de_dust2.dem")
SONG = r"D:\CloudMusic\在虚无中永存 - 英雄主义.mp3"
OUT = config.ROOT / "out" / "_hlae_e2e"
CLIPS = 3                     # 只跑 3 段, 够证明链路即可

for d in (OUT, H.record_output_dir()):
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)

print("=== 1) 音乐 + demo + 编排 ===")
a = M.analyze_music(SONG)
res = D.analyze_demo(DEMO, max_cards=30, with_utility=False)
cards = D.cards_from_dicts(res["highlights"])
edl = P.plan_edit(a, cards, use_llm=False, max_clips=CLIPS, verbose=False)
print(f"  歌 {a.duration:.1f}s | {len(cards)} 张卡 | EDL {len(edl.clips)} 段, "
      f"覆盖 {edl.total_duration:.1f}s")

print("\n=== 2) 生成录制计划与 CS2 脚本 ===")
plan = H.plan_from_edl(edl, cards)
scripts = H.write_cs2_scripts(res["demo_path"], plan,
                             output_dir=H.record_output_dir(),
                             fps=H.hlae_capture_fps(),
                             fallback_dir=OUT / "cs2_cfg")
print(f"  {len(plan)} 段; 脚本目录 {scripts['cfg_dir']} "
      f"(在 CS2 cfg 目录内={scripts['in_cs2_cfg_dir']})")
H.write_plan(plan, OUT / "hlae_plan.json")

print("\n=== 3) 造出模拟的 HLAE 录制产物 ===")
rec = H.record_output_dir()
rec.mkdir(parents=True, exist_ok=True)
prefix = rec.name
for seg in plan:
    # 名字形态与 cfg 里生成的完全一致: <输出目录名>_<片段名>
    d = rec / f"{prefix}_{seg.name}"
    d.mkdir(parents=True, exist_ok=True)
    # 帧数按真实情况给: 录制帧率 × demo 时长
    n = max(int(seg.demo_span * H.hlae_capture_fps()), 10)
    for i in range(n):
        img = Image.new("RGB", (1920, 1080), (18, 22, 30))
        dr = ImageDraw.Draw(img)
        # 画点东西, 顺便让每帧不同 (可用来判断不是静止/重复帧)
        x = int(200 + (1720 * i / max(n - 1, 1)))
        dr.rectangle([x - 40, 500, x + 40, 580], fill=(240, 160, 32))
        dr.text((60, 60), f"{seg.name}  R{seg.round_num}  {seg.player}",
                fill=(230, 237, 243))
        dr.text((60, 110), f"frame {i + 1}/{n}  demo {seg.demo_start_sec:.2f}s",
                fill=(154, 167, 184))
        img.save(d / f"frame_{i:08d}.png")
    print(f"  {d.name}: {n} 帧 (demo 跨 {seg.demo_span:.2f}s → 成片 "
          f"{seg.out_duration:.2f}s, realtime={seg.realtime_factor:.2f})")

found = H.discover_recordings(rec)
print(f"  探测到 {len(found['frame_dirs'])} 个帧序列目录, 共 {found['total_frames']} 帧")

print("\n=== 4) 规整成精确时长片段 ===")
items, notes = H.build_from_recordings(plan, rec, OUT / "clips",
                                      fps=H.hlae_capture_fps(),
                                      size=config.aspect_size("square"))
for n in notes:
    print("  [note]", n)
for it in items:
    d = compose.probe_duration(it.path)
    print(f"  {it.path.name}: 声明 {it.duration:.2f}s, 实际 {d:.2f}s, "
          f"转场={it.transition}")
assert len(items) == len(plan), f"应 {len(plan)} 段, 实际 {len(items)}"

print("\n=== 5) 拼接 + 铺音乐 ===")
final = OUT / "final.mp4"
compose.concat_clips(items, final, music_path=SONG,
                     total_duration=edl.total_duration)
dur = compose.probe_duration(final)
size_mb = final.stat().st_size / 1024 / 1024
print(f"  {final}")
print(f"  时长 {dur:.2f}s (EDL 声明 {edl.total_duration:.2f}s), {size_mb:.2f} MB")

print("\n=== 6) 抽帧确认画面确实来自录制素材 ===")
ff = str(config.require_ffmpeg())
shot = OUT / "verify_frame.png"
import subprocess
subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "1.0", "-i", str(final), "-frames:v", "1", str(shot)],
               check=True)
img = Image.open(shot)
print(f"  抽帧 {img.size[0]}x{img.size[1]}, {shot.stat().st_size} bytes")
# 画面里应当有我们画的那个橙色方块 -> 说明走的确实是"录制素材"这条路
colors = img.convert("RGB").getcolors(maxcolors=1 << 20) or []
orange = sum(c for c, px in colors if px[0] > 200 and 120 < px[1] < 200 and px[2] < 90)
print(f"  橙色像素 {orange} 个 (>0 说明成片用的是我们造的录制素材)")

print("\n=== 7) 结论 ===")
ok_dur = abs(dur - edl.total_duration) < 0.6
ok_frames = orange > 0
print(f"  时长对齐: {'通过' if ok_dur else '不通过'}")
print(f"  画面来自录制素材: {'通过' if ok_frames else '不通过'}")
sys.exit(0 if (ok_dur and ok_frames) else 1)
