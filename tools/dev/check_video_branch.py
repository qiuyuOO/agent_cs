"""证明 ffmpeg 预设产物的匹配缺陷确实存在, 且已修好.

背景: 引导脚本用 `mirv_streams edit cs2clipper settings afxFfmpegYuv420p`,
该预设在 take 目录写出的是**已编码的 mp4**, 不是帧序列:
    <record name>/take0000/take0000.mp4

旧代码把视频按**文件名**归类后拿去匹配片段名, 永远匹配不上 ->
"真录成功也报没有素材"。本脚本同时跑旧逻辑与新逻辑, 两边对照。

用法: .venv\\Scripts\\python.exe tools\\dev\\check_video_branch.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image                                        # noqa: E402

from cs2clipper import compose, config, hlaerec as H          # noqa: E402


def main() -> int:
    tmp = config.WORK_DIR / "_vbranch_probe"
    seed = config.WORK_DIR / "_vbranch_seed"
    for d in (tmp, seed):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    stream = tmp / "_hlae_route_rec_clip_001"
    take = stream / "take0000"
    take.mkdir(parents=True, exist_ok=True)
    src = take / "take0000.mp4"
    seg = H.RecSegment(index=0, highlight_id="a", player="P", round_num=1,
                       demo_start_sec=0.0, demo_end_sec=1.0,
                       out_duration=0.8, speed=1.0, name="clip_001")
    rc = 0
    try:
        for i in range(60):
            Image.new("RGB", (160, 90), (i * 4 % 256, 30, 60)).save(
                seed / f"frame_{i:08d}.png")
        H.frames_to_clip(seed, src, fps=60, target_duration=1.0, size=(160, 90))
        print(f"[素材] {src}  ({src.stat().st_size / 1024:.1f} KiB, "
              f"{compose.probe_duration(src):.2f}s)")

        found = H.discover_recordings(tmp)
        print(f"[探测] videos={found['videos']}")

        # ---- 旧逻辑: 按文件名当键 ----
        old_keys = [v["name"] for v in found["videos"]]
        old_hit = H._match_stream(seg, old_keys)
        print(f"[旧逻辑] 键={old_keys} -> match={old_hit!r}")
        if old_hit is None:
            print("         复现缺陷: 文件名键永远匹配不上片段名 -> 判成'没有素材'")
        else:
            print("         !! 未能复现缺陷, 与本脚本的前提不符")
            rc = 1

        # ---- 旧逻辑的路径拼接 ----
        wrong = Path(tmp) / "clip_001"
        print(f"[旧逻辑] src=record_dir/命中名 -> {wrong} (存在={wrong.exists()})")

        # ---- 新逻辑: 走真实入口 ----
        items, notes = H.build_from_recordings([seg], tmp, tmp / "clips",
                                               fps=60, size=(160, 90))
        for n in notes:
            print(f"[新逻辑] note: {n}")
        if len(items) != 1:
            print(f"         !! 修好后仍拿不到片段, items={len(items)}")
            rc = 1
        else:
            got = compose.probe_duration(items[0].path)
            ok = abs(got - 0.8) < 0.15
            print(f"[新逻辑] 产出 {items[0].path.name} = {got:.2f}s "
                  f"(目标 0.80s)  {'OK' if ok else '!! 时长不对'}")
            if not ok:
                rc = 1

        print("结论:", "缺陷已复现且修复有效" if rc == 0 else "**检查未通过**")
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(seed, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
