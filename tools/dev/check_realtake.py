"""用**真机录下来的 take** 跑一遍素材规整, 验证能进成片.

这是"实机画面"这条链路上最后一个没被证明的环节: 前面已经确认 HLAE 真的写出
了 1080p60 的 mp4 (`<record name>/take0000/<stream name>/video.mp4`), 这里验证
`build_from_recordings` 能认出它、并把它规整成 EDL 要求的精确时长。

用法: .venv\\Scripts\\python.exe tools\\dev\\check_realtake.py [计划json]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cs2clipper import compose, config, hlaerec as H          # noqa: E402


def main(argv: list[str]) -> int:
    plan_path = Path(argv[1]) if len(argv) > 1 else \
        config.WORK_DIR / "_hlae_route" / "hlae_plan.json"
    if not plan_path.is_file():
        print(f"找不到计划: {plan_path}")
        return 1

    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = [H.RecSegment(**{k: v for k, v in d.items()
                            if k in H.RecSegment.__dataclass_fields__})
            for d in raw]
    print(f"计划: {plan_path.name}, {len(plan)} 段")

    rec_dir = H.record_output_dir()
    found = H.discover_recordings(rec_dir)
    print(f"\n探测 {rec_dir}")
    print(f"  frame_dirs={len(found['frame_dirs'])}  videos={len(found['videos'])}")
    for v in found["videos"]:
        print(f"    [{v['stream']}] {v['take']}/{v['name']}  "
              f"{v['bytes'] / 2**20:.2f} MB")

    clips_dir = config.WORK_DIR / "_realtake_clips"
    shutil.rmtree(clips_dir, ignore_errors=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    items, notes = H.build_from_recordings(plan, rec_dir, clips_dir,
                                           fps=H.hlae_capture_fps())
    print("\n规整结果:")
    for n in notes:
        print(f"  note: {n}")

    rc = 0
    print(f"\n产出 {len(items)}/{len(plan)} 段:")
    for it in items:
        got = compose.probe_duration(it.path)
        ok = abs(got - it.duration) < 0.15
        print(f"  {Path(it.path).name}: {got:.2f}s  (EDL 要求 {it.duration:.2f}s)"
              f"  {'OK' if ok else '!! 时长不符'}   转场={it.transition}")
        if not ok:
            rc = 1
    if not items:
        print("  **一段都没产出 —— 真录下来的素材还是进不了流水线**")
        rc = 1

    # 抽一帧确认不是黑屏/灰屏
    if items:
        frame = clips_dir / "_probe.png"
        compose.make_thumbnail(items[0].path, frame, at=2.0)
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(frame).convert("RGB"), dtype=float)
        print(f"\n产出的画帧: 分辨率={a.shape[1]}x{a.shape[0]}  "
              f"均值={a.mean():.1f}  标准差={a.std():.1f}")
        if a.std() < 3:
            print("  !! 几乎是纯色, 画面可能是空的")
            rc = 1
        else:
            print("  画面有内容 (非纯色)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
