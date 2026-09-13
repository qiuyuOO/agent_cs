"""Stage 5 —— ffmpeg 合成.

两个层次:
    1. `render_clip_video()` 把一帧序列 + 音频写成单段 MP4 (clip)
    2. `concat_clips()` 把多段拼接成成片, 可选 xfade 转场

为什么要"每段先独立成片再拼接", 而不是把所有帧一次性喂给 ffmpeg:
    * 每段可以有自己的速度/滤镜, 互不影响
    * 渲染可以分段重试 (某一段渲崩了不用从头来)
    * 拼接阶段能用 concat demuxer, 几乎是纯流拷贝, 很快
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import config


def _run(cmd: Sequence[str], *, desc: str = "") -> subprocess.CompletedProcess:
    """执行命令并捕获输出; 失败时把 stderr 尾部带进异常."""
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"ffmpeg 失败 ({desc}, exit={proc.returncode}):\n{tail}")
    return proc


def probe_duration(path: str | Path) -> float:
    """用 ffprobe 读媒体时长 (秒)."""
    probe = config.find_ffprobe()
    if probe is None:
        raise RuntimeError("找不到 ffprobe")
    proc = _run(
        [
            probe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        desc=f"probe {Path(path).name}",
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


# ------------------------------------------------------------------
# 单段渲染
# ------------------------------------------------------------------
def render_clip_video(
    frames: Iterable,
    out_path: str | Path,
    *,
    fps: int = config.FPS,
    duration: float,
    size: tuple[int, int] | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """把帧序列编码成一段**无音轨**的 MP4.

    这里刻意不封装音频: 早期实现给每段都裁一段音乐并用 atempo 校正速度,
    结果是 (a) atempo 要求 0.5~100, 慢放镜头直接报错; (b) 每段各自重采样,
    接缝处的音乐相位对不齐; (c) 段落之间的空隙会变成静音。
    改成整片最后统一铺一遍完整音乐 (见 concat_clips), 音画天然同步,
    也彻底没有接缝问题。

    Args:
        frames: 可迭代的 PIL.Image
        out_path: 输出 mp4
        fps: 帧率
        duration: 该段时长(秒), 用于精确截断
        size: 画布尺寸 (宽, 高); 默认取 config 的默认画幅
    """
    ffmpeg = config.require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(duration, 0.05)
    W, H = size or config.aspect_size(config.DEFAULT_ASPECT)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", f"trim=duration={dur:.3f},setpts=PTS-STARTPTS,format=yuv420p",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-t", f"{dur:.3f}",
        "-an",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=W * H * 3 * 2,
    )
    n = 0
    try:
        assert proc.stdin is not None
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H))
            proc.stdin.write(img.tobytes())
            n += 1
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()

    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"ffmpeg 编码失败 (exit={proc.returncode}):\n{tail}")
    if n == 0:
        raise RuntimeError(f"没有渲染出任何帧: {out_path}")
    return out_path


def finalize_clip(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    target_duration: float,
    size: tuple[int, int] | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """把原始段规整成**精确时长**的片段.

    必要性: 渲染器按 `speed` 生成帧数 (`n_frames = duration/fps/speed`), 慢放
    镜头 (speed < 1) 生成的帧比它的时间窗要少, 于是每段实际时长都比 EDL 声明
    的短一点, 累积起来视频轨比音频轨短好几秒 —— 成片结尾会没画面。

    这里用 tpad 克隆最后一帧补齐差额 (只补不裁), 保证每段严格等于
    target_duration, 从而视频总长 == 音乐总长。
    """
    ffmpeg = config.require_ffmpeg()
    raw_path, out_path = Path(raw_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actual = probe_duration(raw_path)
    deficit = max(target_duration - actual, 0.0)

    vf = "format=yuv420p"
    if deficit > 1e-3:
        # stop_mode=clone: 冻结最后一帧; stop_duration 单位是秒
        vf = f"tpad=stop_mode=clone:stop_duration={deficit:.3f},{vf}"

    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_path),
            "-vf", vf,
            "-t", f"{max(target_duration, 0.05):.3f}",
            "-r", str(config.FPS),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart",
            str(out_path),
        ],
        desc=f"finalize {out_path.name} (补 {deficit:.2f}s)",
    )
    raw_path.unlink(missing_ok=True)
    return out_path


# ------------------------------------------------------------------
# 拼接
# ------------------------------------------------------------------
@dataclass
class ConcatItem:
    path: Path
    duration: float
    transition: str | None = None      # 与**下一段**之间的转场
    transition_duration: float = 0.4


def concat_clips(
    items: Sequence[ConcatItem],
    out_path: str | Path,
    *,
    music_path: str | Path | None = None,
    total_duration: float,
    fade_out: float = 2.5,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """拼接多段视频, 并铺上**整首音乐**.

    关键设计: 视频段用 concat demuxer 直接拼 (纯流拷贝, 快), 然后统一把
    整条音乐编码进去。因为 EDL 的时间轴已经保证各段首尾相接、总长等于
    音乐长度, 所以音画天然对齐, 不需要给每段单独裁音频。
    """
    ffmpeg = config.require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        raise ValueError("没有可拼接的片段")

    total = max(total_duration, 0.5)

    # 清单里的路径**必须转成绝对路径**: concat demuxer 把清单文件所在的目录
    # 当作相对路径的基准, 而清单在 work/ 下 —— 传进来的相对路径 (例如
    # `out/xxx/clips/c1.mp4`) 会被解析成 `work/out/xxx/clips/c1.mp4`, ffmpeg
    # 只报 "Impossible to open ... Error opening input file <清单>" 这种指向
    # 清单本身的误导性错误。同时提前校验文件存在, 免得错误信息指向错的地方。
    missing = [str(it.path) for it in items if not Path(it.path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"待拼接片段不存在 ({len(missing)}/{len(items)}): {missing[:3]}"
        )

    # --- 1. 纯视频拼接 (流拷贝) ---
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8", dir=config.WORK_DIR
    ) as f:
        for it in items:
            p = str(Path(it.path).resolve()).replace("'", "'\\''")
            f.write(f"file '{p}'\n")
        listfile = f.name
    joined = config.WORK_DIR / f"_joined_{out_path.stem}.mp4"
    try:
        _run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", listfile,
                "-c", "copy", "-an", str(joined),
            ],
            desc="concat",
        )
    finally:
        Path(listfile).unlink(missing_ok=True)

    # --- 2. 铺音乐 + 收尾淡出 ---
    has_music = music_path is not None and Path(music_path).is_file()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(joined)]
    if has_music:
        cmd += ["-i", str(music_path)]

    if has_music:
        fo = min(fade_out, total * 0.4)
        cmd += [
            "-filter_complex",
            f"[1:a]atrim=duration={total:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=out:st={max(total - fo, 0):.3f}:d={fo:.3f}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        ]
    else:
        cmd += ["-map", "0:v", "-an"]

    cmd += [
        "-c:v", "copy",
        "-t", f"{total:.3f}",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd, desc="mux music")

    joined.unlink(missing_ok=True)
    return out_path


def make_thumbnail(video_path: str | Path, out_png: str | Path, at: float = 1.0) -> Path:
    """抽一帧当封面 (验收用)."""
    ffmpeg = config.require_ffmpeg()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{at:.2f}", "-i", str(video_path),
            "-frames:v", "1", str(out_png),
        ],
        desc="thumbnail",
    )
    return out_png


def silence_audio(path: str | Path, duration: float, out_path: str | Path) -> Path:
    """生成一段指定时长的静音音轨 (没有音乐素材时的兜底)."""
    ffmpeg = config.require_ffmpeg()
    out_path = Path(out_path)
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{duration:.3f}", "-c:a", "aac", "-b:a", "192k", str(out_path),
        ],
        desc="silence",
    )
    return out_path
