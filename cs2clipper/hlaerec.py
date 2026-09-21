"""HLAE + CS2 游戏内录制 —— 真实游戏画面源 (替代 2D 雷达).

这条链路与雷达渲染的**接口约定完全一致**: 每个 EDL 片段产出一个"仅视频、时长
精确"的 mp4, 再交给 compose.concat_clips 统一铺音乐。所以音乐情绪分析、编排、
拼接铺乐三段完全复用, 只有"画面从哪来"被换掉。

事实来源 (不是凭印象): 本模块用到的 mirv 命令与语义, 是从本机已校验过 SHA256
的 `tools/hlae/x64/AfxHookSource2.dll` 里提取出来的帮助文本, 见
`tools/dev/dump_hlae_cmds.py`。要点:

  * AfxHookSource2 实现 24 个顶层 mirv 命令, 与录制相关的有
    `mirv_streams`(录制) / `mirv_skip`(跳 demo 时间) / `mirv_campath` / `mirv_camio`
  * **没有 `mirv_camimport`** —— 那是 Source 1 的命令, 所以"从文件导入电影级
    运镜"在 CS2 上不可用。全自动能做的只有: 观战视角 + 玩家第一人称 + 平滑跟随
  * `mirv_streams record start|end|name|format|fps|settings`, `mirv_skip time to <s>`

⚠️ 必须由人在桌面上完成的部分 (本模块无法代替):
    1. 装好 HLAE 并在 GUI 里选好 cs2.exe (会写进 hlaeconfig.xml)
    2. 启动 CS2 (HLAE 的 `AvoidVac` 会加 `-insecure`; 会弹 VAC 警告需点确认)
    3. 把生成的 cfg 交给游戏执行, 并等待录制跑完

本模块负责的是: **体检 + 生成可粘贴的指令/cfg + 录完之后把素材接回流水线**。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config, compose
from .demo import TICKRATE


# ------------------------------------------------------------------
# 路径与体检
# ------------------------------------------------------------------
def hlae_dir() -> Path:
    """HLAE 安装目录 (优先用户配置, 否则用项目内 tools/hlae)."""
    from . import store

    try:
        configured = str(store.load_prefs().get("hlae_dir", "") or "").strip()
    except Exception:
        configured = ""
    if configured:
        return Path(configured)
    return config.ROOT / "tools" / "hlae"


def hlae_exe() -> Path:
    return hlae_dir() / "HLAE.exe"


def record_output_dir() -> Path:
    """录制素材输出目录 (优先用户配置, 否则 work/hlae_record)."""
    from . import store

    try:
        configured = str(store.load_prefs().get("hlae_output_dir", "") or "").strip()
    except Exception:
        configured = ""
    if configured:
        return Path(configured)
    return config.WORK_DIR / "hlae_record"


def hlae_capture_fps() -> int:
    from . import store

    try:
        v = int(store.load_prefs().get("hlae_fps", 60) or 60)
    except Exception:
        v = 60
    return max(10, min(v, 240))


def hook_dll() -> Path:
    return hlae_dir() / "x64" / "AfxHookSource2.dll"


def hlae_version() -> str:
    """从 changelog.xml 里读出版本号 (读不到就返回空).

    注意**不能**用宽泛的 `version="..."` 去搜: 文件第一行是
    `<?xml version="1.0" ...?>`, 那样搜出来永远是 "1.0" —— 体检会显示一个
    假版本号 (实测踩到)。真正的版本在 `<release>` 里的 `<version>` 元素。
    """
    for name in ("changelog.xml", "x64/AfxHookSource2_changelog.xml"):
        p = hlae_dir() / name
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:8000]
        except OSError:
            continue
        # 元素形式 (HLAE 的 changelog 用的就是这个)
        m = re.search(r"<version>\s*([^<\s]+)\s*</version>", text)
        if m:
            return m.group(1)
        # 退路: 属性形式, 但要求属性名以 version 结尾 (排除 xml 声明的 version)
        m = re.search(r'\b(?:release|app|build)[-_]?version="([^"]+)"', text, re.I)
        if m:
            return m.group(1)
    return ""


def find_cs2_exe() -> Path | None:
    """在常见 Steam 库位置找 cs2.exe.

    两份安装的情况真实存在 (实测本机 D:\\Steam 那份只有 0.4 GB 是空壳,
    E:\\SteamLibrary 那份 70.5 GB 才是本体), 所以按**目录体积**优先取大的,
    避免让用户对着一个跑不起来的 exe 排查半天。
    """
    from . import store

    try:
        configured = str(store.load_prefs().get("cs2_exe", "") or "").strip()
    except Exception:
        configured = ""
    if configured and Path(configured).is_file():
        return Path(configured)

    candidates: list[Path] = []
    for drive in ("C:", "D:", "E:", "F:", "G:"):
        for lib in (rf"{drive}\Steam\steamapps\common",
                    rf"{drive}\SteamLibrary\steamapps\common",
                    rf"{drive}\Games\Steam\steamapps\common"):
            p = Path(lib) / "Counter-Strike Global Offensive" / "game" / "bin" / "win64" / "cs2.exe"
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None

    def size_of(exe: Path) -> int:
        root = exe.parents[3]           # .../Counter-Strike Global Offensive
        try:
            return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        except OSError:
            return 0
    return max(candidates, key=size_of)


def procs_running() -> dict[str, bool | None]:
    """CS2 / HLAE / Steam 是否在运行 (Windows, 用 tasklist).

    值可以是 **None —— 表示"问不出来", 而不是"没在运行"**。真实教训:
    `tasklist` 在受限环境 (沙箱 / 安全策略 / 部分杀软) 下直接返回
    `ERROR: Access denied` 且 stdout 为空, 早先这里把这种情况当成"三个都没在
    运行", 于是体检报出**假警报** "Steam 没在运行 —— 需要先登录 Steam",
    让人跑去查一个根本不存在的问题 (实测 CS2/HLAE/Steam 三个进程都活着)。
    枚举失败时一律返回 None, 由调用方决定要不要提。
    """
    out: dict[str, bool | None] = {"cs2": None, "hlae": None, "steam": None}
    try:
        r = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=20)
    except (OSError, subprocess.SubprocessError):
        return out
    text = (r.stdout or "").lower()
    if r.returncode != 0 or not text.strip():
        return out
    out["cs2"] = "cs2.exe" in text
    out["hlae"] = "hlae.exe" in text
    out["steam"] = "steam.exe" in text
    return out


@dataclass
class Preflight:
    """体检结果 —— 给 CLI / Web / 自检共用, 不抛异常只报状态."""

    ok: bool = False
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": self.problems,
                "warnings": self.warnings, "info": self.info}

    def text(self) -> str:
        lines = [f"体检结果: {'就绪' if self.ok else '有问题'}"]
        for k, v in self.info.items():
            lines.append(f"  {k:22} = {v}")
        for p in self.problems:
            lines.append(f"  [必须解决] {p}")
        for w in self.warnings:
            lines.append(f"  [提醒]     {w}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# HLAE 配置 (把 cs2.exe 路径写进 HLAE 自己的 hlaeconfig.xml)
# ------------------------------------------------------------------
def hlae_config_path() -> Path:
    return Path.home() / "AppData" / "Roaming" / "HLAE" / "hlaeconfig.xml"


#: HLAE 放外部 ffmpeg 的地方: `<HLAE>/ffmpeg/`。
#: 它的 readme.advancedfx.txt 明确写了两种提供方式:
#:   A) 把 ffmpeg 的 bin/ 放进 `<HLAE>/ffmpeg/`, 即 `<HLAE>/ffmpeg/bin/ffmpeg.exe`
#:   B) 在同目录建 ffmpeg.ini, 内容:
#:        [Ffmpeg]
#:        Path=<绝对路径>\ffmpeg.exe
HLAE_FFMPEG_INI = "[Ffmpeg]\nPath={path}\n"
#: afxFfmpeg* 预设所需的编码器 (yuv420p / lossless / prores)
FFMPEG_NEEDED_ENCODERS = ("libx264", "libx264rgb", "prores_ks")


def find_ffmpeg_for_hlae() -> Path | None:
    """找一个能给 HLAE 用的 ffmpeg.exe (项目自带的优先)."""
    cands = [config.FFMPEG, config.ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"]
    for c in cands:
        p = Path(str(c)) if c else None
        if p and p.is_file():
            return p
    return None


def ffmpeg_supports_hlae_encoders(exe: Path) -> tuple[bool, list[str]]:
    """确认这个 ffmpeg 带 afxFfmpeg* 预设要用的编码器.

    实测教训: HLAE 的 `ffmpeg` 目录**默认是空的** (只有一个 readme), 而
    `afxFfmpegYuv420p` 这类预设**依赖外部 ffmpeg**。没配好的表现是
    "流建起来了、take 目录也建了, 但控制台刷
     `AFXERROR: Failed writing image for screen recording.`、一个画面都没落盘" ——
    看起来在录, 其实全丢。
    """
    import subprocess

    try:
        r = subprocess.run([str(exe), "-hide_banner", "-encoders"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, []
    text = r.stdout or ""
    have = [e for e in FFMPEG_NEEDED_ENCODERS if e in text]
    return len(have) == len(FFMPEG_NEEDED_ENCODERS), have


def setup_hlae_ffmpeg(ffmpeg_exe: str | Path | None = None,
                      hlae_path: str | Path | None = None) -> tuple[Path, str]:
    """把 ffmpeg 路径写给 HLAE (写 ffmpeg.ini, 指向绝对路径).

    用 readme 里的方式 B: 不用复制几十 MB, 升级 ffmpeg 时也只改这一个文件。

    Args:
        ffmpeg_exe: 要用哪个 ffmpeg; 默认自动找 (项目自带的优先)。
        hlae_path: HLAE 目录; 默认按配置取。**测试时传临时目录**, 免得改到真实安装。

    Returns:
        (写入的文件路径, 说明)
    """
    base = Path(hlae_path) if hlae_path else hlae_dir()
    d = base / "ffmpeg"
    d.mkdir(parents=True, exist_ok=True)
    exe = Path(ffmpeg_exe) if ffmpeg_exe else find_ffmpeg_for_hlae()
    if exe is None or not exe.is_file():
        raise FileNotFoundError(
            "找不到可用的 ffmpeg.exe; 可显式传路径, 或把 ffmpeg 解压到 "
            f"{d} 下 (使其成为 {d / 'bin' / 'ffmpeg.exe'})"
        )
    ok, have = ffmpeg_supports_hlae_encoders(exe)
    ini = d / "ffmpeg.ini"
    ini.write_text(HLAE_FFMPEG_INI.format(path=exe), encoding="utf-8")
    note = f"ffmpeg.ini -> {exe} (已有编码器 {have})"
    if not ok:
        missing = [e for e in FFMPEG_NEEDED_ENCODERS if e not in have]
        note += f"; 警告: 缺少 {missing}, afxFfmpeg* 预设可能不可用"
    return ini, note


def set_hlae_cs2_exe(cs2_exe: str | Path | None = None) -> tuple[Path, str]:
    """把 cs2.exe 路径写进 HLAE 的配置, 少让用户手工点一次.

    只做**定点替换** (`<Cs2Exe>...</Cs2Exe>` 与 `<AvoidVac>`), 不整份重写 ——
    HLAE 的配置项很多, 用 XML 解析器序列化回去会丢注释/改格式, 甚至可能因为
    版本差异漏字段。改之前留一份 .bak。

    Returns:
        (配置文件路径, 说明)
    """
    cfg = hlae_config_path()
    exe = Path(cs2_exe) if cs2_exe else find_cs2_exe()
    if exe is None or not Path(exe).is_file():
        raise FileNotFoundError("没找到 cs2.exe, 请显式传路径")
    if not cfg.is_file():
        raise FileNotFoundError(
            f"HLAE 还没生成配置 {cfg} —— 先手动启动一次 HLAE.exe "
            f"(它启动/退出时会写出配置), 或直接用 HLAE 的 GUI 选一次游戏"
        )
    text = cfg.read_text(encoding="utf-8")
    backup = cfg.with_suffix(".xml.bak")
    if not backup.is_file():
        shutil.copy2(cfg, backup)

    notes: list[str] = []
    # 替换串必须用**函数**: Windows 路径里的 `\S`、`\U` 会被 re 当成转义模板,
    # 直接传字符串会报 "bad escape \S" (实测踩到)。
    new, n = re.subn(r"<Cs2Exe>.*?</Cs2Exe>",
                     lambda _m: f"<Cs2Exe>{exe}</Cs2Exe>",
                     text, flags=re.S)
    if n == 0:
        raise RuntimeError("配置里没有 <Cs2Exe> 节点, HLAE 版本可能不同; 请手动在 GUI 里选")
    notes.append(f"Cs2Exe -> {exe}")
    text = new

    # AvoidVac=true 才会带 -insecure, 否则注入会被 VAC 拦下
    if "<AvoidVac>true</AvoidVac>" not in text:
        if "<AvoidVac>false</AvoidVac>" in text:
            text = text.replace("<AvoidVac>false</AvoidVac>", "<AvoidVac>true</AvoidVac>")
        else:
            text = text.replace("</Settings>", "  <AvoidVac>true</AvoidVac>\n</Settings>", 1) \
                if "</Settings>" in text else text
        notes.append("AvoidVac -> true (注入需要 -insecure)")

    # 录制分辨率: 默认 1920x1080 非全屏, 避免抢焦点且便于无头录制
    if "<GfxFull>true</GfxFull>" in text:
        text = text.replace("<GfxFull>true</GfxFull>", "<GfxFull>false</GfxFull>")
        notes.append("GfxFull -> false (窗口化, 避免抢焦点)")

    cfg.write_text(text, encoding="utf-8")
    return cfg, "; ".join(notes)


def preflight(*, check_running: bool = True) -> Preflight:
    """检查 HLAE 录制所需的一切是否就绪."""
    pf = Preflight()
    d = hlae_dir()
    pf.info["hlae_dir"] = str(d)
    pf.info["hlae_version"] = hlae_version()

    if not hlae_exe().is_file():
        pf.problems.append(
            f"没找到 HLAE.exe (期望在 {hlae_exe()}); 下载: "
            f"https://github.com/advancedfx/advancedfx/releases"
        )
    if not hook_dll().is_file():
        pf.problems.append(f"没找到 CS2 注入器 {hook_dll()} —— CS2 录制靠它, "
                           f"缺了只能录 CS:GO")

    cs2 = find_cs2_exe()
    pf.info["cs2_exe"] = str(cs2) if cs2 else None
    if cs2 is None:
        pf.problems.append("没在常见 Steam 库位置找到 cs2.exe, 可用 --set cs2_exe <路径> 指定")
    else:
        # 空壳安装是真实存在的坑
        root = cs2.parents[3]
        try:
            gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30
        except OSError:
            gb = 0.0
        pf.info["cs2_size_gb"] = round(gb, 1)
        if gb < 5:
            pf.problems.append(
                f"{root} 只有 {gb:.1f} GB, 看起来是空壳安装 (完整安装约 70 GB); "
                f"换用体积最大的那份"
            )

    cfg = hlae_config_path()
    pf.info["hlae_config"] = str(cfg)
    if cfg.is_file():
        try:
            text = cfg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        m = re.search(r"<Cs2Exe>([^<]*)</Cs2Exe>", text)
        chosen = (m.group(1) if m else "").strip()
        pf.info["hlae_cs2_exe_setting"] = chosen
        if not chosen or chosen == "请选择" or not Path(chosen).is_file():
            pf.warnings.append(
                "HLAE 里还没选好 cs2.exe (hlaeconfig.xml 的 Cs2Exe) —— "
                "需要打开 HLAE.exe → 选择游戏 → CS2, 选一次即可"
            )
        if "AvoidVac>true" not in text:
            pf.warnings.append("HLAE 的 AvoidVac 不是 true, 注入可能失败 (需要 -insecure)")
    else:
        pf.warnings.append(f"HLAE 还没生成配置 ({cfg}); 先手动启动一次 HLAE.exe")

    # --- 外部 ffmpeg: afxFfmpeg* 预设的前提 ---
    # 实测教训: HLAE 的 ffmpeg 目录默认是空的 (只有 readme), 没配的话
    # 流能建起来、take 目录也建, 但控制台刷
    #   AFXERROR: Failed writing image for screen recording.
    # 一个画面都不落盘 —— 看起来在录, 其实全丢。这一条必须在**开录之前**报出来。
    ff_dir = hlae_dir() / "ffmpeg"
    ini = ff_dir / "ffmpeg.ini"
    bundled = ff_dir / "bin" / "ffmpeg.exe"
    pf.info["hlae_ffmpeg_ini"] = str(ini) if ini.is_file() else None
    pf.info["hlae_ffmpeg_bundled"] = bundled.is_file()
    if not ini.is_file() and not bundled.is_file():
        pf.problems.append(
            f"HLAE 没有可用的 ffmpeg: {ff_dir} 下既没有 bin/ffmpeg.exe 也没有 "
            f"ffmpeg.ini。afxFfmpeg* 预设依赖外部 ffmpeg, 缺了会一路刷 "
            f"'AFXERROR: Failed writing image for screen recording.' 且录不到任何画面。"
            f" 修法: python -m cs2clipper.pipeline --hlae-setup-ffmpeg"
        )
    else:
        target = bundled
        if ini.is_file():
            try:
                m2 = re.search(r"Path\s*=\s*(.+)", ini.read_text(encoding="utf-8"))
                if m2:
                    target = Path(m2.group(1).strip())
            except OSError:
                pass
        if not Path(target).is_file():
            pf.problems.append(f"ffmpeg.ini 指向的 ffmpeg 不存在: {target}")
        else:
            ok_enc, have = ffmpeg_supports_hlae_encoders(Path(target))
            pf.info["hlae_ffmpeg"] = str(target)
            pf.info["hlae_ffmpeg_encoders"] = have
            if not ok_enc:
                missing = [e for e in FFMPEG_NEEDED_ENCODERS if e not in have]
                pf.warnings.append(
                    f"HLAE 用的 ffmpeg ({target}) 缺少编码器 {missing}; "
                    f"afxFfmpeg* 预设可能失败"
                )

    if check_running:
        run = procs_running()
        pf.info["running"] = run
        if run["steam"] is None:
            # 问不出来就说问不出来 —— 不能当成"没在运行"去报警
            pf.info["running_note"] = "tasklist 被拒, 未判断进程状态 (不是问题)"
        elif not run["steam"]:
            pf.warnings.append("Steam 没在运行 —— HLAE 启动 CS2 需要先登录 Steam")

    pf.ok = not pf.problems
    return pf


# ------------------------------------------------------------------
# 录制计划: EDL → 每段在 demo 里的取景与时间
# ------------------------------------------------------------------
@dataclass
class RecSegment:
    """一个 EDL 片段对应的录制指令."""

    index: int
    highlight_id: str
    player: str
    round_num: int
    # demo 内时间 (秒) —— mirv_skip time to 用的就是它
    demo_start_sec: float
    demo_end_sec: float
    # 该片段在成片里占的时长 (秒), 由 EDL 决定
    out_duration: float
    speed: float                 # EDL 要求的播放倍速 (>1 快放, <1 慢放)
    name: str = ""               # mirv_streams record name
    # 与**下一段**之间的转场 (原样从 EDL 带过来, 保证两条画面源产物等价)
    transition: str | None = None
    transition_duration: float = 0.4

    @property
    def demo_span(self) -> float:
        """素材在 demo 里跨越的真实时长 (秒)."""
        return max(self.demo_end_sec - self.demo_start_sec, 0.01)

    @property
    def realtime_factor(self) -> float:
        """demo 素材时长 / 成片时长.

        >1 = 素材比成片长 → 录制产物要**快放**压缩;
        <1 = 素材比成片短 → 要**慢放**拉伸 (慢镜头);
        =1 = 原速。
        真正的变速由 frames_to_clip 用 setpts 完成, 这里只是给人看的诊断量。
        """
        return self.demo_span / max(self.out_duration, 0.01)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "highlight_id": self.highlight_id,
            "player": self.player, "round_num": self.round_num,
            "demo_start_sec": round(self.demo_start_sec, 3),
            "demo_end_sec": round(self.demo_end_sec, 3),
            "demo_span": round(self.demo_span, 3),
            "out_duration": round(self.out_duration, 3),
            "speed": self.speed,
            "realtime_factor": round(self.realtime_factor, 4),
            "name": self.name,
            "transition": self.transition,
            "transition_duration": self.transition_duration,
        }


def plan_from_edl(edl, highlights: Sequence[Any]) -> list[RecSegment]:
    """把 EDL 翻译成录制计划.

    注意 EDL 的 `src_start_tick/src_end_tick` 是 **demo tick**, 而 `out_start/
    out_end` 是成片时间轴。录制的取景窗口应当用前者, 时长用后者 —— 两者之比
    就是 ffmpeg 需要做的变速比。

    **转场也要一起带过来**: compose.concat_clips 目前只做硬切 + 段首尾淡入淡出,
    但它的契约里是有 transition 的。雷达路径会把它传下去, HLAE 路径若丢掉,
    两条路的产物就不等价了 (以后 concat_clips 真做转场时, HLAE 路径会静默少一层)。
    """
    by_id = {c.id: c for c in highlights}
    out: list[RecSegment] = []
    for i, cl in enumerate(edl.clips):
        card = by_id.get(cl.highlight_id)
        if card is None:
            continue
        out.append(
            RecSegment(
                index=i,
                highlight_id=cl.highlight_id,
                player=getattr(card, "player", "") or "",
                round_num=int(getattr(card, "round_num", 0) or 0),
                demo_start_sec=cl.src_start_tick / TICKRATE,
                demo_end_sec=cl.src_end_tick / TICKRATE,
                out_duration=cl.duration,
                speed=float(cl.speed or 1.0),
                name=f"clip_{i + 1:03d}",
                transition=getattr(cl, "transition", None),
                transition_duration=float(getattr(cl, "transition_duration", 0.4) or 0.4),
            )
        )
    return out


# ------------------------------------------------------------------
# 生成 CS2 指令 / cfg
# ------------------------------------------------------------------
#: 这些 cvar 我**没有**在本机核实过 (CS2 与 CS:GO 的 cvar 名不完全一样)。
#: 所以默认只生成"确定有效"的那几条, 其余归到可选段, 由用户按需打开。
#: 宁可少写几条, 也不要凭印象写一堆 CS2 里不存在的命令去误导人。
#: (注意别写成 `("...",)` —— 末尾那个逗号会把它变成**元组**, 后面 join 时才炸)
CERTAIN_PREFIX = "// ---- 以下命令来自 AfxHookSource2.dll 的帮助文本, 语义确定 ----"
OPTIONAL_PREFIX = (
    "// ---- 以下为可选项: CS2 与 CS:GO 的 cvar 名不完全一致, 未在本机核实,"
    " 若报 Unknown command 属正常, 删掉那一行即可 ----"
)


#: 画面流的名字 (mirv_streams add normal <这个名字>)。录制产物会落到
#: `<录制目录>/<这个名字>/takeNNNN/` 下。
STREAM_NAME = "cs2clipper"

#: 每个流要用的录制设置。
#: 可用内置预设 (来自 DLL): afxFfmpeg / afxFfmpegYuv420p / afxFfmpegLosslessBest /
#: afxFfmpegLosslessFast / afxFfmpegProres / afxFfmpegProresAlpha / afxFfmpegRaw /
#: afxFfmpegHuffyuv。
#: 默认用 Yuv420p: 有损但足够 (成片最后也编码成 yuv420p), 文件比无损小得多。
#: 想要无损改成 "afxFfmpegLosslessFast"。
STREAM_SETTINGS = "afxFfmpegYuv420p"


def build_cs2_config(
    plan: Sequence[RecSegment],
    *,
    demo_path: str,
    output_dir: str,
    fps: int = 60,
    start_index: int = 0,
    include_optional: bool = True,
) -> tuple[str, str, list[tuple[str, str]]]:
    """生成录制用的 CS2 脚本.

    返回 `(引导脚本, 停录脚本, [(片段脚本名, 内容), ...])`。

    ⚠️ 为什么不是"一个脚本全自动录完":
      1. 一条 `mirv_streams record end` 必须在**该片段的 demo 时间走到头**时执行。
         能"在指定 demo 时间自动执行命令"的 `mirv_cmd` 虽然出现在 DLL 的注册
         命令表里, 但**帮助文本一条都没有** —— 我无法核实它在本版
         AfxHookSource2 里的语法是否可用。把自动化建在没核实过的命令上, 结果
         是"看起来在录、其实一段都没录到", 比让人按键更糟。
      2. 录制输出目录由 `mirv_streams record name` 决定。如果整份脚本只在开头
         设一次名字, 那么每段的输出都会落到**同一个目录并被下一段覆盖** ——
         22 段最后只剩最后一段 (这是我在写第一版时真的犯过的错)。
         所以必须"一段一个独立名字"。

    因此: 一段一个 cfg。操作者按顺序 exec, 每段末尾按一次绑好的键停录。
    22 段 = 22 次 exec + 22 次按键, 全程可见, 出问题立刻能发现。

    只使用 AfxHookSource2.dll 帮助文本里**确定存在**的命令:
    `mirv_streams record screen enabled`、`record name/fps/format/startMovieWav/
    start/end`、`mirv_skip time to`、`demo_pause/demo_resume`、`playdemo`、
    `spec_player`、`bind`。

    ⚠️ **必须先 `add` 一个画面流**。这里踩过两次坑 (都是用户真机跑出来的):
      1. 只设 `record name` / `fps` 就 start —— 三个 take 里只有 audio.wav。
      2. 加了 `mirv_streams record screen enabled 1` 仍然不行 —— 用户跑
         `mirv_streams print` 得到 **"Total streams: 0"**, 说明那个命令
         **不会创建流**, 它只是配置一个已存在的"屏幕流"。
      正确做法 (语法来自 DLL):
        mirv_streams add normal <名字>                     创建流
        mirv_streams edit <名字> settings <预设>            指定录制设置
        mirv_streams edit <名字> enabled 1 / record 1      打开并纳入录制
    """
    segs = list(plan[start_index:])
    name_prefix = Path(output_dir).name or "hlae_record"
    stream_name = STREAM_NAME

    bootstrap: list[str] = []
    bootstrap.append("// cs2clipper 自动生成的 HLAE 录制引导脚本")
    bootstrap.append(f"// demo: {demo_path}")
    bootstrap.append(f"// 片段数: {len(segs)}   录制帧率: {fps}")
    bootstrap.append(f"// 录制输出根目录: {output_dir}")
    bootstrap.append("")
    bootstrap.append("// 用法: CS2 控制台里依次执行")
    bootstrap.append("//   1) exec cs2clipper_bootstrap.cfg      (放 demo + 全局设置)")
    bootstrap.append("//   2) exec cs2clipper_clip_001.cfg       (录第 1 段)")
    bootstrap.append("//   3) 画面走到该段末尾时按 F8 停录")
    bootstrap.append("//   4) exec cs2clipper_clip_002.cfg ... 以此类推")
    bootstrap.append("//")
    bootstrap.append("// 更省事的办法: 第一次 exec 之后, 在控制台按 ↑ 调出上一条命令,")
    bootstrap.append("// 只改末尾的数字; 或者用下面的 bind 把「停录」固定到 F8。")
    bootstrap.append("")
    bootstrap.append(CERTAIN_PREFIX)
    # ⚠️ 必须**先 add 一个流**, 否则什么画面都录不到。
    # 实测教训 (用户真机跑出来的):
    #   1) `record screen enabled 1` **不会创建流** —— 跑完 `mirv_streams print`
    #      显示 "Total streams: 0", 于是 record start 之后只有 audio.wav。
    #   2) 正确做法是 add 一个流 (类型 normal = 完整画面):
    #        %s normal|depth|hudBlack|... <sUniqueStreamName> - Adds a stream of given type.
    #      然后给它指定录制设置:
    #        %s settings <name> - Set recording settings to use from mirv_streams settings.
    #      可用的内置预设见 DLL 里的字符串:
    #        afxFfmpeg / afxFfmpegYuv420p / afxFfmpegLosslessBest / afxFfmpegLosslessFast
    #        / afxFfmpegProres / afxFfmpegProresAlpha / afxFfmpegRaw / afxFfmpegHuffyuv
    bootstrap.append(f"mirv_streams add normal {stream_name}")
    # 有损 yuv420p 足够 (成片最后也编码成 yuv420p); 想无损可以换成
    # afxFfmpegLosslessFast, 代价是文件大得多
    bootstrap.append(f"mirv_streams edit {stream_name} settings {STREAM_SETTINGS}")
    bootstrap.append(f"mirv_streams edit {stream_name} enabled 1")
    bootstrap.append(f"mirv_streams edit {stream_name} record 1")
    # 音频另配 (我们最后统一铺音乐, 不需要录到的游戏音)
    bootstrap.append("mirv_streams record startMovieWav 0")
    bootstrap.append("mirv_streams record format tga")
    bootstrap.append(f'mirv_streams record name "{name_prefix}"')
    bootstrap.append(f"mirv_streams record fps {int(fps)}")
    bootstrap.append("bind F8 \"exec cs2clipper_stop.cfg\"")
    bootstrap.append("")
    bootstrap.append("// 自检: 这里应当列出 1 个流且 enabled=1 / record=1。")
    bootstrap.append("// 若仍显示 'Total streams: 0', 把下面几行输出发出来。")
    bootstrap.append("mirv_streams settings print")
    bootstrap.append("mirv_streams print")
    bootstrap.append("")
    bootstrap.append("// 先把 demo 放起来 (路径里的反斜杠写成双斜杠)")
    bootstrap.append(f'playdemo "{demo_path}"')
    bootstrap.append("")
    if include_optional:
        bootstrap.append(OPTIONAL_PREFIX)
        bootstrap.append("sv_cheats 1")
        bootstrap.append("demo_ui 0                // 隐藏回放控制条")
        bootstrap.append("cl_draw_only_deathnotices 1   // 只留击杀提示, 画面干净")
        bootstrap.append("spec_show_xray 0")
        bootstrap.append("")
    bootstrap.append("// 放起来之后再逐段 exec; 每段的定位/开录在各自的脚本里")

    stop = "\n".join([
        "// cs2clipper: 由 F8 调用 —— 停止当前片段的录制",
        "// (CS2 的控制台脚本没有条件/游标, 所以'停'和'进下一段'拆成两步:",
        "//  按 F8 停录, 再从下面清单里 exec 下一段的脚本)",
        "mirv_streams record end",
        'echo "[cs2clipper] 已停止当前片段, 请 exec 下一段的 clip_XXX.cfg"',
    ]) + "\n"

    clips: list[tuple[str, str]] = []
    for i, seg in enumerate(segs):
        lines: list[str] = []
        lines.append(f"// cs2clipper 片段 {seg.index + 1}/{len(plan)}"
                     f"   R{seg.round_num} {seg.player}")
        lines.append(f"// demo {seg.demo_start_sec:.2f}~{seg.demo_end_sec:.2f}s"
                     f"  →  成片 {seg.out_duration:.2f}s ({seg.speed}x)")
        lines.append("// 录完这一段时按 F8 停录")
        lines.append("")
        lines.append(CERTAIN_PREFIX)
        # 每段一个**独立**的 record name: 否则各段输出会互相覆盖, 最后只剩最后一段
        lines.append(f'mirv_streams record name "{name_prefix}_{seg.name}"')
        lines.append("demo_pause")
        lines.append(f"mirv_skip time to {max(seg.demo_start_sec, 0.0):.3f}")
        if seg.player:
            lines.append(f'spec_player "{_escape(seg.player)}"')
        lines.append("demo_resume")
        lines.append("mirv_streams record start")
        clips.append((f"cs2clipper_clip_{i + 1:03d}.cfg", "\n".join(lines) + "\n"))

    return ("\n".join(bootstrap) + "\n", stop, clips)


def _escape(name: str) -> str:
    """把玩家名放进双引号里: 名里可能有引号/分号, 不转义会破坏 cfg."""
    return name.replace('"', "'").replace(";", ",").strip()


def cs2_cfg_dir() -> Path:
    """CS2 的 cfg 目录 (放进这里才能被 exec 到)."""
    cs2 = find_cs2_exe()
    if cs2 is None:
        return config.ROOT / "work"
    d = cs2.parents[3] / "game" / "csgo" / "cfg"
    return d if d.is_dir() else config.ROOT / "work"


def write_cs2_config(text: str, *, filename: str = "cs2clipper_record.cfg",
                     target_dir: Path | None = None) -> Path:
    """把单个 cfg 写进 CS2 的 cfg 目录 (找不到就退回项目 work/ 并提示)."""
    d = Path(target_dir) if target_dir else cs2_cfg_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(text, encoding="utf-8")
    return p


def write_cs2_scripts(
    demo_path: str,
    plan: Sequence[RecSegment],
    *,
    output_dir: str | Path | None = None,
    fps: int = 60,
    target_dir: Path | None = None,
    fallback_dir: Path | None = None,
) -> dict[str, Any]:
    """生成并写出整套录制脚本 (引导 + 停录 + 每段一个).

    写入位置按顺序尝试: 显式 `target_dir` → CS2 的 cfg 目录 → `fallback_dir`
    (默认 out_dir/cs2_cfg)。**必须**有兜底: CS2 可能装在需要管理员权限的目录
    (实测本机 `E:\\SteamLibrary\\...\\game\\csgo\\cfg` 就直接 PermissionError),
    没有兜底的话整个出片流程会被一个"写不进去"的辅助文件搞崩。

    返回一份清单, 直接给 CLI/Web 展示"接下来要 exec 哪些文件"。
    """
    out_dir = Path(output_dir) if output_dir else record_output_dir()
    bootstrap, stop, clips = build_cs2_config(
        plan, demo_path=demo_path, output_dir=str(out_dir), fps=fps,
    )

    attempts: list[Path] = []
    if target_dir is not None:
        attempts.append(Path(target_dir))
    attempts.append(cs2_cfg_dir())
    if fallback_dir is not None:
        attempts.append(Path(fallback_dir))
    else:
        attempts.append(out_dir.parent / "cs2_cfg")

    target: Path | None = None
    notes: list[str] = []
    last_err: Exception | None = None
    for cand in attempts:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".cs2clipper_write_test"
            probe.write_text("x", encoding="utf-8")
            probe.unlink(missing_ok=True)
            target = cand
            break
        except OSError as exc:
            last_err = exc
            notes.append(f"{cand} 不可写 ({type(exc).__name__}), 换下一个位置")
    if target is None:
        raise RuntimeError(
            f"找不到可写位置放 CS2 脚本 (试过 {[str(a) for a in attempts]}): {last_err}"
        )

    written: list[str] = []
    written.append(str(write_cs2_config(bootstrap, filename="cs2clipper_bootstrap.cfg",
                                        target_dir=target)))
    written.append(str(write_cs2_config(stop, filename="cs2clipper_stop.cfg",
                                        target_dir=target)))
    for name, text in clips:
        written.append(str(write_cs2_config(text, filename=name, target_dir=target)))

    in_cs2 = target == cs2_cfg_dir()
    if not in_cs2:
        notes.append(
            f"脚本写在 {target} (不是 CS2 的 cfg 目录) —— CS2 的 exec 只在它自己的 "
            f"cfg 目录里找文件, 所以需要把这些 .cfg 复制到 {cs2_cfg_dir()}, "
            f"或在控制台里用绝对路径 exec"
        )
    return {
        "cfg_dir": str(target),
        "in_cs2_cfg_dir": in_cs2,
        "notes": notes,
        "bootstrap": written[0],
        "stop": written[1],
        "clips": written[2:],
        "record_dir": str(out_dir),
        "segments": len(plan),
        "fps": fps,
    }


def write_plan(plan: Sequence[RecSegment], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([s.to_dict() for s in plan], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ------------------------------------------------------------------
# 录完之后: 素材探测 + 规整成 EDL 要求的片段
# ------------------------------------------------------------------
IMAGE_EXT = (".tga", ".bmp", ".png", ".jpg", ".jpeg")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def _match_stream(seg: "RecSegment", names) -> str | None:
    """把录制产物对回计划里的片段.

    真实缺陷: cfg 里生成的 record name 是 `hlae_record_clip_001`
    (输出目录名 + 片段名, 见 build_cs2_config), 而一开始这里只会做
    `name.startswith(seg.name)` —— `hlae_record_clip_001`.startswith(`clip_001`)
    是 False, 于是**真录完了也一段都匹配不上**, 成片直接报"没有可用素材"。

    所以按"先精确、再后缀、最后才退化到包含"的顺序匹配。
    """
    names = list(names)
    if not names:
        return None
    if seg.name in names:
        return seg.name
    for n in names:                       # 产物名通常以片段名结尾
        if n.endswith(seg.name):
            return n
    for n in names:                       # 兜底: 名字里带片段名
        if seg.name in n:
            return n
    return None


def cs2_work_dirs() -> list[Path]:
    """CS2 实际的工作目录 —— `mirv_streams record name` 的**相对路径基准**.

    实测教训: 引导脚本里给的是绝对路径 `E:\\agent_cs\\work\\hlae_record`,
    但 HLAE 仍把帧写到了
        <游戏>\\game\\bin\\win64\\<name>\\
    也就是**把 record name 当成相对路径**处理 (相对 CS2 的工作目录, 而
    HLAE 启动 CS2 时的工作目录就是 win64)。所以探测录制产物时必须把这些
    位置也算进去, 否则"其实录到了, 但流水线说没有素材"。
    """
    out: list[Path] = []
    cs2 = find_cs2_exe()
    if cs2 is not None:
        out.append(cs2.parent)                       # ...\game\bin\win64
        out.append(cs2.parents[2])                   # ...\game
    return out


def discover_recordings(output_dir: str | Path) -> dict[str, Any]:
    """看录制目录里有什么 —— 用来判断"录成功了没".

    HLAE 的 mirv_streams 会按 stream 名建目录, 里面再按 take 分子目录:
        <name>/take0000/frame_00000000.tga ...
    所以要往下钻一层找帧序列; 同时也会认已经编码好的视频文件。

    **两个位置都找**: 配置的输出目录, 以及 CS2 自己的工作目录 (相对路径会把
    产物丢在那里)。只查前者会出现"录到了却报没有素材"。
    """
    roots: list[Path] = [Path(output_dir).resolve()]
    for d in cs2_work_dirs():
        d = Path(d).resolve()
        roots.append(d)
        roots.append(d / Path(str(output_dir)).name)

    res: dict[str, Any] = {
        "dir": str(output_dir), "exists": False,
        "searched": sorted({str(r) for r in roots}),
        "frame_dirs": [], "videos": [], "total_frames": 0,
    }
    seen: set[str] = set()

    for root in roots:
        if not root.is_dir():
            continue
        res["exists"] = True
        # 候选: root 下的一级目录 (stream 名), 以及 root 自身 (帧直接放这儿)
        cands = [root] + [p for p in sorted(root.iterdir()) if p.is_dir()]
        for stream_dir in cands:
            # take 子目录优先; 没有 take 就直接用 stream_dir
            takes = [p for p in sorted(stream_dir.iterdir())
                     if p.is_dir() and p.name.lower().startswith("take")]
            for holder in (takes or [stream_dir]):
                try:
                    entries = list(holder.iterdir())
                except OSError:
                    continue
                frames = [f for f in entries if f.suffix.lower() in IMAGE_EXT]
                if frames:
                    key = str(holder)
                    if key in seen:
                        continue
                    seen.add(key)
                    res["frame_dirs"].append({
                        "name": stream_dir.name, "take": holder.name,
                        "path": key, "frames": len(frames),
                        "first": sorted(f.name for f in frames)[0],
                        "ext": frames[0].suffix.lower(),
                    })
                    res["total_frames"] += len(frames)
                for f in entries:
                    if f.suffix.lower() in VIDEO_EXT and f.is_file():
                        key = str(f)
                        if key in seen:
                            continue
                        seen.add(key)
                        # `stream` 才是回对片段用的键。`name` 只是文件名
                        # (ffmpeg 类预设写出的是 `take0000/take0000.mp4`),
                        # 拿文件名去匹配 `clip_001` 永远匹配不上 —— 真录完了
                        # 也会被判成"没有素材"。
                        res["videos"].append({"name": f.name, "path": key,
                                              "stream": stream_dir.name,
                                              "take": holder.name,
                                              "bytes": f.stat().st_size})
    return res


def _default_size(aspect: str | None) -> tuple[int, int]:
    """画布尺寸.

    这里**不**直接用 config.aspect_size(aspect): 它的入参不允许 None
    (aspect_size(None) 会 KeyError)。调用方 (pipeline node_render_hlae) 传的是
    state 里的 aspect, 可能为空, 所以统一在这里兜底成默认画幅。
    """
    key = aspect if aspect else config.DEFAULT_ASPECT
    try:
        return config.aspect_size(key)
    except KeyError:
        return config.aspect_size(config.DEFAULT_ASPECT)


def frames_to_clip(
    frame_dir: str | Path,
    out_path: str | Path,
    *,
    fps: int,
    target_duration: float,
    size: tuple[int, int] | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """帧序列 → 精确时长的无音轨 mp4 (与雷达路径输出契约一致).

    变速逻辑: HLAE 按"录制 fps"抓帧, 一段 demo 里 t 秒的素材会得到约
    `t * capture_fps` 帧。要在成片里占 `target_duration` 秒, 就用
    `setpts=PTS * (录制时长 / 目标时长)` 整体重采样 —— 这就是慢放/快放。

    **两种情况都要处理**, 因为录制素材的长度由人工按停录键的时机决定:
      * 录短了 → 慢放拉伸 (setpts 放大)
      * 录长了 → 快放压缩 (setpts 缩小)。实测第一次录制留下 5165 帧 / 86.08 秒,
        而该片段只需要 3.6 秒 —— 只补不裁的 finalize_clip 处理不了这种超长素材,
        86 秒会原样进成片。所以这里用 compose.fit_clip (两端都管)。

    用 ffmpeg 的 image2 demuxer 读帧序列, 不引入 imageio/PIL 逐帧读取
    (一秒 60 帧、一段 4 秒就是 240 张 1080p 图, 走 Python 循环会明显变慢)。
    """
    frame_dir, out_path = Path(frame_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if not frames:
        raise FileNotFoundError(f"目录里没有帧序列: {frame_dir}")

    first = frames[0].name
    m = re.search(r"(\d+)(?=\D*$)", first)
    start_number = int(m.group(1)) if m else 0
    pattern = re.sub(r"\d+(?=\D*$)", "%08d", first)

    ffmpeg = config.require_ffmpeg()
    raw = out_path.with_name("_raw_" + out_path.name)
    W, H = size or _default_size(None)

    # 先把帧序列编码成"捕获帧率"下的视频
    _ffmpeg([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-start_number", str(start_number),
        "-i", str(frame_dir / pattern),
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        str(raw),
    ], desc=f"encode frames {frame_dir.name}")

    # 再拉伸/压缩到精确时长 (两端都处理)
    try:
        return compose.fit_clip(
            raw, out_path, target_duration=target_duration, size=(W, H),
            crf=crf, preset=preset,
        )
    finally:
        raw.unlink(missing_ok=True)


def _ffmpeg(cmd: list[str], *, desc: str = "") -> None:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise RuntimeError(f"ffmpeg 失败 ({desc}):\n{tail}")


def _safe_duration(path: str | Path) -> float:
    """探时长, 失败就返回 0.0 —— 只用来在多个 take 之间排序, 不值得抛异常."""
    try:
        return compose.probe_duration(path)
    except Exception:                                  # noqa: BLE001
        return 0.0


def _best_video(cands: Sequence[dict[str, Any]]) -> Path:
    """同一个 stream 有多个 take 时挑一个用.

    停录时机由人工掌握, 所以同一个片段可能有补录出来的好几个 take
    (`take0000`, `take0001` ...)。取**时长最长**的那个: 它最可能包含完整动作,
    再由 fit_clip 压/拉到目标时长。时长一个都探不出来时退回编号最大的
    (take 号越大越新)。
    """
    if len(cands) == 1:
        return Path(cands[0]["path"])
    best = max(cands, key=lambda v: _safe_duration(v["path"]))
    if _safe_duration(best["path"]) <= 0:
        return Path(cands[-1]["path"])
    return Path(best["path"])


def build_from_recordings(
    plan: Sequence[RecSegment],
    record_dir: str | Path,
    clips_dir: str | Path,
    *,
    fps: int = 60,
    size: tuple[int, int] | None = None,
) -> tuple[list[compose.ConcatItem], list[str]]:
    """把录制素材按计划规整成 ConcatItem 列表 (可直接交给 concat_clips).

    返回 (items, 问题说明)。缺素材的片段**跳过并报出来**, 而不是静默少一段 ——
    否则成片会比音乐短一截且没有解释。
    """
    record_dir = Path(record_dir)
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    found = discover_recordings(record_dir)
    # name -> [该 stream 的全部 take, 按 take 名排序]。一个 stream 会有多个 take
    # (补录, 或某次按 F8 早了/晚了)。
    #
    # 只取**帧最多的那个 take** 当素材, 不是把所有 take 拼起来: 各 take 是独立
    # 目录、帧文件名从 0 重新编号, 硬拼会撞号且顺序无法保证。取最饱满的一个,
    # 再由 fit_clip 压/拉到目标时长 —— 素材短了慢放、长了快放, 两头都管得住。
    by_name: dict[str, list[dict[str, Any]]] = {}
    for d in found["frame_dirs"]:
        by_name.setdefault(d["name"], []).append(d)
    for v in by_name.values():
        v.sort(key=lambda x: x.get("take", ""))

    # 视频产物同样按 **stream 名** 归类。ffmpeg 类预设 (afxFfmpegYuv420p 等)
    # 写出来的是 `take0000/take0000.mp4` 这种文件, 文件名跟片段名毫无关系,
    # 所以键必须是 stream 目录名而不是文件名。
    by_video: dict[str, list[dict[str, Any]]] = {}
    for v in found["videos"]:
        by_video.setdefault(v.get("stream") or v["name"], []).append(v)
    for v in by_video.values():
        v.sort(key=lambda x: x.get("take", ""))

    items: list[compose.ConcatItem] = []
    notes: list[str] = []

    # 转场必须跟着走: 雷达路径是把 EDL 的 transition 原样交给 ConcatItem 的,
    # 这里若丢掉, 两条画面源的产物就不等价 (将来 concat_clips 真做转场时,
    # HLAE 路径会静默少一层)。
    def _item(seg: "RecSegment", path: Path) -> compose.ConcatItem:
        return compose.ConcatItem(
            path=path, duration=seg.out_duration,
            transition=seg.transition,
            transition_duration=seg.transition_duration,
        )

    for seg in plan:
        # 产物名可能是 `clip_001`, 也可能是 `hlae_record_clip_001` —— 见 _match_stream
        hit = _match_stream(seg, by_name)
        if hit:
            out = clips_dir / f"clip_{seg.index + 1:03d}.mp4"
            takes = by_name[hit]
            best = max(takes, key=lambda t: int(t.get("frames") or 0))
            src = Path(best["path"])
            if len(takes) > 1:
                notes.append(f"{seg.name}: 有 {len(takes)} 个 take, 取帧最多的 "
                             f"{best['take']} ({best['frames']} 帧)")
            frames_to_clip(src, out, fps=fps,
                           target_duration=seg.out_duration, size=size)
            items.append(_item(seg, out))
            continue

        vhit = _match_stream(seg, by_video)
        if vhit:
            out = clips_dir / f"clip_{seg.index + 1:03d}.mp4"
            cands = by_video[vhit]
            src = _best_video(cands)
            actual = _safe_duration(src)
            extra = f", 共 {len(cands)} 个 take 取最长的" if len(cands) > 1 else ""
            notes.append(f"{seg.name}: 用录好的视频 {src.name} ({actual:.2f}s), "
                         f"目标 {seg.out_duration:.2f}s{extra}")
            # 同样用 fit_clip: 录好的视频也可能比目标长 (取决于停录时机)
            compose.fit_clip(src, out, target_duration=seg.out_duration, size=size)
            items.append(_item(seg, out))
            continue

        notes.append(f"{seg.name}: 没有找到录制素材 (R{seg.round_num} {seg.player}), 已跳过")

    return items, notes
