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
    """从 changelog.xml 里读出版本号 (读不到就返回空)."""
    for name in ("changelog.xml", "x64/AfxHookSource2_changelog.xml"):
        p = hlae_dir() / name
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        m = re.search(r'version="([^"]+)"', text)
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


def procs_running() -> dict[str, bool]:
    """CS2 / HLAE / Steam 是否在运行 (Windows, 用 tasklist)."""
    out = {"cs2": False, "hlae": False, "steam": False}
    try:
        r = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=20)
        text = (r.stdout or "").lower()
    except (OSError, subprocess.SubprocessError):
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

    if check_running:
        run = procs_running()
        pf.info["running"] = run
        if not run["steam"]:
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
        }


def plan_from_edl(edl, highlights: Sequence[Any]) -> list[RecSegment]:
    """把 EDL 翻译成录制计划.

    注意 EDL 的 `src_start_tick/src_end_tick` 是 **demo tick**, 而 `out_start/
    out_end` 是成片时间轴。录制的取景窗口应当用前者, 时长用后者 —— 两者之比
    就是 ffmpeg 需要做的变速比。
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
    `mirv_streams record name/fps/start/end`、`mirv_skip time to`、
    `demo_pause/demo_resume`、`playdemo`、`spec_player`、`bind`。
    """
    segs = list(plan[start_index:])
    name_prefix = Path(output_dir).name or "hlae_record"

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
    bootstrap.append(f'mirv_streams record fps {int(fps)}')
    bootstrap.append("bind F8 \"exec cs2clipper_stop.cfg\"")
    bootstrap.append("")
    bootstrap.append("// 先把 demo 放起来 (路径里的反斜杠写成双反斜杠)")
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
) -> dict[str, Any]:
    """生成并写出整套录制脚本 (引导 + 停录 + 每段一个).

    返回一份清单, 直接给 CLI/Web 展示"接下来要 exec 哪些文件"。
    """
    out_dir = Path(output_dir) if output_dir else record_output_dir()
    bootstrap, stop, clips = build_cs2_config(
        plan, demo_path=demo_path, output_dir=str(out_dir), fps=fps,
    )
    written: list[str] = []
    written.append(str(write_cs2_config(bootstrap, filename="cs2clipper_bootstrap.cfg",
                                        target_dir=target_dir)))
    written.append(str(write_cs2_config(stop, filename="cs2clipper_stop.cfg",
                                        target_dir=target_dir)))
    for name, text in clips:
        written.append(str(write_cs2_config(text, filename=name, target_dir=target_dir)))
    return {
        "cfg_dir": str(target_dir or cs2_cfg_dir()),
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


def discover_recordings(output_dir: str | Path) -> dict[str, Any]:
    """看录制目录里有什么 —— 用来判断"录成功了没".

    HLAE 的 mirv_streams 会按 stream 名建子目录, 里面是**帧序列** (默认 tga/bmp)
    或 ffmpeg 编码后的视频。两种都要能认出来。
    """
    d = Path(output_dir)
    res: dict[str, Any] = {"dir": str(d), "exists": d.is_dir(),
                           "frame_dirs": [], "videos": [], "total_frames": 0}
    if not d.is_dir():
        return res
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            frames = [f for f in sub.iterdir() if f.suffix.lower() in IMAGE_EXT]
            if frames:
                res["frame_dirs"].append({
                    "name": sub.name, "frames": len(frames),
                    "first": sorted(f.name for f in frames)[0],
                    "ext": frames[0].suffix.lower(),
                })
                res["total_frames"] += len(frames)
        elif sub.suffix.lower() in VIDEO_EXT:
            res["videos"].append({"name": sub.name,
                                  "bytes": sub.stat().st_size})
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
    `setpts=PTS * (录制时长 / 目标时长)` 整体重采样 —— 这就是慢放/快放,
    且**时长严格等于 EDL 要求**, 与 compose.finalize_clip 的契约一致。

    用 ffmpeg 的 image2 demuxer 读帧序列, 不引入 imageio/PIL 逐帧读取
    (一秒 60 帧、一段 4 秒就是 240 张 1080p 图, 走 Python 循环会明显变慢)。
    """
    frame_dir, out_path = Path(frame_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if not frames:
        raise FileNotFoundError(f"目录里没有帧序列: {frame_dir}")

    ext = frames[0].suffix.lower()
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

    # 再规整到精确时长 (变速 + 必要时代码补帧)
    return compose.finalize_clip(
        raw, out_path, target_duration=target_duration, size=(W, H),
        crf=crf, preset=preset,
    )


def _ffmpeg(cmd: list[str], *, desc: str = "") -> None:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise RuntimeError(f"ffmpeg 失败 ({desc}):\n{tail}")


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
    by_name = {d["name"]: d for d in found["frame_dirs"]}
    videos = {v["name"]: v for v in found["videos"]}

    items: list[compose.ConcatItem] = []
    notes: list[str] = []
    for seg in plan:
        # 约定: stream 名就是 clip_001 这种; 子目录名可能带后缀
        hit = next((k for k in by_name if k == seg.name or k.startswith(seg.name)), None)
        if hit:
            out = clips_dir / f"clip_{seg.index + 1:03d}.mp4"
            frames_to_clip(record_dir / hit, out, fps=fps,
                           target_duration=seg.out_duration, size=size)
            items.append(compose.ConcatItem(path=out, duration=seg.out_duration))
            continue

        vhit = next((k for k in videos if k == seg.name or k.startswith(seg.name)), None)
        if vhit:
            out = clips_dir / f"clip_{seg.index + 1:03d}.mp4"
            src = record_dir / vhit
            actual = compose.probe_duration(src)
            notes.append(f"{seg.name}: 用录好的视频 {vhit} ({actual:.2f}s), "
                         f"目标 {seg.out_duration:.2f}s")
            compose.finalize_clip(src, out, target_duration=seg.out_duration, size=size)
            items.append(compose.ConcatItem(path=out, duration=seg.out_duration))
            continue

        notes.append(f"{seg.name}: 没有找到录制素材 (R{seg.round_num} {seg.player}), 已跳过")

    return items, notes
