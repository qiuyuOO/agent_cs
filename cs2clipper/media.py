"""Web 界面用的文件发现与路径安全.

两条路径策略 (刻意不同, 因为用途不同):

* **素材 (音乐 / demo)** —— 用户可以选工作区之外的任何文件 (音乐库通常在别的盘)。
  这是本地工具, 所以允许任意路径, 但只接受**确有此扩展名的常规文件**, 且拒绝
  目录遍历以外的危险形状 (非绝对路径、含空字节等)。解析失败也不能抛 500。

* **产物 (视频 / 缩略图 / JSON)** —— 只允许工作区内的文件, 由
  `safe_under()` 做 realpath 前缀校验, 防止 `../../.env` 之类被下载走。

另外提供 `list_music_files()`, 把"选择音乐"做成可浏览的列表而不是让用户手打路径。
"""
from __future__ import annotations

from pathlib import Path

from . import config

MUSIC_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
DEMO_EXT = {".dem", ".dem.gz", ".bz2"}

# demo 常被解压成"同名文件夹里放同名 .dem", 搜索时要往下钻一层
_MAX_SCAN_DEPTH = 3


class PathNotAllowed(PermissionError):
    """路径不在允许范围内 (Web 端专用异常, 与沙箱无关)."""


def safe_under(path: str | Path, root: Path) -> Path:
    """把 path 解析成 root 下的真实文件路径, 否则抛 PathNotAllowed.

    用 `resolve()` 之后再比较前缀 —— 直接比较字符串会被 `..` 和符号链接绕过。
    """
    root = root.resolve()
    try:
        p = Path(path).resolve()
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"路径无法解析: {path}") from exc
    if p != root and root not in p.parents:
        raise PathNotAllowed(f"路径不在允许目录内: {p}")
    return p


def is_media_file(path: str | Path, exts: set[str]) -> bool:
    """存在、是常规文件、且扩展名在允许集合内."""
    try:
        p = Path(path)
    except (OSError, ValueError, TypeError):
        return False
    if not p.is_file():
        return False
    # .dem.gz 这类双扩展名要整体比较
    name = p.name.lower()
    if name.endswith(".dem.gz"):
        return ".dem.gz" in exts
    return p.suffix.lower() in exts


def resolve_media(path: str, exts: set[str], *, what: str) -> Path:
    """校验用户提交的素材路径; 不合法时给出可读的中文原因."""
    raw = (path or "").strip().strip('"')
    if not raw:
        raise ValueError(f"未提供{what}路径")
    if "\x00" in raw:
        raise ValueError(f"{what}路径包含非法字符")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        # 相对路径按工程根解析 —— Web 端传进来的都是选中项, 不该出现相对路径,
        # 但真出现时给出确定的解释比报错更友好
        p = (config.ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{what}不存在: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{what}是目录而不是文件: {p} (请选择里面的 .dem 文件)")
    if not is_media_file(p, exts):
        allowed = "/".join(sorted(exts))
        raise ValueError(f"{what}扩展名不支持 ({p.suffix or '无扩展名'}), 支持: {allowed}")
    return p.resolve()


def list_music_files(root: str | Path | None = None, limit: int = 400
                     ) -> tuple[list[dict], list[dict], str]:
    """列出一个目录下的音乐文件与子目录 (不递归, 按名称排序).

    返回 `(音乐文件, 子目录, 实际使用的目录)`。目录不可读时返回空列表而不抛错
    —— 用户可能选到一个没有权限的盘或光驱。
    """
    base = Path(root).expanduser() if root else Path.home()
    if not base.is_absolute():
        raise ValueError("目录必须是绝对路径")
    out: list[dict] = []
    if not base.is_dir():
        return out
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError):
        return out
    dirs = []
    for p in entries:
        try:
            if p.is_dir():
                if p.name.startswith("$") or p.name.startswith("."):
                    continue
                dirs.append({"name": p.name, "path": str(p)})
                continue
        except OSError:
            continue
        if is_media_file(p, MUSIC_EXT):
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            out.append({"name": p.name, "path": str(p), "size": size})
        if len(out) >= limit:
            break
    return out, dirs[:200], str(base)  # type: ignore[return-value]


def list_demos(roots: list[Path] | None = None, limit: int = 200) -> list[dict]:
    """在候选目录里找 .dem (含"同名文件夹"这种解压结构)."""
    found: list[dict] = []
    seen: set[str] = set()
    bases = roots or [config.DEFAULT_DEMO.parent, config.ROOT]
    for base in bases:
        base = Path(base)
        if not base.is_dir():
            continue
        stack = [(base, 0)]
        while stack and len(found) < limit:
            d, depth = stack.pop()
            try:
                entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
            except (PermissionError, OSError):
                continue
            for p in entries:
                if p.is_dir():
                    if depth + 1 <= _MAX_SCAN_DEPTH and p.name != ".git":
                        stack.append((p, depth + 1))
                    continue
                if not is_media_file(p, DEMO_EXT):
                    continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                found.append({"name": p.name, "path": key, "size": size,
                              "dir": str(p.parent)})
        if len(found) >= limit:
            break
    found.sort(key=lambda x: x["name"].lower())
    return found


def default_music_dir() -> str:
    """音乐选择器的默认起始目录 (上次用过 > 家目录)."""
    from . import store

    try:
        last = str(store.load_prefs().get("last_music_dir", "") or "").strip()
    except Exception:
        last = ""
    if last and Path(last).is_dir():
        return last
    return str(Path.home())
