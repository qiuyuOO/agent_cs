"""前端静态一致性检查 (无浏览器时的替代验证).

没有 headless 浏览器可用, 所以用静态方式抓三类最常见的"前端一打开就废"的错误:

1. JS 里 `$('#foo')` / `$('.foo')` 选择的 id/class 在 index.html 里根本不存在
   (拼错、改名后漏改) —— 这类错误在浏览器里表现为 null.textContent 抛错,
   整个界面卡死, 但后端日志里一个字都没有。
2. HTML 与 JS 的 id 重复 / 缺失
3. JS 语法错误 (用 node 检查; 没有 node 就退化为括号配平粗检)

用法: python tools/dev/check_frontend.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "cs2clipper" / "web"
HTML = WEB / "index.html"
JS = WEB / "app.js"
CSS = WEB / "app.css"

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    html_ids = set(re.findall(r'\bid="([^"]+)"', html))
    html_classes: set[str] = set()
    for attr in re.findall(r'\bclass="([^"]+)"', html):
        html_classes.update(attr.split())
    # JS 里动态插入的元素也算"存在": class="..."、className = '...':
    # classList.add('...')、以及模板字符串里拼接的 class
    for attr in re.findall(r'class="([^"$]+)"', js):
        html_classes.update(attr.split())
    for attr in re.findall(r"""className\s*=\s*['"]([^'"]+)['"]""", js):
        html_classes.update(attr.split())
    for attr in re.findall(r"""classList\.(?:add|toggle|remove)\(\s*['"]([^'"]+)['"]""", js):
        html_classes.update(attr.split())
    for attr in re.findall(r"""class="([^"]*?)"',\s*'([^']+)'""", js):
        html_classes.update((attr[0] + " " + attr[1]).split())

    js_ids = set(re.findall(r"""\$\(\s*['"]#([A-Za-z0-9_\-]+)['"]""", js))
    js_classes = set(re.findall(r"""\$\(\s*['"]\.([A-Za-z0-9_\-]+)['"]""", js))
    # querySelectorAll 里的复合选择器
    for sel in re.findall(r"""querySelectorAll\(\s*['"]([^'"]+)['"]""", js):
        js_ids.update(re.findall(r"#([A-Za-z0-9_\-]+)", sel))
        js_classes.update(re.findall(r"\.([A-Za-z0-9_\-]+)", sel))

    missing_ids = sorted(js_ids - html_ids)
    check("JS 用到的 id 都存在于 HTML", not missing_ids, str(missing_ids))

    missing_cls = sorted(js_classes - html_classes)
    check("JS 用到的 class 都存在于 HTML 或由 JS 动态创建", not missing_cls, str(missing_cls))

    dup_ids = [i for i in html_ids if html.count(f'id="{i}"') > 1]
    check("HTML 没有重复 id", not dup_ids, str(dup_ids))

    # JS 里写死的接口路径必须在后端真实存在
    api_paths = set(re.findall(r"""['"](/api/[A-Za-z0-9_\-/]+)""", js))
    # 形如 '/api/jobs/' + id 的拼接: 去掉结尾的 / 再做前缀匹配
    api_paths = {p.rstrip("/") or p for p in api_paths}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cs2clipper import webapp  # noqa: E402

    app = webapp.create_app()
    routes = {getattr(r, "path", "") for r in app.routes}
    unknown = []
    for p in api_paths:
        if p in routes:
            continue
        if any(r.startswith(p + "/") or r.startswith(p) for r in routes if r.startswith("/api/")):
            continue
        if any(re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", r), p)
               for r in routes if r.startswith("/api/")):
            continue
        unknown.append(p)
    check("JS 调用的 /api 路径后端都存在", not unknown, str(sorted(unknown)))

    # 后端路由与 JS 调用之间的反向检查: 明显没被前端用到的端点列出来 (仅提示)
    used_prefixes = {p.rsplit("/", 1)[0] for p in api_paths}
    unused = sorted(r for r in routes
                    if r.startswith("/api/") and "{" not in r
                    and r not in api_paths and r.rsplit("/", 1)[0] not in used_prefixes)
    print(f"  [INFO] 后端有但前端没直接调用的端点: {unused}")

    # JS 语法
    node = shutil.which("node")
    if node:
        r = subprocess.run([node, "--check", str(JS)], capture_output=True, text=True)
        check("app.js 语法 (node --check)", r.returncode == 0, r.stderr.strip()[:200])
    else:
        print("  [INFO] 没有 node, 跳过语法检查 (退化为括号配平)")
        for open_c, close_c in (("{", "}"), ("(", ")"), ("[", "]")):
            check(f"app.js {open_c}{close_c} 配平",
                  js.count(open_c) == js.count(close_c),
                  f"{js.count(open_c)} vs {js.count(close_c)}")

    # CSS 里定义的类是否覆盖 JS 动态插入的 class (纯提示, 不算失败)
    css_classes = set(re.findall(r"\.([A-Za-z][A-Za-z0-9_\-]*)", css))
    unstyled = sorted(c for c in html_classes
                      if c not in css_classes and c not in ("active", "hidden"))
    print(f"  [INFO] HTML 有但 CSS 没定义的 class: {unstyled[:12]}")

    print()
    print("=" * 56)
    print(f"前端静态检查: {'全部通过' if not FAIL else '失败 ' + ', '.join(FAIL)}")
    print("=" * 56)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
