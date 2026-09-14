"""轻量 AST 静态检查 (不引第三方依赖).

检查项:
  * 未使用的导入
  * 定义但从未被引用的私有函数/方法 (_xxx)
  * 函数里定义但未使用的局部变量
  * 过长的函数 (可维护性信号)
  * 同一模块内重复定义的函数名 (后者静默覆盖前者)
  * 可疑的 except 吞异常 (except Exception: pass)
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\agent_cs\cs2clipper")

# 报告容器
unused_imports: list[tuple[str, int, str]] = []
unused_private: list[tuple[str, int, str]] = []
unused_locals: list[tuple[str, int, str, str]] = []
long_funcs: list[tuple[str, str, int]] = []
dup_defs: list[tuple[str, str, int, int]] = []
swallowed: list[tuple[str, int, str]] = []


def _names_used(tree: ast.AST) -> Counter:
    """收集所有被读取的标识符名 (Load 上下文)."""
    used: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used[node.id] += 1
        elif isinstance(node, ast.Attribute):
            # obj.attr -> obj 已被 Name 覆盖; attr 不算模块级名字
            pass
    return used


def _docstring_strings(tree: ast.AST) -> set[int]:
    """标注出文档字符串节点 (它们不该被当作"表达式语句"误报)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0]))
    return out


for path in sorted(ROOT.glob("*.py")):
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[语法错误] {path.name}: {e}")
        continue

    mod = path.name
    used = _names_used(tree)
    # 名字在字符串注解/__all__/装饰器里也可能用到, 宽松处理
    text_names = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for tok in node.value.replace("|", " ").replace("[", " ").replace("]", " ").split():
                text_names[tok.strip(",.()'\"")] += 1

    # ---- 未使用的导入 ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                name = (a.asname or a.name).split(".")[0]
                if used[name] == 0 and text_names[name] == 0:
                    unused_imports.append((mod, node.lineno, name))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                name = a.asname or a.name
                if used[name] == 0 and text_names[name] == 0:
                    unused_imports.append((mod, node.lineno, name))

    # ---- 重复定义 ----
    seen: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                dup_defs.append((mod, node.name, seen[node.name], node.lineno))
            seen[node.name] = node.lineno

    # ---- 私有函数/方法未被引用 ----
    all_src = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob("*.py"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nm = node.name
            if not nm.startswith("_") or nm.startswith("__"):
                continue
            # 在全部源码里数引用次数 (减去定义本身那次)
            refs = all_src.count(nm)
            if refs <= 1:
                unused_private.append((mod, node.lineno, nm))
            n_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if n_lines > 120:
                long_funcs.append((mod, nm, n_lines))

    # ---- 未使用的局部变量 ----
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: dict[str, int] = {}
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        assigned.setdefault(t.id, sub.lineno)
        loads = Counter()
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                loads[sub.id] += 1
        for name, ln in assigned.items():
            if name.startswith("_"):
                continue                      # 下划线开头的常是占位
            if loads[name] == 0:
                unused_locals.append((mod, ln, fn.name, name))

    # ---- 吞异常 ----
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                nm = ast.unparse(node.type) if node.type else "bare"
                swallowed.append((mod, node.lineno, nm))

print("=" * 78)
print(f"未使用的导入 ({len(unused_imports)})")
print("=" * 78)
for mod, ln, nm in unused_imports:
    print(f"  {mod}:{ln}  {nm}")

print("\n" + "=" * 78)
print(f"疑似从未被引用的私有函数 ({len(unused_private)})")
print("=" * 78)
for mod, ln, nm in unused_private:
    print(f"  {mod}:{ln}  {nm}")

print("\n" + "=" * 78)
print(f"重复定义 (后者覆盖前者) ({len(dup_defs)})")
print("=" * 78)
for mod, nm, l1, l2 in dup_defs:
    print(f"  {mod}: {nm}  L{l1} 与 L{l2}")

print("\n" + "=" * 78)
print(f"未使用的局部变量 ({len(unused_locals)})")
print("=" * 78)
for mod, ln, fn, nm in unused_locals:
    print(f"  {mod}:{ln}  在 {fn}() 里  {nm}")

print("\n" + "=" * 78)
print(f"静默吞异常 except: pass ({len(swallowed)})")
print("=" * 78)
for mod, ln, nm in swallowed:
    print(f"  {mod}:{ln}  except {nm}: pass")

print("\n" + "=" * 78)
print(f"超长函数 > 120 行 ({len(long_funcs)})")
print("=" * 78)
for mod, nm, n in sorted(long_funcs, key=lambda x: -x[2]):
    print(f"  {mod}  {nm}()  {n} 行")
