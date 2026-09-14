"""deep agent 主循环: 一个编排 agent + 一个"读 demo"子 agent.

架构 (deepagents 0.7.x):
    主 agent (create_deep_agent)
      ├─ 自带工具: ls / read_file / write_file / edit_file / glob / grep / task
      └─ task() 派活 → 子 agent "demo-analyst"
                         └─ MCP tool: get_kill / get_prop (ReadDemoMcp.py)

原版为什么崩:
    `subagents=[ReadDemoAgent]` 传的是一个 **Python 模块**。deepagents 期望的是
    声明式 spec (TypedDict), 它在里面找 `"graph_id" in spec` 就抛了
        TypeError: argument of type 'module' is not iterable
    SubAgent 的必填键只有两个: `name` 与 `description`, 其余 (tools / model /
    system_prompt) 可选。

另外两个一并修掉的坑:
    1. `from model import model` 在 langchain-deepseek 缺失时**静默**退化成
       `model=None`, 于是 create_deep_agent 收到 None → 那条弃用告警 +
       默认模型跑偏。现在用 `require_model()`, 缺什么直接说清楚。
    2. 原版 while True: continue 是空转死循环 (什么都不做)。现在是个真 REPL。

用法:
    E:\\chat_wm\\.venv\\Scripts\\python.exe main.py                    # 交互
    E:\\chat_wm\\.venv\\Scripts\\python.exe main.py "分析第 8 回合"     # 单次
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepagents import create_deep_agent  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from demo_mcp import mcp_tool_session  # noqa: E402
from model import IMPORT_ERROR, describe, require_model  # noqa: E402

load_dotenv()

CHECKPOINT_DB = Path(__file__).parent / "data" / "checkpoints.db"

MAIN_PROMPT = """你是一个 CS2 demo 剪辑编排助手。

你有一个子 agent 叫 demo-analyst, 它能用 MCP 工具读取一场 demo 的击杀与道具数据
(get_kill / get_prop)。

你的职责:
  * 需要比赛事实时, 用 task() 把**明确的**问题交给 demo-analyst, 例如
    "用 get_kill 读 <路径>, 找出第 8 回合全部击杀, 按 tick 排序"
  * 拿到事实后做判断与编排 (哪些片段值得剪、怎么排序、给什么理由)
  * 不要自己编造 demo 里没有的数据; 子 agent 拿不到就如实说拿不到

demo 路径示例:
  D:\\5E_cs2_demo\\g161-n-20260912132632481838433_de_dust2\\g161-n-20260912132632481838433_de_dust2.dem
"""

SUBAGENT_NAME = "demo-analyst"


def build_subagent(tools):
    """构造声明式 subagent spec (SubAgent TypedDict).

    必填只有 name / description; tools 必须在这里显式给, 否则子 agent 只会
    继承主 agent 的默认工具 (文件操作), **拿不到 MCP tool** —— 那正是原版
    想做的事却做不成的地方。
    """
    return {
        "name": SUBAGENT_NAME,
        "description": (
            "CS2 demo 数据读取专家。用 MCP 工具 get_kill / get_prop 从 .dem 文件里"
            "取击杀记录与道具使用情况。需要比赛事实 (谁杀了谁、第几回合、什么武器、"
            "残局人数、道具覆盖) 时调用它。它只负责取数与汇总, 不做剪辑决策。"
        ),
        "system_prompt": (
            "你是 CS2 demo 数据分析师。\n"
            "可用工具:\n"
            "  * get_kill(demo_path): 全部击杀 (attacker/victim/weapon/headshot/"
            "tick/round/distance/attacker_place ...)\n"
            "  * get_prop(demo_path): 每回合道具 (throws/damages/flash_groups)\n"
            "规则:\n"
            "  * demo_path 一律用**绝对路径**, Windows 路径要带转义的反斜杠\n"
            "  * 回答要给出具体数字与 tick, 不要泛泛而谈\n"
            "  * 数据里没有的字段就说不确定, 不要猜"
        ),
        "tools": tools,
        "model": require_model(),
    }


async def run_once(agent, prompt: str, thread_id: str) -> str:
    """跑一条用户消息, 打印最终回答并返回文本."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages") or []
    final = messages[-1] if messages else None
    content = getattr(final, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


async def main_loop() -> None:
    model = require_model()
    print("=" * 72)
    print("CS2 demo 剪辑编排 agent")
    print(f"  {describe()}")
    print(f"  子 agent: {SUBAGENT_NAME} (MCP: get_kill / get_prop)")
    print(f"  检查点库: {CHECKPOINT_DB}")
    print("=" * 72)

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    # 检查点库必须**一直开着**到整个会话结束。
    # 反例: `cm = AsyncSqliteSaver.from_conn_string(...); await cm.__aenter__()`
    # 只是进了上下文却没把 cm 本身留在栈上 —— 一离开那个函数的栈帧, 上下文
    # 管理器就可能被回收并触发 __aexit__, 连接被关掉, 之后第一次 ainvoke
    # 报 `ValueError: Connection closed`。用 async with 把生命周期写死。
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        # MCP 子进程同理, 必须整个会话期间存活 —— tools 绑定在那个 session 上,
        # 退出 with 就等于 kill 掉子进程, 之后调用全失败。
        async with mcp_tool_session() as tools:
            names = [t.name for t in tools]
            print(f"[MCP] 已加载工具 {names}")
            if not names:
                raise RuntimeError("MCP server 没有暴露任何工具, 检查 ReadDemoMcp.py")

            agent = create_deep_agent(
                model=model,
                tools=[],                       # 主 agent 不需要直接持有 MCP tool
                checkpointer=checkpointer,
                subagents=[build_subagent(tools)],
                system_prompt=MAIN_PROMPT,
            )

            prompt = " ".join(sys.argv[1:]).strip()
            if prompt:
                answer = await run_once(agent, prompt, "cli")
                print("\n=== 最终回答 ===")
                print(answer)
                return

            print("\n输入问题 (空行 / exit 退出):\n")
            turn = 0
            while True:
                try:
                    line = input("你 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line or line.lower() in ("exit", "quit", ":q"):
                    break
                turn += 1
                try:
                    answer = await run_once(agent, line, f"cli-{turn}")
                except Exception as exc:        # 一轮失败不该结束整个会话
                    print(f"\n[出错] {type(exc).__name__}: {exc}\n")
                    continue
                print("\n=== 最终回答 ===")
                print(answer)
                print()


if __name__ == "__main__":
    if IMPORT_ERROR is not None:
        print(f"[致命] 模型构造失败: {IMPORT_ERROR}")
        print("      检查 .env 里的 DEEPSEEK_API_KEY, 以及是否装了 langchain-deepseek")
        raise SystemExit(2)
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n已退出")
