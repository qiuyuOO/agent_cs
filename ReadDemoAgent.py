"""读 demo 的 LangGraph agent —— 单轮 ReAct 版本 (MCP tool + create_agent).

数据流:
  stdio_client(server_params)         # 启动 MCP server 子进程 (ReadDemoMcp.py)
   └─ ClientSession(read, write)      # MCP 客户端连上子进程的 stdio
       └─ session.initialize()         # MCP 协议握手 (告诉 server 客户端是谁)
           └─ mcp_to_langchain_tools() # 拿出 server 暴露的 tools 转成 LangChain tool
               └─ create_agent         # 预构建 agent: agent 节点 → tools 节点 → 回 agent
                  └─ agent.ainvoke     # 给一句话, agent 自己决定调哪些 tool

这条链路本身是好的, 问题只在于 `main.py` 曾经把**整个模块**当成 subagent 交给
deepagents (deepagents 要的是声明式 spec dict)。所以这里把手握逻辑抽到
`demo_mcp.mcp_tool_session()`, 让两边都能安全复用:

  * 想跑单 agent:  `await main("...")`            (本文件)
  * 想跑 deep agent: `async with mcp_tool_session() as tools: ...` (main.py)
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from demo_mcp import mcp_tool_session
from model import build_model

load_dotenv()

SYSTEM_PROMPT = "你是一个处理 CS2 游戏数据的 agent"


async def main(user_prompt: str):
    """单轮问答: 建一个带 MCP tool 的 ReAct agent, 跑一条用户消息, 返回结果."""
    from langchain.agents import create_agent

    async with mcp_tool_session() as tools:
        print(f"[已加载 MCP tools] {[t.name for t in tools]}")
        agent = create_agent(model=build_model(), tools=tools)
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return await agent.ainvoke({"messages": test_messages})


def _print_result(result) -> None:
    messages = (result or {}).get("messages") or []
    if not messages:
        print("(没有返回消息)")
        return
    final = messages[-1]
    content = getattr(final, "content", final)
    if isinstance(content, list):
        content = "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    print("\n=== 最终回答 ===")
    print(content)


if __name__ == "__main__":  # pragma: no cover - 手工验证入口
    import sys

    prompt = sys.argv[1] if len(sys.argv) > 1 else "这场 demo 一共多少回合?"
    _print_result(asyncio.run(main(prompt)))
