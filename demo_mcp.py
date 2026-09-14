"""MCP 客户端封装 —— 把 ReadDemoMcp.py 暴露的 tool 取成 LangChain tool.

单独抽成模块的原因: 这些 tool 必须**在 async 上下文里**才能拿到 (要先用管道
拉起 MCP server 子进程、做协议握手), 而 `main.py` 需要在建 agent **之前**就
拿到它们传给 deepagents 的 subagent。原来这段逻辑埋在 ReadDemoAgent.main()
里, 于是 main.py 只能把整个模块塞进 `subagents=` —— 那是错的 (deepagents 要的
是声明式的 spec dict)。

用法:
    async with mcp_tool_session() as tools:      # events 里就是这里的 tools
        agent = create_deep_agent(..., subagents=[{... "tools": tools}])
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_adapter import mcp_to_langchain_tools

#: 装了 awpy + mcp 2.x 的那个虚拟环境的 python; 用环境变量可覆盖
DEFAULT_PYTHON = r"E:\chat_wm\.venv\Scripts\python.exe"
DEFAULT_SERVER = r"E:\agent_cs\ReadDemoMcp.py"


@asynccontextmanager
async def mcp_tool_session(
    *,
    python: str | None = None,
    server: str | None = None,
):
    """拉起 MCP server 子进程 → 握手 → 产出 LangChain tool 列表.

    退出时自动关 session 并 kill 子进程 —— 所以**调用方必须在使用期间**
    持有这个上下文, 不能 "拿到 tools 就退出 with"。
    """
    import os

    params = StdioServerParameters(
        command=python or os.environ.get("DEMO_MCP_PYTHON") or DEFAULT_PYTHON,
        args=[server or os.environ.get("DEMO_MCP_SERVER") or DEFAULT_SERVER],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()          # 不握手 list_tools 会是空的
            tools = await mcp_to_langchain_tools(session)
            yield tools
