"""
读 demo 的 Langgraph agent (主循环)

数据流:
  stdio_client(server_params)         # 启动 MCP server 子进程 (ReadDemoMcp.py)
   └─ ClientSession(read, write)      # MCP 客户端连上子进程的 stdio
       └─ session.initialize()         # MCP 协议握手 (告诉 server 客户端是谁)
           └─ mcp_to_langchain_tools() # 拿出 server 暴露的 tools 转成 LangChain tool
               └─ create_react_agent  # 用预构建 ReAct agent: 内部已搭好
                                      #   agent 节点 → should_continue → tools 节点 → 回 agent
                  └─ agent.ainvoke    # 给一句话, agent 自己决定调哪些 tool
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.agents import create_agent

from mcp_adapter import mcp_to_langchain_tools
from model import model

load_dotenv()


async def main(user_prompt: str) -> str:
    # ============================================================
    # MCP server 子进程参数
    # ============================================================
    # stdio_client 会 fork 一个子进程跑这个命令, 用 stdin/stdout 跟它通信
    server_params = StdioServerParameters(
        # 装了 awpy + mcp 2.x 的那个虚拟环境的 python
        command=r"E:\chat_wm\.venv\Scripts\python.exe",
        # 启动 ReadDemoMcp.py, 里面定义了 get_kill / get_prop 两个 MCP tool
        args=[r"E:\agent_cs\ReadDemoMcp.py"],
    )
    # 1. 启动 MCP server 子进程
    #    stdio_client 是 async context manager, 进入时 spawn 子进程,
    #    退出时自动 kill. 返回 (read_stream, write_stream, get_session_id)
    async with stdio_client(server_params) as (read, write):
        # 2. 建 MCP 客户端 session, 绑定到上面那对 stdio 流
        #    ClientSession 也是 async context manager, 退出时关 session
        async with ClientSession(read, write) as session:
            # 3. MCP 握手 — 必须做, 否则 list_tools 会返回空
            #    握手交换 capabilities, 告诉 server 客户端支持什么协议版本
            await session.initialize()

            # 4. 拿 server 暴露的全部 tools, 转成 LangChain StructuredTool
            tools = await mcp_to_langchain_tools(session)
            print(f"[已加载 MCP tools] {[t.name for t in tools]}")

            # 5. 用预构建 ReAct agent
            #    langgraph.prebuilt.create_react_agent 内部已经搭好:
            #      - agent 节点: 调 model, model 决定要不要用 tool
            #      - tools 节点: 执行 tool, 把结果塞回 message
            #      - 条件边 should_continue: model 输出有 tool_calls → 走 tools 节点, 否则 → END
            #    不用自己写 StateGraph / Node / Edge
            agent = create_agent(
                model=model,
                tools=tools
            )

            test_messages = [
                {
                    "role": "system",
                    "content": "你是一个处理cs2游戏数据的agent"
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]

            result = await agent.ainvoke({"messages": test_messages})

            return result
