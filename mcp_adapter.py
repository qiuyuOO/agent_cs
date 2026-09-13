"""
MCP → LangChain 适配层

把 MCP session 里的 tools 转成 LangChain StructuredTool,
这样 langgraph 的 create_react_agent 就能直接用.

核心难点: MCP tool 的 inputSchema 是 JSON Schema dict,
而 LangChain 要的是 pydantic Model 类. 用 pydantic.create_model 动态生成.
"""
import asyncio

from pydantic import create_model
from langchain_core.tools import StructuredTool


# ============================================================
# JSON Schema → pydantic Model
# ============================================================

# MCP tool 的 inputSchema 一般长这样:
#   {
#     "type": "object",
#     "properties": {
#       "round_num": {"type": "integer"},
#       "name":       {"type": "string"}
#     },
#     "required": ["round_num"]
#   }
# pydantic.create_model 接收的字段格式是:
#   field_name = (type, default)
# required 字段 default 用 Ellipsis(...), 可选字段用 None.

# JSON Schema type 字符串 → Python type 映射
_TYPE_MAP = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}


def schema_to_pydantic(input_schema: dict, model_name: str = "MCPArgs"):
    """把一个 MCP tool 的 inputSchema dict 转成 pydantic Model 类.

    没有 properties 的就给个空 model (无字段), 这样 from_function 不会报错.
    """
    if not input_schema:
        # 没参数的工具也要给个空 schema, 否则 StructuredTool 会以为接收任意 dict
        return create_model(model_name)

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields = {}
    for name, spec in properties.items():
        json_type = spec.get("type", "string")
        py_type = _TYPE_MAP.get(json_type, str)  # 未知类型兜底成 str
        # required 字段: default = ... (Ellipsis) 表示必填
        # 可选字段: default = None
        default = ... if name in required else None
        fields[name] = (py_type, default)

    return create_model(model_name, **fields)


# ============================================================
# 单个 MCP tool → LangChain StructuredTool
# ============================================================

def make_langchain_tool(mcp_tool, session):
    """
    把一个 MCP Tool (有 name/description/inputSchema) 包成 LangChain StructuredTool.

    关键点:
    - session.call_tool(name, args_dict) 才是真正调 MCP server 的入口
    - 返回的是 CallToolResult, 真正内容在 .content, 一个 list[TextContent | ImageContent | ...]
    - ReAct agent 期待 tool 返回 string, 所以把 TextContent.text 用 "\n".join 拼起来
    - 闭包捕获 mcp_tool 和 session: 每个生成的 LangChain tool 都绑定到自己的 mcp_tool.name
      和同一个 session
    """
    # 动态生成 args schema, 模型名用 tool name + Args 方便调试
    # mcp 2.x 用 snake_case: input_schema (不是 inputSchema)
    args_schema = schema_to_pydantic(mcp_tool.input_schema, f"{mcp_tool.name}Args")

    # 异步实现 — langgraph 内部走 async, 这是主路径
    async def _arun(**kwargs):
        result = await session.call_tool(mcp_tool.name, kwargs)
        # result.content 是 list, 每个元素可能是 TextContent / ImageContent
        # 这里只处理 text, 其他类型跳过 (CS2 demo 项目用不到图片返回)
        texts = []
        for c in result.content:
            # TextContent 有 .text 属性, 其他类型 (Image) 没有 → 用 getattr 兜底
            text = getattr(c, "text", None)
            if text is not None:
                texts.append(text)
        return "\n".join(texts) if texts else ""

    # 同步实现 — 兼容 sync 调用, 用 asyncio.run 跑一遍 _arun
    def _run(**kwargs):
        return asyncio.run(_arun(**kwargs))

    return StructuredTool.from_function(
        coroutine=_arun,                # async 路径
        func=_run,                       # sync 路径
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=args_schema,
    )


# ============================================================
# 批量转换: 一次拿全部 MCP tools
# ============================================================

async def mcp_to_langchain_tools(session):
    """从已初始化的 MCP ClientSession 拿出全部 tools, 转成 LangChain tool 列表.

    注意 session 必须已经 await session.initialize() 过,
    否则 list_tools 会返回空.
    """
    tools_resp = await session.list_tools()
    # tools_resp.tools 是 list[mcp.Tool]
    return [make_langchain_tool(t, session) for t in tools_resp.tools]
