"""DeepSeek 模型工厂 (已修复原版的三处问题).

原版问题:
    1. 环境变量名拼错: 读的是 MODEL_DEEPSKKE_FLASH (DEEPSKKE), .env 里是
       MODEL_DEEPSEEK_FLASH -> 取到 None, init_chat_model 直接 TypeError。
    2. 参数名错误: init_chat_model 的密钥参数是 `api_key`, 不是 `key`
       (传 `key` 会被 **kwargs 透传给模型类, 报 unexpected keyword)。
    3. init_chat_model 无法从 "deepseek-v4-flash" 这类名字推断 provider,
       必须显式给 model_provider="deepseek"。

另外 .env 里 `MODEL_DEEPSEEK_FLASH = deepseek-v4-flash;` 结尾多了个分号,
这里顺手 strip 掉。
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# .env 里真正的键名 (兼容旧拼写)
_ENV_KEYS = ("MODEL_DEEPSEEK_FLASH", "MODEL_DEEPSKKE_FLASH")

DEFAULT_MODEL = "deepseek-chat"


def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip().strip(";").strip().strip('"').strip("'")
    return v or None


def get_model_name() -> str:
    for k in _ENV_KEYS:
        v = _clean(os.getenv(k))
        if v:
            return v
    return DEFAULT_MODEL


def get_api_key() -> str | None:
    return _clean(os.getenv("DEEPSEEK_API_KEY"))


def get_base_url() -> str | None:
    return _clean(os.getenv("DEEPSEEK_BASE_URL"))


@lru_cache(maxsize=4)
def build_model(model: str | None = None, temperature: float = 0.3):
    """构造一个 DeepSeek ChatModel.

    需要 `langchain-deepseek` (langchain 1.x 里 DeepSeek 是独立包)。
    """
    from langchain.chat_models import init_chat_model

    name = model or get_model_name()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY, 请检查 .env")

    kwargs: dict = {"api_key": api_key}
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature

    return init_chat_model(model=name, model_provider="deepseek", **kwargs)


# 向后兼容: 原代码 `from model import model`
# 注意这会在 import 时**立即**构造模型。构造失败时这里**不抛错**, 而是把
# model 置成 None 并记下原因 —— 这样 `import model` 本身不会炸掉整个程序
# (老代码里 import 阶段就崩很难排查)。
#
# 代价是: 拿到 None 之后如果直接传给 create_deep_agent / create_agent, 只会
# 得到一条看不懂的报错或一条弃用告警, 而且会**静默**用上别家的默认模型。
# 所以新代码请用 require_model(): 缺什么它会直接说清楚。
try:
    model = build_model()
except Exception as _exc:  # pragma: no cover - 依赖外部配置
    model = None
    IMPORT_ERROR = _exc
else:
    IMPORT_ERROR = None


def require_model():
    """返回可用的模型; 构造失败时抛出带修复建议的错误.

    典型失败原因: 没装 `langchain-deepseek` (langchain 1.x 里 DeepSeek 是独立
    包), 或者 .env 里缺 DEEPSEEK_API_KEY。
    """
    if model is not None:
        return model
    raise RuntimeError(
        f"模型不可用: {type(IMPORT_ERROR).__name__ if IMPORT_ERROR else '?'}: "
        f"{IMPORT_ERROR}\n"
        f"排查建议:\n"
        f"  1) 是否装了 langchain-deepseek:  pip install langchain-deepseek\n"
        f"  2) .env 里 DEEPSEEK_API_KEY 是否存在且没有多余的分号/引号\n"
        f"  当前状态: {describe()}"
    )


def describe() -> str:
    return (
        f"model={get_model_name()} | base_url={get_base_url()} | "
        f"api_key={'已设置' if get_api_key() else '缺失'} | "
        f"built={'是' if model is not None else f'否 ({IMPORT_ERROR})'}"
    )


if __name__ == "__main__":
    print(describe())
