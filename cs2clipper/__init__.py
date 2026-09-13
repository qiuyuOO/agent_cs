"""CS2 demo + 音乐自动剪辑 Agent.

模块划分:
    config.py        —— 路径常量 + 环境引导 (必须在 import 其他重库前导入)
    music.py         —— Stage 1: 音乐分析 (BPM/节拍/能量/频谱 → 情绪分段)
    demo.py          —— Stage 2: demo 解析 → 亮点卡片 (HighlightCard)
    planner.py       —— Stage 3: LLM 编排 (音乐段落 × 亮点 → EDL)
    radar.py         —— Stage 4: 雷达帧渲染 (逐帧 PNG)
    compose.py       —— Stage 5: ffmpeg 合成成片
    pipeline.py      —— LangGraph 串起 Stage 1-5
"""

__version__ = "0.1.0"
