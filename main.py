from dotenv import load_dotenv
from langchain.tools import tool
import asyncio
import ReadDemoAgent
from deepagents import create_deep_agent
from model import model
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

checkpoint_db_path = Path(__file__).parents[0] / "data" / "checkpoints.db"
checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

async def main_loop():
    _checkpoint_cm = AsyncSqliteSaver.from_conn_string(str(checkpoint_db_path))
    checkpointer = await _checkpoint_cm.__aenter__()

    agent = create_deep_agent(
        model =model ,
        tools = [],
        checkpointer = checkpointer,
        subagents  = [
            ReadDemoAgent
        ]
    )
    while True :
        continue

if __name__ == '__main__':
    asyncio.run(main_loop())

