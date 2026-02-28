import asyncio
import os
from engine.db import db_manager
from engine.repository import repository
from engine.contracts import AnalysisRequest, DatasetMetadata
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()

async def add_fake_analysis():
    db_manager.connect_db()
    
    analysis_id = str(uuid.uuid4())
    req = AnalysisRequest(
        query="Test query",
        dataset_metadata=DatasetMetadata(filename="test.csv", total_rows=10, columns=["A"], dtypes={"A":"int"}),
        sample_data="A\n1"
    )
    
    await repository.create_analysis(analysis_id, req)
    print(f"Created fake analysis {analysis_id}")
    
    # Now try to get history
    history = await repository.get_history()
    print(f"History fetched: {len(history)} items")
    print(history)
    
    db_manager.close_db()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(add_fake_analysis())
