import time
from backend.services.summary_agent import SummaryAgent

def test_summary_perf():
    agent = SummaryAgent()
    chunks = ["This is a test sentence."] * 500
    start = time.time()
    summary = agent.summarize(chunks, target_words=200)
    elapsed = time.time() - start
    assert elapsed < 5  # must be fast
    assert len(summary.split()) > 50
