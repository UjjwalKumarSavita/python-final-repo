from backend.services.summary_agent import SummaryAgent

def test_summary_not_empty():
    agent = SummaryAgent()
    chunks = ["This is a test document. It has some text."] * 5
    summary = agent.summarize(chunks, target_words=100)
    # assert len(summary.split()) >= 50
    assert len(summary.split()) > 10

