from backend.services.vector_store import VectorStore
from backend.services.qa_agent import QAAgent

def test_qa_fallback():
    vs = VectorStore(dim=10)
    vs.upsert_document("doc1", ["This is about Python programming."])
    qa = QAAgent(vs)
    result = qa.answer(question="What is Python?")
    assert "Python" in result["answer"]
