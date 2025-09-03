"""
Minimal orchestrator. Later we'll plug in AutoGen agents and pgvector.
Function signatures are designed to stay stable.
"""
from .document_store import DocumentStore
from ..core.logger import get_logger

log = get_logger("orchestrator")

class Orchestrator:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def ingest_document(self, *, filename: str, saved_path: str) -> str:
        """
        Register an uploaded doc. Future: parse + embed + store in pgvector.
        """
        doc_id = self.store.add_document(filename=filename, path=saved_path)
        log.info("Document registered: %s", doc_id)
        # Milestone 2: trigger parse/summarize pipeline async and update summary/status
        return doc_id

    def get_summary(self, *, document_id: str) -> dict:
        """
        Return summary/status if available.
        """
        doc = self.store.get(document_id)
        if not doc:
            return {"status": "not_found", "summary": None}
        return {"status": doc.status, "summary": doc.summary}

    def answer_question(self, *, question: str, document_ids: list[str] | None) -> dict:
        """
        Stub Q&A. Later: retrieve from pgvector, run Q&A agent.
        """
        log.info("Q&A requested: %s on docs=%s", question, document_ids)
        return {"answer": "This is a placeholder answer. Pipeline to be added in Milestone 3.", "sources": document_ids or []}
