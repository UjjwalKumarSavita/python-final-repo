from .document_store import DocumentStore
from ..core.logger import get_logger
from ..core.config import settings
from .parser import parse_file
from .chunker import chunk_text
from .vector_store import VectorStore
from .summary_agent import SummaryAgent
from .entity_extraction import extract_entities
from pathlib import Path

log = get_logger("orchestrator")

class Orchestrator:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.vstore = VectorStore(dim=384)
        self.summarizer = SummaryAgent()
        self._entities: dict[str, dict] = {}
        self.default_summary_words = settings.summary_words_default  # NEW

    def _reparse_chunks(self, saved_path: str) -> list[str]:
        ext = Path(saved_path).suffix.lower()
        text = parse_file(saved_path, ext)
        return chunk_text(text)

    def ingest_document(self, *, filename: str, saved_path: str) -> str:
        doc_id = self.store.add_document(filename=filename, path=saved_path)
        log.info("Document registered: %s", doc_id)
        try:
            chunks = self._reparse_chunks(saved_path)
            self.vstore.upsert_document(doc_id, chunks)

            summary = self.summarizer.summarize(chunks, target_words=self.default_summary_words)  # UPDATED
            self.store.set_summary(doc_id, summary)

            entities = extract_entities(summary)
            self._entities[doc_id] = entities
            log.info("Ingest complete: %s", doc_id)
        except Exception as e:
            log.exception("Processing error")
            self.store.set_error(doc_id, f"Processing error: {e}")
        return doc_id

    def generate_summary(self, *, document_id: str, target_words: int) -> dict:  # NEW
        doc = self.store.get(document_id)
        if not doc:
            return {"ok": False, "error": "not_found"}
        try:
            chunks = self._reparse_chunks(doc.path)
            # (Optional) keep vectors fresh if text changed
            self.vstore.upsert_document(document_id, chunks)
            summary = self.summarizer.summarize(chunks, target_words=target_words)
            self.store.set_summary(document_id, summary)
            self._entities[document_id] = extract_entities(summary)
            return {"ok": True}
        except Exception as e:
            log.exception("Regeneration error")
            self.store.set_error(document_id, f"Processing error: {e}")
            return {"ok": False, "error": "processing_error"}

    def get_summary(self, *, document_id: str) -> dict:
        doc = self.store.get(document_id)
        if not doc:
            return {"status": "not_found", "summary": None}
        return {"status": doc.status, "summary": doc.summary}

    def save_summary(self, *, document_id: str, summary: str) -> dict:
        doc = self.store.get(document_id)
        if not doc:
            return {"ok": False, "error": "not_found"}
        self.store.set_summary(document_id, summary)
        self._entities[document_id] = extract_entities(summary)
        return {"ok": True}

    def get_entities(self, *, document_id: str) -> dict:
        if document_id in self._entities:
            return {"status": "ready", "entities": self._entities[document_id]}
        doc = self.store.get(document_id)
        if not doc or not doc.summary:
            return {"status": "pending", "entities": None}
        ents = extract_entities(doc.summary)
        self._entities[document_id] = ents
        return {"status": "ready", "entities": ents}

    def save_entities(self, *, document_id: str, entities: dict) -> dict:
        self._entities[document_id] = entities
        return {"ok": True}

    def answer_question(self, *, question: str, document_ids: list[str] | None) -> dict:
        results = self.vstore.search(question, top_k=3)
        context_snips = [r[3][:200] for r in results]
        answer = (
            "Stub answer (Milestone 3 will use an LLM over retrieved context).\n\n"
            "Top matches:\n- " + "\n- ".join(context_snips)
        )
        return {"answer": answer, "sources": [f"{r[0]}:chunk{r[1]}" for r in results]}
