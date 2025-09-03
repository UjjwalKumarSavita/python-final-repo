from pathlib import Path
from .document_store import DocumentStore
from ..core.logger import get_logger
from ..core.config import settings
from .parser import parse_file
from .chunker import chunk_text
from .vector_store import VectorStore
from .summary_agent import SummaryAgent
from .entity_extraction import extract_entities
from .qa_agent import QAAgent
from .validator_agent import validate_summary, validate_answer
import time
from .summary_agent import SummaryAgent
from .entity_extraction import extract_entities

log = get_logger("orchestrator")

class Orchestrator:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.vstore = VectorStore(dim=384)
        self.summarizer = SummaryAgent()
        self.qa = QAAgent(self.vstore)
        self._entities: dict[str, dict] = {}
        self.default_summary_words = settings.summary_words_default if hasattr(settings, "summary_words_default") else 350

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

            summary = self.summarizer.summarize(chunks, target_words=self.default_summary_words)
            # Validate and save as a version
            val = validate_summary(summary, min_words=int(self.default_summary_words*0.6), max_words=int(self.default_summary_words*1.6))
            self.store.push_summary_version(doc_id, summary, note="ingest_summary", validation=val)

            entities = extract_entities(summary)
            self._entities[doc_id] = entities
            log.info("Ingest complete: %s", doc_id)
        except Exception as e:
            log.exception("Processing error")
            self.store.set_error(doc_id, f"Processing error: {e}")
        return doc_id

    # def generate_summary(self, *, document_id: str, target_words: int) -> dict:
        doc = self.store.get(document_id)
        if not doc:
            return {"ok": False, "error": "not_found"}
        try:
            chunks = self._reparse_chunks(doc.path)
            self.vstore.upsert_document(document_id, chunks)
            summary = self.summarizer.summarize(chunks, target_words=target_words)
            val = validate_summary(summary, min_words=int(target_words*0.6), max_words=int(target_words*1.6))
            self.store.push_summary_version(document_id, summary, note=f"regen_{target_words}", validation=val)
            self._entities[document_id] = extract_entities(summary)
            return {"ok": True, "validation": val}
        except Exception as e:
            log.exception("Regeneration error")
            self.store.set_error(document_id, f"Processing error: {e}")
            return {"ok": False, "error": "processing_error"}

    def generate_summary(self, *, document_id: str, target_words: int, mode: str = "extractive_mmr", temperature: float = 0.2, seed: int | None = None) -> dict:
        doc = self.store.get(document_id)
        if not doc:
            return {"ok": False, "error": "not_found"}
        try:
            chunks = self._reparse_chunks(doc.path)
            self.vstore.upsert_document(document_id, chunks)
            # compute entities first to feed to structured output
            cur_summary = doc.summary or ""
            ents = extract_entities(cur_summary) if cur_summary else {"names": [], "dates": [], "organizations": []}
            seed_val = int(seed if seed is not None else time.time() % 10_000)
            summary = self.summarizer.summarize(
                chunks,
                target_words=target_words,
                mode=mode,
                temperature=temperature,
                seed=seed_val,
                entities=(ents.get("names", []), ents.get("dates", []), ents.get("organizations", [])),
            )
            val = validate_summary(summary, min_words=int(target_words*0.6), max_words=int(target_words*1.6))
            self.store.push_summary_version(document_id, summary, note=f"regen_{mode}_{target_words}_t{temperature}_seed{seed_val}", validation=val)
            self._entities[document_id] = extract_entities(summary)
            return {"ok": True, "validation": val, "summary": summary}
        except Exception as e:
            log.exception("Regeneration error")
            self.store.set_error(document_id, f"Processing error: {e}")
            return {"ok": False, "error": "processing_error"}

    

    # def generate_summary(self, *, document_id: str, target_words: int, mode: str = "extractive_mmr", temperature: float = 0.2, seed: int | None = None) -> dict:
        doc = self.store.get(document_id)
        if not doc:
            return {"ok": False, "error": "not_found"}
        try:
            chunks = self._reparse_chunks(doc.path)
            self.vstore.upsert_document(document_id, chunks)

            # prepare entities (optional for formatting)
            cur_summary = doc.summary or ""
            ents = extract_entities(cur_summary) if cur_summary else {"names": [], "dates": [], "organizations": []}

            # >>> ensure randomness when seed is None
            seed_val = int(seed) if seed is not None else int(time.time() * 1000) % 1_000_000

            summary = self.summarizer.summarize(
                chunks,
                target_words=target_words,
                mode=mode,
                temperature=temperature,
                seed=seed_val,
                entities=(ents.get("names", []), ents.get("dates", []), ents.get("organizations", [])),
            )

            val = validate_summary(summary, min_words=int(target_words*0.6), max_words=int(target_words*1.6))
            self.store.push_summary_version(
                document_id, summary,
                note=f"regen_{mode}_{target_words}_t{temperature}_seed{seed_val}",
                validation=val
            )
            self._entities[document_id] = extract_entities(summary)
            return {"ok": True, "validation": val, "summary": summary, "seed": seed_val}  # return seed for debugging
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
        val = validate_summary(summary, min_words=int(self.default_summary_words*0.6), max_words=int(self.default_summary_words*1.6))
        self.store.push_summary_version(document_id, summary, note="manual_save", validation=val)
        self._entities[document_id] = extract_entities(summary)
        return {"ok": True, "validation": val}

    def validate_current_summary(self, *, document_id: str) -> dict:
        doc = self.store.get(document_id)
        if not doc or not doc.summary:
            return {"ok": False, "error": "not_found_or_empty"}
        val = validate_summary(doc.summary, min_words=int(self.default_summary_words*0.6), max_words=int(self.default_summary_words*1.6))
        return {"ok": True, "validation": val}

    def list_summary_versions(self, *, document_id: str) -> list[dict]:
        doc = self.store.get(document_id)
        if not doc:
            return []
        out = []
        for idx, v in enumerate(self.store.list_summary_versions(document_id)):
            out.append({
                "index": idx,
                "created_at": v.created_at,
                "note": v.note,
                "validation": v.validation,
                "word_count": v.validation.get("word_count") if v.validation else None
            })
        return out

    def rollback_summary(self, *, document_id: str, version_index: int) -> dict:
        ok = self.store.rollback_summary(document_id, version_index)
        return {"ok": ok}

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
        # (document_ids filtering is future work; we search across all indexed docs for now)
        result = self.qa.answer(question=question, top_k=5)
        val = validate_answer(result["answer"], result.get("contexts", []))
        result["validation"] = val
        return result
