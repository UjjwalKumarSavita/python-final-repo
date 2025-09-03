"""
A tiny document store abstraction.
Milestone 2 will replace internals with Postgres/pgvector but keep signatures.
"""
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

@dataclass
class DocumentRecord:
    id: str
    filename: str
    path: str
    summary: Optional[str] = None
    status: str = "pending"  # pending | ready | error

class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, DocumentRecord] = {}

    def add_document(self, filename: str, path: str) -> str:
        doc_id = str(uuid4())
        self._docs[doc_id] = DocumentRecord(id=doc_id, filename=filename, path=path)
        return doc_id

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    def set_summary(self, doc_id: str, summary: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].summary = summary
            self._docs[doc_id].status = "ready"

    def set_error(self, doc_id: str, message: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].status = "error"
            self._docs[doc_id].summary = message
