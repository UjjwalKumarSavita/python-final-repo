from dataclasses import dataclass, field
from typing import Optional, List, Dict
from uuid import uuid4
from datetime import datetime

@dataclass
class SummaryVersion:
    content: str
    created_at: str
    note: str = ""
    validation: Optional[Dict] = None

@dataclass
class DocumentRecord:
    id: str
    filename: str
    path: str
    summary: Optional[str] = None
    status: str = "pending"  # pending | ready | error
    summary_versions: List[SummaryVersion] = field(default_factory=list)

class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, DocumentRecord] = {}

    def add_document(self, filename: str, path: str) -> str:
        doc_id = str(uuid4())
        self._docs[doc_id] = DocumentRecord(id=doc_id, filename=filename, path=path)
        return doc_id

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def push_summary_version(self, doc_id: str, content: str, note: str = "", validation: Optional[Dict] = None) -> int:
        if doc_id not in self._docs:
            return -1
        ver = SummaryVersion(content=content, created_at=self._now(), note=note, validation=validation)
        self._docs[doc_id].summary_versions.append(ver)
        # also set current summary to this version
        self._docs[doc_id].summary = content
        self._docs[doc_id].status = "ready"
        return len(self._docs[doc_id].summary_versions) - 1

    def set_summary(self, doc_id: str, summary: str) -> None:
        # keep compatibility with older calls (no note/validation)
        self.push_summary_version(doc_id, summary, note="set_summary")

    def set_error(self, doc_id: str, message: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].status = "error"
            self._docs[doc_id].summary = message

    def list_summary_versions(self, doc_id: str) -> List[SummaryVersion]:
        doc = self._docs.get(doc_id)
        return doc.summary_versions if doc else []

    def rollback_summary(self, doc_id: str, version_index: int) -> bool:
        doc = self._docs.get(doc_id)
        if not doc:
            return False
        if 0 <= version_index < len(doc.summary_versions):
            chosen = doc.summary_versions[version_index]
            doc.summary = chosen.content
            # push a new version marking rollback (so history is linear)
            self.push_summary_version(doc_id, chosen.content, note=f"rollback_to_{version_index}", validation=chosen.validation)
            return True
        return False
