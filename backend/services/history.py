"""
Simple in-memory Q&A history for conversation explorer in the UI.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
from typing import Any, Dict, List

@dataclass
class QARecord:
    ts: float
    question: str
    answer: str
    sources: List[str]
    validation: Dict[str, Any] | None

class QAHistory:
    def __init__(self, max_items: int = 200) -> None:
        self._items: List[QARecord] = []
        self._max = max_items

    def add(self, *, question: str, answer: str, sources: List[str], validation: Dict[str, Any] | None) -> None:
        self._items.append(QARecord(ts=time(), question=question, answer=answer, sources=sources, validation=validation))
        if len(self._items) > self._max:
            self._items = self._items[-self._max:]

    def list(self, last: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in self._items[-last:][::-1]:
            out.append({
                "ts": r.ts,
                "question": r.question,
                "answer": r.answer,
                "sources": r.sources,
                "validation": r.validation,
            })
        return out

qa_history = QAHistory()
