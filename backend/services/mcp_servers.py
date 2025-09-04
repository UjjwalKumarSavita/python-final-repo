"""
Tiny "MCP-like" servers (local stubs) exposed via HTTP routes:
- File operations in backend/data/uploads/
- Vector search over your corpus (pgvector or in-memory, same interface)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from .parser import parse_file
from .vector_store import VectorStore  # interface type only

UPLOADS = Path(__file__).resolve().parents[1] / "data" / "uploads"

class MCPFileOps:
    @staticmethod
    def list_files() -> List[str]:
        if not UPLOADS.exists():
            return []
        return sorted([p.name for p in UPLOADS.iterdir() if p.is_file()])

    @staticmethod
    def read_file(name: str) -> str:
        p = UPLOADS / name
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(name)
        return parse_file(str(p), p.suffix.lower())

    @staticmethod
    def write_text(name: str, content: str) -> str:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        p = UPLOADS / name
        p.write_text(content, encoding="utf-8")
        return str(p)

class MCPSearch:
    def __init__(self, vstore: VectorStore) -> None:
        self.vs = vstore

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        rows = self.vs.search(query, top_k=top_k)  # (doc_id, idx, score, content)
        return [{"doc_id": d, "chunk_idx": i, "score": s, "snippet": c[:300]} for (d, i, s, c) in rows]
