"""
Vector store with optional pgvector backend.
"""
from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

from .embeddings import HashedEmbeddings

DB_URL = os.getenv("DATABASE_URL")

class VectorStore:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.embedder = HashedEmbeddings(dim=dim)
        if DB_URL:
            self._init_pg()
        else:
            self._data: list[tuple[str, int, np.ndarray, str]] = []  # (doc_id, chunk_idx, vec, content)

    # ---------- In-memory ----------
    def _upsert_memory(self, document_id: str, chunks: List[str], vecs: np.ndarray) -> None:
        # Remove old entries for this doc
        self._data = [row for row in self._data if row[0] != document_id]
        for i, (c, v) in enumerate(zip(chunks, vecs)):
            self._data.append((document_id, i, v, c))

    def _search_memory(self, qvec: np.ndarray, top_k: int) -> List[Tuple[str, int, float, str]]:
        scores = []
        for doc_id, idx, v, c in self._data:
            score = float(np.dot(qvec, v))  # cosine-ish (vectors normalized)
            scores.append((doc_id, idx, score, c))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    # ---------- Postgres+pgvector ----------
    def _init_pg(self) -> None:
        import psycopg
        self.conn = psycopg.connect(DB_URL, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS doc_embeddings (
                id BIGSERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_idx INT NOT NULL,
                embedding VECTOR({self.dim}) NOT NULL,
                content TEXT NOT NULL
            );
            """)
            # Optional: IVFFlat index requires `SET enable_seqscan = off` for full benefit in some cases
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc ON doc_embeddings(doc_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_embed ON doc_embeddings USING ivfflat (embedding vector_cosine_ops);")
            except Exception:
                pass  # index create can fail if PRAMs not set; safe to continue

    def _upsert_pg(self, document_id: str, chunks: List[str], vecs: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM doc_embeddings WHERE doc_id=%s;", (document_id,))
            for i, (c, v) in enumerate(zip(chunks, vecs)):
                # psycopg 3 can adapt Python lists to pgvector with the extension installed
                cur.execute(
                    "INSERT INTO doc_embeddings (doc_id, chunk_idx, embedding, content) VALUES (%s,%s,%s,%s);",
                    (document_id, i, v.tolist(), c),
                )

    def _search_pg(self, qvec: np.ndarray, top_k: int) -> List[Tuple[str, int, float, str]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, chunk_idx, 1 - (embedding <=> %s) AS score, content "
                "FROM doc_embeddings ORDER BY embedding <=> %s ASC LIMIT %s;",
                (qvec.tolist(), qvec.tolist(), top_k),
            )
            return [(r[0], r[1], float(r[2]), r[3]) for r in cur.fetchall()]

    # ---------- Public API ----------
    def upsert_document(self, document_id: str, chunks: List[str]) -> None:
        vecs = self.embedder.embed_texts(chunks)
        if DB_URL:
            self._upsert_pg(document_id, chunks, vecs)
        else:
            self._upsert_memory(document_id, chunks, vecs)

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        qvec = self.embedder.embed_texts([query_text])[0]
        if DB_URL:
            return self._search_pg(qvec, top_k)
        return self._search_memory(qvec, top_k)
