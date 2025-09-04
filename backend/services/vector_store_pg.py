from __future__ import annotations
import os
import math
from typing import List, Tuple

import psycopg  # pip install psycopg[binary]
from .embeddings import HashedEmbeddings


def _l2_normalize(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _vec_lit(v: List[float]) -> str:
    # pgvector literal format: '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgVectorStore:
    """
    pgvector-backed vector store with the same interface as the in-memory store.

    Schema required (run once in your DB):
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS chunks (
          doc_id TEXT NOT NULL,
          chunk_idx INT NOT NULL,
          content TEXT NOT NULL,
          embedding VECTOR(384) NOT NULL,
          PRIMARY KEY (doc_id, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivf
          ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        ANALYZE chunks;

    Env:
        DATABASE_URL=postgresql://postgres:PASS@127.0.0.1:5432/mydb
        PGVECTOR_DIM=384
    """

    def __init__(self, dim: int | None = None, dsn: str | None = None) -> None:
        self.dim = int(dim or os.getenv("PGVECTOR_DIM", "384"))
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("DATABASE_URL not set")
        # one simple autocommit connection is enough for this app
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.embedder = HashedEmbeddings(dim=self.dim)
        # Safety: ensure extension exists (no-op if already installed)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def upsert_document(self, doc_id: str, chunks: List[str]) -> None:
        if not chunks:
            return
        embs = self.embedder.embed_texts(chunks)
        embs = [_l2_normalize(list(map(float, e))) for e in embs]
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))
            for i, (content, emb) in enumerate(zip(chunks, embs)):
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_idx, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (doc_id, i, content, _vec_lit(emb)),
                )

    def delete_document(self, doc_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        q = self.embedder.embed_texts([query])[0]
        q = _l2_normalize(list(map(float, q)))
        with self.conn.cursor() as cur:
            # cosine distance operator `<=>` (smaller is closer).
            # Convert to similarity in [(-inf..1] as (1 - distance)
            cur.execute(
                """
                SELECT doc_id, chunk_idx, content, (1 - (embedding <=> %s::vector)) AS sim
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vec_lit(q), _vec_lit(q), top_k),
            )
            rows = cur.fetchall()
        # return (doc_id, chunk_idx, similarity, content)
        return [(r[0], int(r[1]), float(r[3]), r[2]) for r in rows]
