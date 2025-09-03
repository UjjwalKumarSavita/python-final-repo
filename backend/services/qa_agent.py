"""
Q&A agent: retrieves top chunks and answers.
- If OpenAI configured: answer with LLM over context.
- Else: simple extractive fallback using the most relevant chunks.
"""
import os
from typing import List, Tuple
from .vector_store import VectorStore

class QAAgent:
    def __init__(self, vstore: VectorStore) -> None:
        self.vstore = vstore
        self.provider = (os.getenv("LLM_PROVIDER") or "").lower()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        if self.provider == "openai" and self.openai_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.openai_key)
            except Exception:
                self._client = None

    def _fallback_answer(self, question: str, contexts: List[str]) -> str:
        # Concatenate top contexts and return a concise stitched answer.
        joined = "\n".join(contexts)
        # Heuristic: take first ~180-220 words from context as a “best-effort” answer.
        words = joined.split()
        if len(words) > 200:
            joined = " ".join(words[:200]) + " ..."
        return f"Based on the most relevant passages:\n\n{joined}"

    def answer(self, *, question: str, top_k: int = 5) -> dict:
        results: List[Tuple[str, int, float, str]] = self.vstore.search(question, top_k=top_k)
        sources = [f"{r[0]}:chunk{r[1]}" for r in results]
        contexts = [r[3] for r in results]

        if self._client and contexts:
            prompt = (
                "Use the context below to answer the user's question. "
                "Be concise and precise. If the context is insufficient, say so.\n\n"
                "=== CONTEXT START ===\n"
                + "\n\n".join(contexts[:5]) +
                "\n=== CONTEXT END ===\n\n"
                f"Question: {question}"
            )
            try:
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                answer = resp.choices[0].message.content.strip()
                return {"answer": answer, "sources": sources, "contexts": contexts[:5]}
            except Exception:
                pass

        # Fallback
        answer = self._fallback_answer(question, contexts[:3])
        return {"answer": answer, "sources": sources, "contexts": contexts[:3]}
