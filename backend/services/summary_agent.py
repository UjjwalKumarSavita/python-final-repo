# """
# Summarization agent with target word control.
# Uses OpenAI if configured; otherwise a simple extractive fallback that
# selects informative sentences until ~target_words is reached.
# """
# import os
# import re
# from typing import List

# _STOP = {
#     "the","and","a","an","of","to","in","on","for","with","by","as","is","it","that",
#     "this","are","was","were","be","or","at","from","which","but","if","then","so",
# }

# _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# def _word_count(text: str) -> int:
#     return len([w for w in re.findall(r"\b\w+\b", text)])

# def _extractive_summary(text: str, target_words: int = 350, max_sents: int = 30) -> str:
#     # Split into sentences
#     sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
#     if not sents:
#         return text[:200]  # fallback tiny

#     # Build global term frequencies
#     words = re.findall(r"\b\w+\b", text.lower())
#     freqs = {}
#     for w in words:
#         if len(w) < 3 or w in _STOP:
#             continue
#         freqs[w] = freqs.get(w, 0) + 1

#     # Score sentences by sum of token frequencies
#     scored = []
#     for i, s in enumerate(sents):
#         toks = [w for w in re.findall(r"\b\w+\b", s.lower()) if len(w) >= 3 and w not in _STOP]
#         score = sum(freqs.get(w, 0) for w in toks) / (len(toks) + 1e-6)
#         scored.append((i, score, s))

#     # Pick top sentences, but present in original order
#     scored.sort(key=lambda x: x[1], reverse=True)
#     chosen = []
#     chosen_idx = set()
#     total_words = 0
#     for i, _, s in scored[:max_sents]:
#         if i in chosen_idx:
#             continue
#         chosen.append((i, s))
#         chosen_idx.add(i)
#         total_words += _word_count(s)
#         if total_words >= target_words:
#             break
#     chosen.sort(key=lambda x: x[0])
#     result = " ".join(s for _, s in chosen)

#     # If still short, append next sentences in order until target
#     j = 0
#     while _word_count(result) < target_words and j < len(sents):
#         if j not in chosen_idx:
#             result += (" " if result else "") + sents[j]
#         j += 1
#     return result

# class SummaryAgent:
#     def __init__(self) -> None:
#         self.provider = (os.getenv("LLM_PROVIDER") or "").lower()
#         self.openai_key = os.getenv("OPENAI_API_KEY")
#         self._client = None
#         if self.provider == "openai" and self.openai_key:
#             try:
#                 from openai import OpenAI
#                 self._client = OpenAI(api_key=self.openai_key)
#             except Exception:
#                 self._client = None

#     def summarize(self, chunks: List[str], target_words: int = 350) -> str:
#         # Cap the amount of text we feed (keep it snappy)
#         text = "\n".join(chunks)[:50000]

#         if self._client:
#             prompt = (
#                 f"Write a clear, well-structured summary of the document in ~{target_words} words. "
#                 "Prioritize key points, obligations, dates, parties, decisions, and definitions. "
#                 "Use short paragraphs and keep it editable.\n\n"
#                 "[DOCUMENT START]\n"
#                 f"{text}\n"
#                 "[DOCUMENT END]\n"
#             )
#             try:
#                 resp = self._client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.2,
#                 )
#                 out = resp.choices[0].message.content.strip()
#                 # Light trim to within ~target_words ±20%
#                 words = out.split()
#                 if len(words) > int(target_words * 1.2):
#                     out = " ".join(words[: int(target_words * 1.2)])
#                 return out
#             except Exception:
#                 pass

#         # Fallback extractive summary
#         return _extractive_summary(text, target_words=target_words)

"""
Improved SummaryAgent with:
- modes: "abstractive" (LLM) or "extractive_mmr" (fallback/diverse)
- target_words, temperature, seed to vary regenerations
- structured output sections
"""
import os
import re
import random
from typing import List, Tuple
import numpy as np

from .embeddings import HashedEmbeddings

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b\w+\b")

def _wc(txt: str) -> int:
    return len(_WORD.findall(txt or ""))

def _sentences(text: str) -> List[str]:
    s = [x.strip() for x in _SENT_SPLIT.split(text.strip()) if x.strip()]
    return s if s else ([text.strip()] if text.strip() else [])

def _tf_score(text: str) -> dict:
    freqs = {}
    for w in _WORD.findall(text.lower()):
        if len(w) >= 3:
            freqs[w] = freqs.get(w, 0) + 1
    return freqs

def _mmr_select(sents: List[str], k: int, embedder: HashedEmbeddings, lambda_weight: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance: diversity-aware top-k sentence indices."""
    if not sents:
        return []
    vecs = embedder.embed_texts(sents)
    centroid = np.mean(vecs, axis=0)
    # relevance: similarity to centroid
    rel = vecs @ centroid
    chosen: List[int] = []
    cand = set(range(len(sents)))
    while len(chosen) < min(k, len(sents)):
        best_i = None
        best_score = -1e9
        for i in cand:
            div = 0.0
            if chosen:
                div = max(float(vecs[i] @ vecs[j]) for j in chosen)
            score = lambda_weight * float(rel[i]) - (1 - lambda_weight) * div
            if score > best_score:
                best_score = score
                best_i = i
        chosen.append(best_i)  # type: ignore
        cand.remove(best_i)    # type: ignore
    chosen.sort()
    return chosen

def _format_structured(summary_body: str, names: List[str], dates: List[str], orgs: List[str]) -> str:
    def bullets(items: List[str], limit=8) -> str:
        return "\n".join(f"- {x}" for x in items[:limit]) if items else "- (none)"
    return (
        "## Overview\n"
        f"{summary_body.strip()}\n\n"
        # "## Key Points\n"
        # "- Summarized for quick review\n\n"
        # "## Dates\n"
        # f"{bullets(dates)}\n\n"
        # "## Entities\n"
        # f"**Names**:\n{bullets(names)}\n\n"
        # f"**Organizations**:\n{bullets(orgs)}\n\n"
        # "## Risks / Open Questions\n"
        # "- (add any risks, ambiguities, or follow-ups here)"
    )

class SummaryAgent:
    def __init__(self) -> None:
        self.provider = (os.getenv("LLM_PROVIDER") or "").lower()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        if self.provider == "openai" and self.openai_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.openai_key)
            except Exception:
                self._client = None
        self.embedder = HashedEmbeddings(dim=384)

    def _extractive_mmr(self, text: str, target_words: int, seed: int) -> str:
        random.seed(seed)
        sents = _sentences(text)
        if not sents:
            return ""

        # Diverse top sentences via MMR
        k = max(5, min(len(sents), target_words // 20))
        chosen_idx = _mmr_select(sents, k=k, embedder=self.embedder, lambda_weight=0.72)
        chosen = [sents[i] for i in chosen_idx]

        # Expand until target by appending additional sentences in order
        out = " ".join(chosen)
        i = 0
        while _wc(out) < target_words and i < len(sents) * 2:
            cand = sents[i % len(sents)]
            if cand not in chosen:  # avoid duplicates
                out += " " + cand
            i += 1

        # Trim to ~target
        words = out.split()
        cap = int(target_words * 1.15)
        if len(words) > cap:
            out = " ".join(words[:cap])
        return out

    def _abstractive_llm(self, text: str, target_words: int, temperature: float, seed: int) -> str:
        if not self._client:
            return ""
        sys_hint = (
            f"You are a professional analyst. Seed={seed}. "
            "Write a clear, structured, and editable summary."
        )
        user_prompt = (
            f"Target length: ~{target_words} words.\n"
            "Format sections as:\n"
            "## Overview\n## Key Points\n## Dates\n## Entities\n## Risks / Open Questions\n\n"
            "[DOCUMENT START]\n" + text + "\n[DOCUMENT END]"
        )
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_hint},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(0.0, min(1.0, temperature)),
            )
            out = resp.choices[0].message.content.strip()
            # light trim
            w = out.split()
            if len(w) > int(target_words * 1.25):
                out = " ".join(w[: int(target_words * 1.25)])
            return out
        except Exception:
            return ""

    def summarize(
        self,
        chunks: List[str],
        target_words: int = 350,
        mode: str = "extractive_mmr",
        temperature: float = 0.2,
        seed: int = 42,
        entities: Tuple[List[str], List[str], List[str]] | None = None,  # (names, dates, orgs)
    ) -> str:
        text = "\n".join(chunks)[:60000]
        if not text.strip():
            return ""

        # Try LLM if requested
        if mode == "abstractive":
            out = self._abstractive_llm(text, target_words, temperature, seed)
            if out:
                return out

        # Fallback: diverse extractive + structured packaging
        body = self._extractive_mmr(text, target_words, seed)
        names, dates, orgs = (entities or ([], [], []))
        return _format_structured(body, names, dates, orgs)
