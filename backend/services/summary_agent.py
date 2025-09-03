"""
Summarization agent with target word control.
Uses OpenAI if configured; otherwise a simple extractive fallback that
selects informative sentences until ~target_words is reached.
"""
import os
import re
from typing import List

_STOP = {
    "the","and","a","an","of","to","in","on","for","with","by","as","is","it","that",
    "this","are","was","were","be","or","at","from","which","but","if","then","so",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _word_count(text: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", text)])

def _extractive_summary(text: str, target_words: int = 350, max_sents: int = 30) -> str:
    # Split into sentences
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents:
        return text[:200]  # fallback tiny

    # Build global term frequencies
    words = re.findall(r"\b\w+\b", text.lower())
    freqs = {}
    for w in words:
        if len(w) < 3 or w in _STOP:
            continue
        freqs[w] = freqs.get(w, 0) + 1

    # Score sentences by sum of token frequencies
    scored = []
    for i, s in enumerate(sents):
        toks = [w for w in re.findall(r"\b\w+\b", s.lower()) if len(w) >= 3 and w not in _STOP]
        score = sum(freqs.get(w, 0) for w in toks) / (len(toks) + 1e-6)
        scored.append((i, score, s))

    # Pick top sentences, but present in original order
    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    chosen_idx = set()
    total_words = 0
    for i, _, s in scored[:max_sents]:
        if i in chosen_idx:
            continue
        chosen.append((i, s))
        chosen_idx.add(i)
        total_words += _word_count(s)
        if total_words >= target_words:
            break
    chosen.sort(key=lambda x: x[0])
    result = " ".join(s for _, s in chosen)

    # If still short, append next sentences in order until target
    j = 0
    while _word_count(result) < target_words and j < len(sents):
        if j not in chosen_idx:
            result += (" " if result else "") + sents[j]
        j += 1
    return result

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

    def summarize(self, chunks: List[str], target_words: int = 350) -> str:
        # Cap the amount of text we feed (keep it snappy)
        text = "\n".join(chunks)[:50000]

        if self._client:
            prompt = (
                f"Write a clear, well-structured summary of the document in ~{target_words} words. "
                "Prioritize key points, obligations, dates, parties, decisions, and definitions. "
                "Use short paragraphs and keep it editable.\n\n"
                "[DOCUMENT START]\n"
                f"{text}\n"
                "[DOCUMENT END]\n"
            )
            try:
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                out = resp.choices[0].message.content.strip()
                # Light trim to within ~target_words ±20%
                words = out.split()
                if len(words) > int(target_words * 1.2):
                    out = " ".join(words[: int(target_words * 1.2)])
                return out
            except Exception:
                pass

        # Fallback extractive summary
        return _extractive_summary(text, target_words=target_words)
