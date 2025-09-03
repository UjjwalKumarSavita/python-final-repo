"""
Simple validators for summaries and answers.
Return a { ok: bool, score: float, messages: [str] } report.
"""
import re
from typing import Dict, List

WORD_RE = re.compile(r"\b\w+\b")

def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))

def validate_summary(summary: str, min_words: int = 150, max_words: int = 800) -> Dict:
    msgs: List[str] = []
    wc = _word_count(summary)
    if wc < min_words:
        msgs.append(f"Summary too short: {wc} words (min {min_words}).")
    if wc > max_words:
        msgs.append(f"Summary too long: {wc} words (max {max_words}).")

    # Structure signals
    if summary and summary.strip() and not summary.strip().endswith((".", "!", "?")):
        msgs.append("Summary should end with a sentence terminator.")

    # Score: start from 1.0, subtract penalties
    score = 1.0
    if wc < min_words: score -= 0.3
    if wc > max_words: score -= 0.2
    if "  " in summary: score -= 0.05
    if len(summary.strip()) == 0: score = 0.0; msgs.append("Empty summary.")

    ok = score >= 0.6 and wc >= min_words
    return {"ok": ok, "score": round(max(0.0, min(1.0, score)), 2), "messages": msgs, "word_count": wc}

def validate_answer(answer: str, contexts: List[str]) -> Dict:
    msgs: List[str] = []
    wc = _word_count(answer)
    if wc == 0:
        msgs.append("Empty answer.")
        return {"ok": False, "score": 0.0, "messages": msgs, "word_count": wc}

    # Simple “uses context” heuristic: overlap with top-k context words
    ctx_text = " ".join(contexts or [])
    ctx_words = set(w.lower() for w in WORD_RE.findall(ctx_text))
    ans_words = [w.lower() for w in WORD_RE.findall(answer)]
    overlap = sum(1 for w in ans_words if w in ctx_words)
    ratio = overlap / max(1, len(ans_words))
    if ratio < 0.15:
        msgs.append("Low overlap with retrieved context; answer may be hallucinated.")

    score = 0.6 + (ratio * 0.4)  # 0.6..1.0
    ok = score >= 0.7
    return {"ok": ok, "score": round(score, 2), "messages": msgs, "word_count": wc, "overlap_ratio": round(ratio, 2)}
