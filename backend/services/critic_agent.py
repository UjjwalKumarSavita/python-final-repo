"""
Critic Agent:
- Reviews summaries and answers for issues like bias, sensitive info, or incompleteness.
- Returns a structured report.
"""
import re
from typing import Dict, List

SENSITIVE = {"password", "ssn", "credit card", "api key"}

def critic_summary(summary: str) -> Dict:
    issues: List[str] = []
    if not summary or not summary.strip():
        issues.append("Empty summary.")
    if "lorem ipsum" in summary.lower():
        issues.append("Looks like placeholder text.")
    if len(summary.split()) < 100:
        issues.append("Too short to be useful.")
    for s in SENSITIVE:
        if s in summary.lower():
            issues.append(f"Contains sensitive term: {s}")

    return {"ok": len(issues) == 0, "issues": issues}

def critic_answer(answer: str) -> Dict:
    issues: List[str] = []
    if not answer or not answer.strip():
        issues.append("Empty answer.")
    if re.search(r"\b(I think|maybe|not sure)\b", answer, flags=re.I):
        issues.append("Uncertain phrasing detected.")
    for s in SENSITIVE:
        if s in answer.lower():
            issues.append(f"Contains sensitive term: {s}")

    return {"ok": len(issues) == 0, "issues": issues}
