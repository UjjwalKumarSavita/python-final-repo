"""
Lightweight entity extraction (regex/heuristics) for names, dates, orgs.
"""
import re
from typing import Dict, List

DATE_PAT = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
ORG_PAT = re.compile(r"\b([A-Z][A-Za-z&.\- ]+\s+(?:Inc|LLC|Ltd|Limited|LLP|PLC|Corp|Company))\b")
NAME_PAT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

def extract_entities(text: str) -> Dict[str, List[str]]:
    dates = sorted(set(DATE_PAT.findall(text)))
    orgs = sorted(set(ORG_PAT.findall(text)))
    orgs = [o[0] if isinstance(o, tuple) else o for o in orgs]

    names: List[str] = []
    for m in NAME_PAT.findall(text):
        s = m.strip()
        if len(s) > 2 and s.lower() not in {"the", "and"}:
            names.append(s)
    names = sorted(set(names))
    return {"names": names[:50], "dates": dates[:50], "organizations": orgs[:50]}
