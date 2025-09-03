# """
# Lightweight entity extraction (regex/heuristics) for names, dates, orgs.
# """
# import re
# from typing import Dict, List

# DATE_PAT = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
# ORG_PAT = re.compile(r"\b([A-Z][A-Za-z&.\- ]+\s+(?:Inc|LLC|Ltd|Limited|LLP|PLC|Corp|Company))\b")
# NAME_PAT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

# def extract_entities(text: str) -> Dict[str, List[str]]:
#     dates = sorted(set(DATE_PAT.findall(text)))
#     orgs = sorted(set(ORG_PAT.findall(text)))
#     orgs = [o[0] if isinstance(o, tuple) else o for o in orgs]

#     names: List[str] = []
#     for m in NAME_PAT.findall(text):
#         s = m.strip()
#         if len(s) > 2 and s.lower() not in {"the", "and"}:
#             names.append(s)
#     names = sorted(set(names))
#     return {"names": names[:50], "dates": dates[:50], "organizations": orgs[:50]}



"""
Lightweight entity extraction (names, dates, organizations) with stricter filtering.
- Strips headings like "Overview", "Key Points", etc. to avoid false positives.
- Names: person-like (single or 2–3 tokens), exclude common capitalized nouns.
- Dates: ISO, slashes, and Month Day, Year patterns.
- Orgs: common suffixes (Inc, LLC, Ltd, Corp, Company, University, Bank, etc.) and keyword patterns.
"""
from __future__ import annotations
import re
from typing import Dict, List

# -------- Preprocessing: remove headings / markdown noise --------
HEADINGS_PHRASES = {
    "overview", "key points", "risks", "open questions",
    "entities", "names", "dates", "organizations",
}

HEADINGS_WORDS = {
    "overview", "key", "points", "risks", "open", "questions",
    "entities", "names", "dates", "organizations",
}

MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
MD_BULLETS_RE = re.compile(r"^\s*[-*+]\s+", re.M)

# -------- Date patterns --------
MONTHS = (
    "Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
    "Sep|Sept|September|Oct|October|Nov|November|Dec|December"
)
DATE_PAT = re.compile(
    rf"\b("
    rf"(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})"             # 12/31/2024 or 12-31-2024
    rf"|(?:\d{{4}}-\d{{2}}-\d{{2}})"                         # 2024-12-31
    rf"|(?:(?:{MONTHS})\s+\d{{1,2}},\s*\d{{4}})"             # December 31, 2024
    rf"|(?:\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}})"              # 31 December 2024
    rf")\b",
    re.I,
)

# -------- Name patterns & filters --------
# Two or three TitleCase tokens, or single TitleCase token (with filters)
NAME_CANDIDATE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

# Common capitalized words we DO NOT want as person names
NAME_STOPWORDS = {
    "I","You","We","He","She","They","This","That","These","Those","However","Instead","Then",
    "Education","Students","Teaching","Overview","Risks","Entities","Names","Dates","Organizations",
    "Key","Points","Open","Questions","Popular","Culture","Essays","Summary","Summarized"
}
# Also exclude all heading words (lowercased match)
NAME_STOPWORDS_LOWER = {w.lower() for w in NAME_STOPWORDS} | HEADINGS_WORDS | {p for p in "and or the a an".split()}

# -------- Organization patterns --------
ORG_SUFFIX = r"(?:Inc|LLC|Ltd|Limited|LLP|PLC|Corp|Corporation|Company|Co\.|AG|GmbH|BV|NV|AB|ASA|Oy|SAS|SA|Sarl)"
ORG_KEYWORDS = (
    "University|College|Institute|Laboratory|Lab|Department|Ministry|Agency|Authority|Council|Committee|"
    "Bank|Exchange|Commission|Foundation|Society|Association|Center|Centre|Board|Office|Bureau|Trust|"
    "United Nations|World Bank|European Commission|IMF|OECD|NATO"
)
# Pattern A: "<Title Case...> <SUFFIX>"
ORG_SUFFIX_PAT = re.compile(rf"\b([A-Z][A-Za-z&.\- ]+?)\s+({ORG_SUFFIX})\b")
# Pattern B: phrases containing org keywords (e.g., "University of X", "Ministry of Y")
ORG_KEYWORD_PAT = re.compile(
    rf"\b([A-Z][A-Za-z&.\- ]*(?:{ORG_KEYWORDS})[A-Za-z&.\- ]*)\b"
)

def _strip_markup(text: str) -> str:
    # remove markdown headers & bullets
    text = MD_HEADER_RE.sub(" ", text)
    text = MD_BULLETS_RE.sub(" ", text)
    # strip common section labels
    for p in sorted(HEADINGS_PHRASES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(p)}\b", " ", text, flags=re.I)
    # strip emphasis markers
    text = re.sub(r"[`*_>#]+", " ", text)
    # collapse spaces
    return re.sub(r"\s+", " ", text).strip()

def _is_bad_name_token(tok: str) -> bool:
    tl = tok.lower()
    return (
        tl in NAME_STOPWORDS_LOWER
        or tl in {m.lower() for m in MONTHS.split("|")}
        or tl in {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
    )

def _keep_name_phrase(phrase: str) -> bool:
    parts = phrase.split()
    # reject heading-like single words and generic capitalized nouns
    if len(parts) == 1:
        return not _is_bad_name_token(parts[0])
    # reject phrases made entirely of heading words (e.g., "Key Points")
    lowers = [p.lower() for p in parts]
    if all(p in HEADINGS_WORDS or p in NAME_STOPWORDS_LOWER for p in lowers):
        return False
    # tiny filter: drop if any token is all-caps (likely acronym) except 2–3 letters
    for p in parts:
        if p.isupper() and len(p) > 3:
            return False
    return True

def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def extract_entities(text: str) -> Dict[str, List[str]]:
    clean = _strip_markup(text or "")

    # ---- Dates
    dates = [m.group(0).strip() for m in DATE_PAT.finditer(clean)]
    dates = _uniq_keep_order(dates)[:50]

    # ---- Orgs
    orgs = []
    for m in ORG_SUFFIX_PAT.finditer(clean):
        orgs.append((m.group(1) + " " + m.group(2)).strip())
    for m in ORG_KEYWORD_PAT.finditer(clean):
        orgs.append(m.group(1).strip())
    # small cleanup: remove trailing punctuation/spaces
    orgs = [re.sub(r"[.,;:)\]]+$", "", o).strip() for o in orgs]
    orgs = _uniq_keep_order(orgs)[:50]

    # ---- Names
    # Extract candidates, then filter hard
    candidates = [m.group(1).strip() for m in NAME_CANDIDATE.finditer(clean)]
    names: List[str] = []
    for cand in candidates:
        if _keep_name_phrase(cand):
            names.append(cand)
    # Remove items that look like organizations we already captured (to reduce noise)
    org_set = {o.lower() for o in orgs}
    names = [n for n in names if n.lower() not in org_set]
    # Quick de-dup and cap
    names = _uniq_keep_order(names)[:50]

    return {"names": names, "dates": dates, "organizations": orgs}
