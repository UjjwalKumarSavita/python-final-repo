"""
Simple character-based chunking with overlap.
"""
from typing import List

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks
