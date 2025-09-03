"""
Parses PDF/DOCX/TXT/HTML to plain text.
"""
from pathlib import Path
from bs4 import BeautifulSoup

def parse_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def parse_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)

def parse_docx(path: str) -> str:
    import docx  # python-docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_html(path: str) -> str:
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def parse_file(path: str, ext: str) -> str:
    ext = ext.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".txt"}:
        return parse_text_file(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)
    return parse_text_file(path)
