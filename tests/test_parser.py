from pathlib import Path
from bs4 import BeautifulSoup
import docx
from PyPDF2 import PdfReader

def parse_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def parse_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_docx_file(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_html_file(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ", strip=True)

def parse_file(path: str, ext: str) -> str:
    ext = ext.lower()
    if ext == ".txt":
        return parse_text_file(path)
    elif ext == ".pdf":
        return parse_pdf_file(path)
    elif ext == ".docx":
        return parse_docx_file(path)
    elif ext in (".html", ".htm"):
        return parse_html_file(path)
    else:
        # Raise here instead of trying to open
        raise ValueError(f"Unsupported file extension: {ext}")
