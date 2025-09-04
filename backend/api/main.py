from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.logger import get_logger

from ..services.document_store import DocumentStore
from ..services.parser import parse_file
from ..services.chunker import chunk_text
from ..services.summary_agent import SummaryAgent
from ..services.entity_extraction import extract_entities
from ..services.validator_agent import validate_summary, validate_answer
from ..services.vector_store import VectorStore
from ..services.vector_store_pg import PgVectorStore
from ..services.qa_agent import QAAgent

# Milestone 5 additions
from ..obs.middleware import RequestLoggingMiddleware
from ..obs.events import record_event
from ..services.history import qa_history
from ..services.mcp_servers import MCPFileOps, MCPSearch
from ..utils.pdf import summary_to_pdf_bytes

log = get_logger("api")

# --- AUTOGEN toggle/import (AgentChat) ---
USE_AUTOGEN = os.getenv("USE_AUTOGEN", "0") == "1"
AUTOGEN_FLAVOR = os.getenv("AUTOGEN_FLAVOR", "agentchat")
autogen_summarize_async = None
if USE_AUTOGEN and AUTOGEN_FLAVOR == "agentchat":
    try:
        from ..agents.autogen_team_ac import autogen_summarize_async  # type: ignore
    except Exception:
        autogen_summarize_async = None  # type: ignore

app = FastAPI(title="Intelligent Docs API", version="0.6.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware (observability)
app.add_middleware(RequestLoggingMiddleware)

# Global exception handler → log to events.jsonl
@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception):
    record_event("error", {"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "internal_error"})

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = DocumentStore()
summarizer = SummaryAgent()

# vector store selection (env or settings)
db_url = (getattr(settings, "database_url", None) or os.getenv("DATABASE_URL"))
use_pg_flag = (str(getattr(settings, "use_pgvector", 0)) == "1" or os.getenv("USE_PGVECTOR") == "1")
if use_pg_flag and not db_url:
    log.warning("USE_PGVECTOR=1 but DATABASE_URL is missing; falling back to in-memory VectorStore.")
    use_pg_flag = False
vstore = PgVectorStore(dim=settings.pgvector_dim) if use_pg_flag else VectorStore(dim=settings.pgvector_dim)
qa = QAAgent(vstore)

# MCP search helper (shares vstore)
_mcp_search = MCPSearch(vstore)

# ----------------- Models -----------------
class DocumentCreateResponse(BaseModel):
    document_id: str
    filename: str
    status: str = "pending"

class SummaryResponse(BaseModel):
    document_id: str
    status: str = Field(description="pending | ready | error | not_found")
    summary: str | None = None

class SummarySaveRequest(BaseModel):
    summary: str

class SummarizeRequest(BaseModel):
    target_words: int = Field(ge=50, le=1200, default=settings.summary_words_default)
    mode: str = Field(default="extractive_mmr")  # "extractive_mmr" | "abstractive" | "autogen"
    temperature: float = Field(ge=0.0, le=1.0, default=0.2)
    seed: int | None = None

class EntitiesResponse(BaseModel):
    status: str
    entities: dict[str, Any] | None = None

class EntitiesSaveRequest(BaseModel):
    entities: dict[str, Any]

class QARequest(BaseModel):
    question: str
    document_ids: List[str] | None = None

class QAResponse(BaseModel):
    answer: str
    sources: List[str] = []
    validation: dict | None = None

class VersionsResponseItem(BaseModel):
    index: int
    created_at: str
    note: str | None = None
    validation: dict | None = None
    word_count: int | None = None

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".html", ".htm"}

# ----------------- Endpoints -----------------
@app.get("/health")
async def health():
    return {"ok": True, "env": settings.app_env, "autogen": USE_AUTOGEN}

@app.post("/documents", response_model=DocumentCreateResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = store.add_document(filename=file.filename, path=str(dest))

    try:
        text = parse_file(str(dest), ext)
        chunks = chunk_text(text)
        vstore.upsert_document(doc_id, chunks)

        target = settings.summary_words_default
        summary = summarizer.summarize(chunks, target_words=target)
        val = validate_summary(summary, min_words=int(target*0.6), max_words=int(target*1.6))
        store.push_summary_version(doc_id, summary, note="ingest_summary", validation=val)
        status = "ready"
        record_event("ingest", {"doc_id": doc_id, "filename": file.filename})
    except Exception as e:
        log.exception("Processing error")
        store.set_error(doc_id, f"Processing error: {e}")
        status = "error"
        record_event("error", {"doc_id": doc_id, "filename": file.filename, "error": str(e)})

    return DocumentCreateResponse(document_id=doc_id, filename=file.filename, status=status)

@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_summary(document_id: str):
    doc = store.get(document_id)
    if not doc:
        return SummaryResponse(document_id=document_id, status="not_found", summary=None)
    if doc.status == "error":
        return SummaryResponse(document_id=document_id, status="error", summary=doc.error_message)
    return SummaryResponse(document_id=document_id, status=doc.status, summary=doc.summary or "")

@app.post("/documents/{document_id}/summary")
async def save_summary(document_id: str, req: SummarySaveRequest):
    doc = store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    val = validate_summary(req.summary, min_words=int(settings.summary_words_default*0.6), max_words=int(settings.summary_words_default*1.6))
    store.push_summary_version(document_id, req.summary, note="manual_save", validation=val)
    record_event("summary_save", {"doc_id": document_id})
    return {"ok": True, "validation": val}

@app.post("/documents/{document_id}/summarize")
async def regenerate_summary(document_id: str, req: SummarizeRequest):
    doc = store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        text = parse_file(doc.path)
        chunks = chunk_text(text)
        vstore.upsert_document(document_id, chunks)
        seed_val = int(req.seed) if req.seed is not None else int(time.time() * 1000) % 1_000_000

        if req.mode == "autogen":
            if not (USE_AUTOGEN and autogen_summarize_async):
                raise HTTPException(status_code=501, detail="AutoGen not enabled or not installed.")
            summary = await autogen_summarize_async(" ".join(chunks), target_words=req.target_words, temperature=req.temperature, seed=seed_val)
        elif req.mode == "abstractive":
            summary = summarizer.summarize(chunks, target_words=req.target_words, mode="abstractive", temperature=req.temperature, seed=seed_val)
        else:
            summary = summarizer.summarize(chunks, target_words=req.target_words, mode="extractive_mmr", seed=seed_val)

        val = validate_summary(summary, min_words=int(req.target_words*0.6), max_words=int(req.target_words*1.6))
        store.push_summary_version(document_id, summary, note=f"regen_{req.mode}_{req.target_words}_seed{seed_val}", validation=val)
        record_event("summary_regen", {"doc_id": document_id, "mode": req.mode, "seed": seed_val})
        return {"ok": True, "validation": val, "summary": summary, "seed": seed_val}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Regenerate failed")
        record_event("error", {"doc_id": document_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/summary/validate")
async def validate_summary_route(document_id: str):
    doc = store.get(document_id)
    if not doc or not doc.summary:
        raise HTTPException(status_code=404, detail="not_found_or_empty")
    val = validate_summary(doc.summary, min_words=int(settings.summary_words_default*0.6), max_words=int(settings.summary_words_default*1.6))
    return {"ok": True, "validation": val}

@app.get("/documents/{document_id}/entities", response_model=EntitiesResponse)
async def get_entities(document_id: str):
    doc = store.get(document_id)
    if not doc or not doc.summary:
        return EntitiesResponse(status="pending", entities=None)
    ents = extract_entities(doc.summary)
    return EntitiesResponse(status="ready", entities=ents)

@app.post("/documents/{document_id}/entities")
async def save_entities(document_id: str, req: EntitiesSaveRequest):
    doc = store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    record_event("entities_save", {"doc_id": document_id})
    return {"ok": True}

@app.get("/documents/{document_id}/summary/versions", response_model=List[VersionsResponseItem])
async def list_versions(document_id: str):
    doc = store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    items: List[VersionsResponseItem] = []
    for idx, v in enumerate(doc.versions):
        items.append(VersionsResponseItem(
            index=idx,
            created_at=v.created_at,
            note=v.note,
            validation=v.validation,
            word_count=(v.validation or {}).get("word_count"),
        ))
    return items

@app.post("/documents/{document_id}/summary/rollback")
async def rollback(document_id: str, version_index: int = 0):
    ok = store.rollback_summary(document_id, version_index)
    if not ok:
        raise HTTPException(status_code=404, detail="rollback_failed")
    record_event("summary_rollback", {"doc_id": document_id, "version_index": version_index})
    return {"ok": True}

@app.post("/qa", response_model=QAResponse)
async def qa_route(req: QARequest):
    result = qa.answer(question=req.question, top_k=5)
    val = validate_answer(result["answer"], result.get("contexts", []))
    qa_history.add(question=req.question, answer=result["answer"], sources=result.get("sources", []), validation=val)
    record_event("qa", {"question": req.question, "sources": result.get("sources", []), "score": val.get("score")})
    return QAResponse(answer=result["answer"], sources=result.get("sources", []), validation=val)

# ---------- Milestone 5: History & Export ----------
@app.get("/qa/history")
async def qa_history_list(limit: int = 50):
    return {"items": qa_history.list(last=limit)}

@app.get("/documents/{document_id}/export/summary.md")
async def export_summary_md(document_id: str):
    doc = store.get(document_id)
    if not doc or doc.status != "ready" or not doc.summary:
        raise HTTPException(status_code=404, detail="summary_not_ready")
    return PlainTextResponse(doc.summary, media_type="text/markdown")

@app.get("/documents/{document_id}/export/entities.json")
async def export_entities_json(document_id: str):
    doc = store.get(document_id)
    if not doc or not doc.summary:
        raise HTTPException(status_code=404, detail="entities_not_ready")
    ents = extract_entities(doc.summary)
    return JSONResponse(ents)

@app.get("/documents/{document_id}/export/summary.pdf")
async def export_summary_pdf(document_id: str):
    doc = store.get(document_id)
    if not doc or doc.status != "ready" or not doc.summary:
        raise HTTPException(status_code=404, detail="summary_not_ready")
    pdf_bytes = summary_to_pdf_bytes(title=f"Summary - {doc.filename}", text=doc.summary)
    headers = {"Content-Disposition": f'attachment; filename="summary_{document_id}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

# ---------- Milestone 5: MCP-like endpoints ----------
@app.get("/mcp/files")
async def mcp_list_files():
    return {"files": MCPFileOps.list_files()}

@app.get("/mcp/files/read")
async def mcp_read_file(name: str):
    try:
        content = MCPFileOps.read_file(name)
        return {"name": name, "content": content[:20000]}  # cap to 20k chars
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file_not_found")

class MCPWriteBody(BaseModel):
    name: str
    content: str

@app.post("/mcp/files/write")
async def mcp_write_file(body: MCPWriteBody):
    path = MCPFileOps.write_text(body.name, body.content)
    return {"ok": True, "path": path}

@app.get("/mcp/search")
async def mcp_search(q: str, k: int = 5):
    return {"results": _mcp_search.search(q, top_k=k)}
