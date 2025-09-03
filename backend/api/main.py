from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from typing import Any

from ..core.config import settings
from ..core.logger import get_logger
from ..services.document_store import DocumentStore
from ..services.orchestrator import Orchestrator

log = get_logger("api")

app = FastAPI(title="Intelligent Docs API", version="0.3.0")  # bumped

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = DocumentStore()
orchestrator = Orchestrator(store=store)

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

class SummarizeRequest(BaseModel):  # NEW
    target_words: int = Field(ge=50, le=1200, default=settings.summary_words_default)

class EntitiesResponse(BaseModel):
    status: str
    entities: dict[str, Any] | None = None

class EntitiesSaveRequest(BaseModel):
    entities: dict[str, Any]

class QARequest(BaseModel):
    question: str
    document_ids: list[str] | None = None

class QAResponse(BaseModel):
    answer: str
    sources: list[str] = []

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".html", ".htm"}

@app.post("/documents", response_model=DocumentCreateResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document_id = orchestrator.ingest_document(filename=file.filename, saved_path=str(dest))
    status = store.get(document_id).status
    return DocumentCreateResponse(document_id=document_id, filename=file.filename, status=status)

@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_summary(document_id: str):
    result = orchestrator.get_summary(document_id=document_id)
    return SummaryResponse(document_id=document_id, **result)

@app.post("/documents/{document_id}/summary")
async def save_summary(document_id: str, req: SummarySaveRequest):
    result = orchestrator.save_summary(document_id=document_id, summary=req.summary)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}

@app.post("/documents/{document_id}/summarize")  # NEW
async def regenerate_summary(document_id: str, req: SummarizeRequest):
    ok = orchestrator.generate_summary(document_id=document_id, target_words=req.target_words)
    if not ok.get("ok"):
        raise HTTPException(status_code=404, detail=ok.get("error","regenerate_failed"))
    return {"ok": True}

@app.get("/documents/{document_id}/entities", response_model=EntitiesResponse)
async def get_entities(document_id: str):
    return orchestrator.get_entities(document_id=document_id)

@app.post("/documents/{document_id}/entities")
async def save_entities(document_id: str, req: EntitiesSaveRequest):
    return orchestrator.save_entities(document_id=document_id, entities=req.entities)

@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    result = orchestrator.answer_question(question=request.question, document_ids=request.document_ids)
    return QAResponse(**result)

@app.get("/health")
async def health():
    return {"ok": True, "env": settings.app_env}
