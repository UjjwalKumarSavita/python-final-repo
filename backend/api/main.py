from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from ..core.config import settings
from ..core.logger import get_logger
from ..core.exceptions import UnsupportedFileType
from ..services.document_store import DocumentStore
from ..services.orchestrator import Orchestrator

log = get_logger("api")

app = FastAPI(title="Intelligent Docs API", version="0.1.0")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons for this small app
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

class QARequest(BaseModel):
    question: str
    document_ids: list[str] | None = None

class QAResponse(BaseModel):
    answer: str
    sources: list[str] = []

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".html"}  # expand later

@app.post("/documents", response_model=DocumentCreateResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document_id = orchestrator.ingest_document(filename=file.filename, saved_path=str(dest))
    return DocumentCreateResponse(document_id=document_id, filename=file.filename)

@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_summary(document_id: str):
    result = orchestrator.get_summary(document_id=document_id)
    return SummaryResponse(document_id=document_id, **result)

@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    result = orchestrator.answer_question(question=request.question, document_ids=request.document_ids)
    return QAResponse(**result)

@app.get("/health")
async def health():
    return {"ok": True, "env": settings.app_env}
