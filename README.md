# Intelligent Document Summarization & Q&A

A FastAPI + Streamlit app that ingests documents, parses & chunks them, builds embeddings (in-memory or **pgvector**), generates editable summaries (extractive, abstractive, or **AutoGen** team), extracts entities (names/dates/organizations), and answers natural-language questions over the corpus. Includes summary versioning/rollback, exports (MD/PDF/JSON), MCP-style utilities, observability logs, deep health checks, and tests/coverage.

---

## Features

- **Upload & Parse**: PDF / DOCX / TXT / HTML → clean text
- **Summarization**:
  - Extractive (MMR over representative chunks)
  - Abstractive (LLM rewrite)
  - **AutoGen** (Planner → Summarizer → Critic) — optional
- **Human-in-the-loop**: editable summaries, save versions, validate, rollback
- **Entities**: names, dates, organizations (editable JSON)
- **Q&A**: retrieval-augmented answers with cited sources + quick validation score
- **Vector store**: in-memory or **Postgres + pgvector**
- **Exports**: summary → Markdown/PDF, entities → JSON
- **Observability**: request/event JSONL logs, optional prompt logs
- **Health**: `GET /health` (shallow) + `GET /health?deep=true` (store/LLM/uploads checks)
- **MCP-style tools**: simple file ops + search view over the indexed corpus
- **Quality**: pytest(+cov), ruff, black, isort, mypy, pre-commit, CI workflow

---

## Project structure

********
backend/
**
api/
main.py # REST endpoints + CORS + global error handler + background ingest

**
agents/
autogen_team_ac.py # AutoGen team (Planner, Summarizer, Critic) - optional

**
core/
config.py # Pydantic settings (.env)
logger.py # Central logger
exceptions.py # Shared exception types

**
data/
logs/ # events.jsonl, prompts.jsonl (runtime)
uploads/ # uploaded docs (runtime)

**
obs/
events.py # record_event() → events.jsonl
middleware.py # RequestLoggingMiddleware
promptlog.py # (optional) prompt/response logging

**
services/
chunker.py ::  chunk_text()
document_store.py ::  in-memory doc registry + summary versioning/rollback
embeddings.py :: embed_texts() (batched)
entity_extraction.py :: names/dates/organizations (precise filters)
history.py ::  Q&A history (in-memory)
mcp_servers.py ::  MCP-style file ops + search facade
parser.py # PDF/DOCX/TXT/HTML → text (PyMuPDF preferred, PyPDF2 fallback)
qa_agent.py ::  retrieval-augmented QA
summary_agent.py ::  extractive/abstractive summarizer (temp/seed support)
validator_agent.py :: summary/answer validators (shape/length/score)
vector_store.py :: in-memory vectors
vector_store_pg.py :: pgvector implementation (bulk upserts)

**
utils/
pdf.py # summary_to_pdf_bytes()

**
tests/
conftest.py # test isolation (tmp uploads, USE_PGVECTOR=0)
test_api.py
test_chunker.py
test_entities.py
test_parser.py
test_performance.py
test_qa.py
test_summary.py
test_health_deep.py


*******
ui/
app.py ::  Streamlit UI (upload, regenerate, validate, rollback, Q&A, exports)


.env.example
requirements.txt




---

## Quick start

```command prompt
# Windows (command prompt)
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env   # then edit .env

uvicorn backend.api.main:app --reload     # API → http://127.0.0.1:8000
# open a new terminal
streamlit run ui/app.py                   # UI  → http://127.0.0.1:8501




How it works (flow)*******

1. pload a PDF/DOCX/TXT/HTML from the UI. Upload returns quickly; heavy processing runs in a background task.

2. Backend parses → chunks → embeds (batched) → upserts to the selected vector store.

3. Generates an initial summary. You can edit, save (versioned), validate, and rollback.

4. Entities (names, dates, organizations) extracted from the summary; editable as JSON.

5. Q&A: ask natural-language questions; answers show sources and a quick validation score.

6. Export: summary as .md/.pdf, entities as .json.



Endpoints (essentials)********

GET /health — shallow ping

GET /health?deep=true — vector store, uploads dir, LLM env checks

POST /documents — multipart upload (file) → returns {status: "pending"} quickly

GET /documents/{id}/summary — status + summary

POST /documents/{id}/summary — save edited summary

POST /documents/{id}/summarize — regenerate

POST /documents/{id}/summary/validate

GET /documents/{id}/entities / POST /documents/{id}/entities

GET /documents/{id}/summary/versions / POST /documents/{id}/summary/rollback?version_index=0

POST /qa — {"question": "...", "document_ids": []} → answer + sources + validation

GET /qa/history — recent Q&A items

Exports:

GET /documents/{id}/export/summary.md

GET /documents/{id}/export/summary.pdf

GET /documents/{id}/export/entities.json

MCP-style:

GET /mcp/files

GET /mcp/files/read?name=...

POST /mcp/files/write → { "name":"...", "content":"..." }

GET /mcp/search?q=...&k=5



Testing & quality

# tests + coverage
pytest -v --maxfail=2 --disable-warnings --cov=backend --cov-report=term-missing