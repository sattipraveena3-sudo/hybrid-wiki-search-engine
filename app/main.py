import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.search import DocumentStore, SearchEngine, evaluate


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=120)
    section: str = Field(default="General", max_length=120)
    text: str = Field(min_length=1)
    url: str = ""
    tags: list[str] = []


DB_PATH = os.getenv("DATABASE_PATH", "data/search.db")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")
store = DocumentStore(DB_PATH)
engine = SearchEngine(store=store)

app = FastAPI(
    title="Hybrid Wiki Search Engine",
    version="2.0.0",
    description="Advanced lexical + semantic retrieval with BM25, latent semantic search, reciprocal-rank fusion, facets, suggestions, analytics and explainable ranking.",
)
static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")


def require_key(value: str | None) -> None:
    if WRITE_API_KEY and value != WRITE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(static / "index.html")


@app.get("/health")
def health():
    stats = asdict(engine.stats())
    return {"status": "ok", "index": stats}


@app.get("/ready")
def ready():
    try:
        store.count()
        return {"status": "ready", "documents": len(engine.docs)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/search")
def search(
    q: str = Query(min_length=2),
    mode: Literal["hybrid", "bm25", "semantic", "dense"] = "hybrid",
    category: str | None = None,
    section: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(8, ge=1, le=25),
    explain: bool = False,
):
    full = engine.search(q, category=category, section=section, limit=min(page * page_size, 50), mode=mode, explain=explain)
    start = (page - 1) * page_size
    full["results"] = full["results"][start:start + page_size]
    full["page"] = page
    full["page_size"] = page_size
    return full


@app.get("/api/documents/{doc_id}")
def document(doc_id: str):
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@app.post("/api/documents", status_code=201)
def index_document(payload: DocumentIn, x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    engine.add_document(payload.model_dump())
    return {"indexed": payload.id, "documents": len(engine.docs)}


@app.post("/api/reindex")
def reindex(x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    engine.rebuild()
    return {"status": "reindexed", "documents": len(engine.docs), "indexed_at": engine.indexed_at}


@app.get("/api/suggest")
def suggest(q: str = Query(min_length=1), limit: int = Query(8, ge=1, le=20)):
    return {"query": q, "suggestions": engine.suggest(q, limit)}


@app.get("/api/stats")
def stats():
    return {
        "index": asdict(engine.stats()),
        "evaluation": evaluate(engine),
        "popular_queries": store.popular_queries(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    s = engine.stats()
    lines = [
        "# HELP wiki_search_documents Number of indexed documents",
        "# TYPE wiki_search_documents gauge",
        f"wiki_search_documents {s.documents}",
        "# HELP wiki_search_queries_total Number of logged search queries",
        "# TYPE wiki_search_queries_total counter",
        f"wiki_search_queries_total {s.queries_logged}",
        "# HELP wiki_search_vocabulary Indexed TF-IDF vocabulary size",
        "# TYPE wiki_search_vocabulary gauge",
        f"wiki_search_vocabulary {s.vocabulary}",
    ]
    return "\n".join(lines) + "\n"


# Backward-compatible endpoint.
@app.get("/search", include_in_schema=False)
def legacy_search(q: str = Query(min_length=2), category: str | None = None, page: int = 1, page_size: int = 5):
    return search(q=q, category=category, page=page, page_size=page_size)
