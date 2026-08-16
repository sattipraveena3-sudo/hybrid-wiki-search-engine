# Hybrid Wiki Search Engine

A complete, runnable advanced information-retrieval platform that combines lexical ranking, semantic retrieval, reciprocal-rank fusion, faceted filtering, autocomplete, persistent indexing, explainable ranking, analytics, and a production-style API/UI.

`documents → SQLite → BM25 + TF-IDF + LSI + char n-grams → RRF fusion → ranked results + facets + explanations`

## Features

- Persistent SQLite-backed document index
- BM25 lexical retrieval
- Word n-gram TF-IDF ranking
- Character n-gram similarity for typo/partial-term robustness
- Latent Semantic Indexing with Truncated SVD
- Reciprocal Rank Fusion across independent rankers
- Title-overlap reranking boost
- Query expansion with curated synonyms
- Category and section facets
- Search suggestions/autocomplete
- Ranking explanations with per-ranker scores
- Document ingestion and live reindex APIs
- Query analytics and popular-query statistics
- Built-in benchmark evaluation with recall@3 and MRR@3
- Prometheus-compatible `/metrics`
- Liveness `/health` and readiness `/ready`
- Optional write API key
- Responsive browser search UI
- Swagger/OpenAPI docs
- Unit and API integration tests
- Docker / Docker Compose
- GitHub Actions compile, pytest, container build, startup and live search smoke checks

## Fastest start

```bash
git clone https://github.com/sattipraveena3-sudo/hybrid-wiki-search-engine.git
cd hybrid-wiki-search-engine
docker compose up --build
```

Open `http://localhost:8000`.

API docs: `http://localhost:8000/docs`

## Search API

```bash
curl 'http://localhost:8000/api/search?q=combine%20lexical%20semantic%20rankings&mode=hybrid&explain=true'
```

Modes:

- `hybrid` — BM25 + semantic + TF-IDF fused with RRF
- `bm25` — lexical BM25 only
- `semantic` — LSI semantic similarity + character similarity

Filters can be supplied with `category` and `section`.

## Index a document

```bash
curl -X POST http://localhost:8000/api/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"rag",
    "title":"Retrieval augmented generation",
    "category":"AI",
    "section":"Search",
    "text":"RAG retrieves relevant evidence before generation.",
    "url":"https://example.com/rag",
    "tags":["rag","retrieval"]
  }'
```

If `WRITE_API_KEY` is configured, include it with `X-API-Key`.

## Main endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/search` | Hybrid/lexical/semantic search |
| GET | `/api/suggest` | Autocomplete suggestions |
| GET | `/api/documents/{id}` | Retrieve indexed document |
| POST | `/api/documents` | Add/update document |
| POST | `/api/reindex` | Rebuild in-memory ranking structures |
| GET | `/api/stats` | Index, evaluation and query analytics |
| GET | `/health` | Liveness |
| GET | `/ready` | Storage/index readiness |
| GET | `/metrics` | Prometheus-style metrics |
| GET | `/docs` | Interactive API docs |

## Ranking architecture

The engine builds several complementary retrieval views. BM25 provides strong exact-term relevance. Word TF-IDF captures phrase and n-gram overlap. Character n-grams improve robustness for partial or slightly misspelled queries. Truncated SVD projects the sparse term matrix into a lower-dimensional latent semantic space. Reciprocal Rank Fusion combines the independent rankings without requiring directly comparable score scales. A small title-overlap boost acts as a lightweight reranker.

This design is intentionally self-contained: no external vector database or model download is required, so the full project starts reliably with one Docker command.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make run
```

In another terminal:

```bash
make smoke
```

## CI validation

Every pull request runs:

1. dependency installation
2. Python compilation
3. full pytest suite
4. Docker image build
5. real container startup
6. readiness and health probes
7. a live hybrid search with ranking explanation
8. suggestion, statistics and metrics smoke requests

## Configuration

```bash
cp .env.example .env
```

- `DATABASE_PATH` — SQLite index path
- `WRITE_API_KEY` — optional protection for document/reindex write APIs

## Production-scale extensions

The repository is fully runnable as-is. Larger deployments could swap SQLite for PostgreSQL/OpenSearch, use transformer embeddings in a vector database, add cross-encoder reranking, Wikipedia dump ingestion, distributed indexing, caching, click-feedback learning-to-rank, spell correction, multilingual analyzers, and offline nDCG/MAP evaluation.

MIT licensed.
