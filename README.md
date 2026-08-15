# Hybrid Documentation Search Engine

I built this as a search engine rather than a chatbot. The bundled, openly authored programming-documentation mini-corpus makes the repository instantly reproducible. BM25 captures exact terminology, a local TF-IDF dense n-gram index captures related phrasing, Reciprocal Rank Fusion combines both rankings, query expansion handles common synonyms, and snippets highlight evidence. Category facets and pagination are exposed through FastAPI.

```text
query → expansion → BM25 ─┐
                          ├→ reciprocal rank fusion → facets → highlighted snippets
              TF-IDF ─────┘
```

```bash
docker compose up --build
```

Open `http://localhost:8000`. `/health` reports precision@3 and recall@3 for BM25, dense, and hybrid retrieval on the committed labeled query set. These are computed at runtime, not estimated. Run `pytest` for known-query, fusion, snippet, and evaluation checks.

I chose a small curated corpus for zero-friction validation; it is not yet “large.” The ingestion boundary can be extended to public Python, FastAPI, Docker, and Kubernetes documentation while respecting each source license. Limitations include TF-IDF rather than neural embeddings, a tiny synonym map, memory-only indexes, and no click feedback. Next steps are licensed corpus downloaders, sentence-transformer embeddings, typo correction, learned fusion weights, feedback logging, and larger relevance judgments.

Suggested commits: `set up search service`, `add documentation corpus`, `build BM25 index`, `add local dense index`, `implement reciprocal rank fusion`, `add query expansion`, `generate highlighted snippets`, `add facets and pagination`, `build search UI`, `add relevance evaluation`, `add Docker setup`, `write README`.

```bash
git init -b main
git add app/search.py && git commit -m "add BM25 dense retrieval and rank fusion"
git add app/main.py app/static && git commit -m "add search API facets and frontend"
git add tests requirements.txt && git commit -m "add relevance evaluation tests"
git add Dockerfile docker-compose.yml && git commit -m "add container setup"
git add README.md && git commit -m "document search quality and limitations"
gh repo create hybrid-wiki-search-engine --public --source=. --remote=origin
git push -u origin main
```

MIT licensed.
