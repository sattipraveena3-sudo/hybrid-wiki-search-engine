import html
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


DEFAULT_DOCS = [
    {"id":"py-async","title":"Python asynchronous programming","category":"Python","section":"Concurrency","text":"asyncio provides an event loop, coroutines, tasks, async and await syntax for concurrent network programs. It is useful for I/O-bound services and cooperative multitasking.","url":"https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)","tags":["python","asyncio","concurrency"]},
    {"id":"py-types","title":"Python type hints","category":"Python","section":"Language","text":"Type annotations document function parameters and return values. Static analyzers such as mypy can detect incompatible types before runtime.","url":"https://en.wikipedia.org/wiki/Python_(programming_language)","tags":["python","typing","mypy"]},
    {"id":"docker-vol","title":"Docker volumes","category":"Containers","section":"Storage","text":"Docker volumes persist container data outside the writable layer. Volumes can be mounted into services with Compose and survive container recreation.","url":"https://en.wikipedia.org/wiki/Docker_(software)","tags":["docker","storage","volume"]},
    {"id":"docker-net","title":"Docker networking","category":"Containers","section":"Networking","text":"Docker Compose creates a default network where services discover each other by service name. User-defined networks isolate application traffic.","url":"https://en.wikipedia.org/wiki/Docker_(software)","tags":["docker","networking","compose"]},
    {"id":"fastapi-deps","title":"FastAPI dependencies","category":"Web","section":"FastAPI","text":"FastAPI dependency injection with Depends can share database sessions, authentication state, configuration and request services across endpoints.","url":"https://en.wikipedia.org/wiki/FastAPI","tags":["fastapi","dependency injection","api"]},
    {"id":"fastapi-validation","title":"FastAPI request validation","category":"Web","section":"FastAPI","text":"Pydantic models validate JSON request bodies and FastAPI generates OpenAPI schemas automatically. Invalid payloads produce structured validation errors.","url":"https://en.wikipedia.org/wiki/FastAPI","tags":["fastapi","pydantic","validation"]},
    {"id":"git-branch","title":"Git branches","category":"Developer Tools","section":"Git","text":"Git branches are movable references to commits. Merge and rebase integrate divergent history using different history-preservation strategies.","url":"https://en.wikipedia.org/wiki/Git","tags":["git","branch","merge","rebase"]},
    {"id":"git-reset","title":"Git reset and restore","category":"Developer Tools","section":"Git","text":"git restore changes files while git reset moves references and may update the index or working tree depending on the selected mode.","url":"https://en.wikipedia.org/wiki/Git","tags":["git","reset","restore"]},
    {"id":"sql-index","title":"SQL indexes","category":"Databases","section":"Performance","text":"B-tree indexes accelerate selective lookups, joins and ordering but add write amplification and storage overhead. Composite indexes depend on column order.","url":"https://en.wikipedia.org/wiki/Database_index","tags":["sql","index","database","performance"]},
    {"id":"sql-window","title":"SQL window functions","category":"Databases","section":"Queries","text":"SQL window functions calculate ranks, running totals and moving averages without collapsing rows. OVER and PARTITION BY define the analytical window.","url":"https://en.wikipedia.org/wiki/Window_function_(SQL)","tags":["sql","window function","analytics"]},
    {"id":"k8s-deploy","title":"Kubernetes deployments","category":"Orchestration","section":"Workloads","text":"Kubernetes Deployments manage ReplicaSets, rolling updates, desired replicas and declarative Pod rollout. Rollbacks restore earlier revisions.","url":"https://en.wikipedia.org/wiki/Kubernetes","tags":["kubernetes","deployment","pods"]},
    {"id":"k8s-service","title":"Kubernetes services","category":"Orchestration","section":"Networking","text":"Kubernetes Services expose stable virtual addresses for changing sets of Pods selected by labels. ClusterIP, NodePort and LoadBalancer are common service types.","url":"https://en.wikipedia.org/wiki/Kubernetes","tags":["kubernetes","service","networking"]},
    {"id":"ml-transformer","title":"Transformer architecture","category":"Machine Learning","section":"NLP","text":"Transformers use self-attention instead of recurrent connections to model token relationships. Multi-head attention and positional encodings enable parallel sequence processing.","url":"https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)","tags":["transformer","attention","nlp"]},
    {"id":"ml-vector","title":"Vector embeddings","category":"Machine Learning","section":"Information Retrieval","text":"Vector embeddings map items into continuous spaces where semantically related concepts have similar representations. Cosine similarity is often used for nearest-neighbor retrieval.","url":"https://en.wikipedia.org/wiki/Word_embedding","tags":["embedding","vector","semantic search"]},
    {"id":"ir-bm25","title":"BM25 ranking","category":"Information Retrieval","section":"Lexical Search","text":"BM25 is a probabilistic ranking function based on term frequency, inverse document frequency and document length normalization. It is widely used for lexical information retrieval.","url":"https://en.wikipedia.org/wiki/Okapi_BM25","tags":["bm25","ranking","lexical search"]},
    {"id":"ir-rrf","title":"Reciprocal rank fusion","category":"Information Retrieval","section":"Hybrid Search","text":"Reciprocal rank fusion combines multiple ranked lists using inverse rank contributions. RRF is robust when component retrieval systems produce scores on incomparable scales.","url":"https://en.wikipedia.org/wiki/Rank_fusion","tags":["rrf","fusion","hybrid search"]},
]

SYNONYMS = {
    "container": ["docker"], "persist": ["volume", "storage"], "api": ["fastapi", "web"],
    "k8s": ["kubernetes"], "database": ["sql"], "semantic": ["embedding", "vector"],
    "ranking": ["retrieval", "search"], "async": ["asynchronous", "asyncio"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def expand_query(query: str) -> str:
    terms = tokenize(query)
    expanded = terms + [syn for term in terms for syn in SYNONYMS.get(term, [])]
    return " ".join(expanded)


def rrf(*rankings: list[int], k: int = 60) -> tuple[list[int], dict[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    ordered = [idx for idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    return ordered, scores


def snippet(text: str, query: str, width: int = 240) -> str:
    terms = tokenize(query)
    lower = text.lower()
    starts = [lower.find(term) for term in terms if lower.find(term) >= 0]
    start = max(0, (min(starts) if starts else 0) - 48)
    value = html.escape(text[start:start + width])
    for term in sorted(set(terms), key=len, reverse=True):
        value = re.sub(f"(?i)({re.escape(term)})", r"<mark>\1</mark>", value)
    return value


@dataclass
class SearchStats:
    documents: int
    categories: int
    vocabulary: int
    queries_logged: int
    indexed_at: float


class DocumentStore:
    def __init__(self, path: str | Path = "data/search.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.setup()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self) -> None:
        with self.connect() as conn:
            conn.execute("create table if not exists documents(id text primary key,title text not null,category text not null,section text not null,text text not null,url text,tags text,updated real not null)")
            conn.execute("create table if not exists queries(id integer primary key,query text not null,mode text not null,result_count integer not null,latency_ms real not null,created real not null)")
            conn.execute("create index if not exists idx_documents_category on documents(category)")
            conn.execute("create index if not exists idx_queries_created on queries(created desc)")

    def count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("select count(*) from documents").fetchone()[0])

    def replace_all(self, docs: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("delete from documents")
            conn.executemany(
                "insert into documents(id,title,category,section,text,url,tags,updated) values(?,?,?,?,?,?,?,?)",
                [(d["id"], d["title"], d["category"], d.get("section", "General"), d["text"], d.get("url", ""), "|".join(d.get("tags", [])), time.time()) for d in docs],
            )

    def upsert(self, doc: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert into documents(id,title,category,section,text,url,tags,updated) values(?,?,?,?,?,?,?,?) on conflict(id) do update set title=excluded.title,category=excluded.category,section=excluded.section,text=excluded.text,url=excluded.url,tags=excluded.tags,updated=excluded.updated",
                (doc["id"], doc["title"], doc["category"], doc.get("section", "General"), doc["text"], doc.get("url", ""), "|".join(doc.get("tags", [])), time.time()),
            )

    def all(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from documents order by id").fetchall()
        return [{**dict(r), "tags": [x for x in r["tags"].split("|") if x]} for r in rows]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("select * from documents where id=?", (doc_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["tags"] = [x for x in data["tags"].split("|") if x]
        return data

    def log_query(self, query: str, mode: str, result_count: int, latency_ms: float) -> None:
        with self.connect() as conn:
            conn.execute("insert into queries(query,mode,result_count,latency_ms,created) values(?,?,?,?,?)", (query, mode, result_count, latency_ms, time.time()))

    def query_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("select count(*) from queries").fetchone()[0])

    def popular_queries(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select query,count(*) as count,avg(latency_ms) as avg_latency_ms from queries group by query order by count desc,id desc limit ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


class SearchEngine:
    def __init__(self, docs: list[dict[str, Any]] | None = None, store: DocumentStore | None = None):
        self.store = store or DocumentStore()
        if docs is not None:
            self.store.replace_all(docs)
        elif self.store.count() == 0:
            self.store.replace_all(DEFAULT_DOCS)
        self.indexed_at = 0.0
        self.rebuild()

    def rebuild(self) -> None:
        self.docs = self.store.all()
        if not self.docs:
            self.tokens = []
            self.bm25 = None
            self.word_vectorizer = None
            self.word_matrix = None
            self.char_vectorizer = None
            self.char_matrix = None
            self.semantic_matrix = None
            self.semantic_transform = None
            self.indexed_at = time.time()
            return
        weighted = [f"{d['title']} {d['title']} {d['section']} {' '.join(d['tags'])} {d['text']}" for d in self.docs]
        self.tokens = [tokenize(text) for text in weighted]
        self.bm25 = BM25Okapi(self.tokens)
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.word_matrix = self.word_vectorizer.fit_transform(weighted)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=5000)
        self.char_matrix = self.char_vectorizer.fit_transform(weighted)
        components = min(64, max(2, min(self.word_matrix.shape) - 1))
        if self.word_matrix.shape[0] >= 3 and self.word_matrix.shape[1] >= 3:
            svd = TruncatedSVD(n_components=components, random_state=7)
            normalizer = Normalizer(copy=False)
            dense = normalizer.fit_transform(svd.fit_transform(self.word_matrix))
            self.semantic_transform = (svd, normalizer)
            self.semantic_matrix = dense
        else:
            self.semantic_transform = None
            self.semantic_matrix = self.word_matrix.toarray()
        self.indexed_at = time.time()

    def add_document(self, doc: dict[str, Any]) -> None:
        self.store.upsert(doc)
        self.rebuild()

    def _orders(self, query: str) -> dict[str, tuple[list[int], np.ndarray]]:
        expanded = expand_query(query)
        bm_scores = np.asarray(self.bm25.get_scores(tokenize(expanded)), dtype=float)
        word_scores = cosine_similarity(self.word_vectorizer.transform([expanded]), self.word_matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([expanded]), self.char_matrix)[0]
        if self.semantic_transform:
            svd, normalizer = self.semantic_transform
            q_sem = normalizer.transform(svd.transform(self.word_vectorizer.transform([expanded])))
            semantic_scores = cosine_similarity(q_sem, self.semantic_matrix)[0]
        else:
            semantic_scores = word_scores
        semantic_scores = 0.7 * semantic_scores + 0.3 * char_scores
        return {
            "bm25": (list(np.argsort(bm_scores)[::-1]), bm_scores),
            "semantic": (list(np.argsort(semantic_scores)[::-1]), semantic_scores),
            "tfidf": (list(np.argsort(word_scores)[::-1]), word_scores),
        }

    def search(self, query: str, category: str | None = None, section: str | None = None, limit: int = 10, mode: str = "hybrid", explain: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        query = query.strip()
        if not query or not self.docs:
            return {"query": query, "mode": mode, "results": [], "total": 0, "facets": {}, "latency_ms": 0.0}
        orders = self._orders(query)
        if mode == "bm25":
            order = orders["bm25"][0]
            fusion_scores = {i: float(orders["bm25"][1][i]) for i in order}
        elif mode in {"semantic", "dense"}:
            order = orders["semantic"][0]
            fusion_scores = {i: float(orders["semantic"][1][i]) for i in order}
        else:
            order, fusion_scores = rrf(orders["bm25"][0], orders["semantic"][0], orders["tfidf"][0])

        query_terms = set(tokenize(query))
        rows = []
        for idx in order:
            doc = self.docs[idx]
            if category and doc["category"] != category:
                continue
            if section and doc["section"] != section:
                continue
            title_terms = set(tokenize(doc["title"]))
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            base = float(fusion_scores.get(idx, 0.0))
            if mode == "hybrid":
                score = base + 0.02 * title_overlap
            else:
                score = base
            result = {
                "id": doc["id"], "title": doc["title"], "category": doc["category"], "section": doc["section"],
                "url": doc.get("url", ""), "tags": doc.get("tags", []), "snippet": snippet(doc["text"], query),
                "score": round(score, 6),
            }
            if explain:
                result["explain"] = {
                    "bm25": round(float(orders["bm25"][1][idx]), 6),
                    "semantic": round(float(orders["semantic"][1][idx]), 6),
                    "tfidf": round(float(orders["tfidf"][1][idx]), 6),
                    "title_overlap": round(title_overlap, 4),
                    "fusion": round(base, 6),
                }
            rows.append(result)
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[: max(1, min(limit, 50))]
        latency = (time.perf_counter() - started) * 1000
        self.store.log_query(query, mode, len(rows), latency)
        category_counts = Counter(d["category"] for d in self.docs)
        section_counts = Counter(d["section"] for d in self.docs)
        return {
            "query": query, "mode": mode, "results": rows, "total": len(rows),
            "facets": {"categories": dict(sorted(category_counts.items())), "sections": dict(sorted(section_counts.items()))},
            "latency_ms": round(latency, 3),
        }

    def suggest(self, prefix: str, limit: int = 8) -> list[str]:
        prefix = prefix.strip().lower()
        values = []
        for doc in self.docs:
            values.extend([doc["title"], doc["category"], doc["section"], *doc.get("tags", [])])
        counts = Counter(v for v in values if prefix in v.lower())
        return [value for value, _ in counts.most_common(limit)]

    def stats(self) -> SearchStats:
        vocab = len(self.word_vectorizer.vocabulary_) if self.word_vectorizer else 0
        return SearchStats(len(self.docs), len({d["category"] for d in self.docs}), vocab, self.store.query_count(), self.indexed_at)


def evaluate(engine: SearchEngine) -> dict[str, dict[str, float]]:
    labels = {
        "persist container data": {"docker-vol"},
        "validate API JSON": {"fastapi-validation"},
        "rolling pod updates": {"k8s-deploy"},
        "running total SQL": {"sql-window"},
        "combine lexical semantic rankings": {"ir-rrf"},
        "self attention NLP": {"ml-transformer"},
    }
    output: dict[str, dict[str, float]] = {}
    for mode in ["bm25", "semantic", "hybrid"]:
        recalls = []
        reciprocal_ranks = []
        for query, relevant in labels.items():
            results = engine.search(query, limit=3, mode=mode)["results"]
            ids = [r["id"] for r in results]
            recalls.append(len(set(ids) & relevant) / len(relevant))
            rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in relevant), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        output[mode] = {"recall@3": round(float(np.mean(recalls)), 4), "mrr@3": round(float(np.mean(reciprocal_ranks)), 4)}
    return output
