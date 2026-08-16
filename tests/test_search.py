from app.search import DocumentStore, SearchEngine, evaluate, rrf, snippet


def test_known_hybrid_queries(tmp_path):
    engine = SearchEngine(store=DocumentStore(tmp_path / "search.db"))
    assert engine.search("persist container data")["results"][0]["id"] == "docker-vol"
    assert engine.search("running total rows")["results"][0]["id"] == "sql-window"
    assert engine.search("combine lexical semantic rankings")["results"][0]["id"] == "ir-rrf"


def test_fusion():
    order, scores = rrf([1, 2, 3], [2, 1, 4])
    assert order[:2] == [1, 2]
    assert scores[1] > 0


def test_snippet_highlights():
    assert "<mark>volume</mark>" in snippet("Docker volume storage", "volume")


def test_filters_suggestions_and_explain(tmp_path):
    engine = SearchEngine(store=DocumentStore(tmp_path / "filters.db"))
    result = engine.search("network", category="Containers", explain=True)
    assert result["results"]
    assert all(row["category"] == "Containers" for row in result["results"])
    assert "bm25" in result["results"][0]["explain"]
    assert engine.suggest("kuber")


def test_persistent_document_ingestion(tmp_path):
    store = DocumentStore(tmp_path / "persist.db")
    engine = SearchEngine(store=store)
    engine.add_document({"id":"custom-rag","title":"Retrieval augmented generation","category":"AI","section":"RAG","text":"RAG retrieves evidence before language model generation.","url":"","tags":["rag","retrieval"]})
    assert store.get("custom-rag")["title"] == "Retrieval augmented generation"
    assert engine.search("retrieve evidence generation")["results"][0]["id"] == "custom-rag"


def test_evaluation_quality(tmp_path):
    metrics = evaluate(SearchEngine(store=DocumentStore(tmp_path / "eval.db")))
    assert metrics["hybrid"]["recall@3"] >= 0.8
    assert metrics["hybrid"]["mrr@3"] >= 0.7
