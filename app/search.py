import html,re
from dataclasses import dataclass,asdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS=[
{"id":"py-async","title":"Python asynchronous programming","category":"Python","section":"Concurrency","text":"asyncio provides an event loop, coroutines, tasks, and async await syntax for concurrent network programs."},
{"id":"py-types","title":"Python type hints","category":"Python","section":"Language","text":"Type annotations document function parameters and return values. Mypy can perform static type checking."},
{"id":"docker-vol","title":"Docker volumes","category":"Containers","section":"Storage","text":"Volumes persist container data outside the writable layer and can be mounted into services with Compose."},
{"id":"docker-net","title":"Docker networking","category":"Containers","section":"Networking","text":"Compose creates a default network where services discover each other by service name."},
{"id":"fastapi-deps","title":"FastAPI dependencies","category":"Web","section":"FastAPI","text":"Dependency injection with Depends shares database sessions authentication and request services."},
{"id":"fastapi-validation","title":"FastAPI request validation","category":"Web","section":"FastAPI","text":"Pydantic models validate JSON request bodies and generate OpenAPI schemas automatically."},
{"id":"git-branch","title":"Git branches","category":"Developer Tools","section":"Git","text":"Branches are movable references to commits. Merge and rebase integrate divergent history."},
{"id":"git-reset","title":"Git reset and restore","category":"Developer Tools","section":"Git","text":"Restore changes files while reset moves references and may update the index or working tree."},
{"id":"sql-index","title":"SQL indexes","category":"Databases","section":"Performance","text":"B-tree indexes accelerate selective lookups and ordering but add write and storage overhead."},
{"id":"sql-window","title":"SQL window functions","category":"Databases","section":"Queries","text":"Window functions calculate ranks running totals and moving averages without collapsing rows."},
{"id":"k8s-deploy","title":"Kubernetes deployments","category":"Orchestration","section":"Workloads","text":"Deployments manage replica sets rolling updates desired replicas and declarative pod rollout."},
{"id":"k8s-service","title":"Kubernetes services","category":"Orchestration","section":"Networking","text":"Services expose stable virtual addresses for changing sets of pods selected by labels."},
]
SYNONYMS={"container":["docker"],"persist":["volume","storage"],"api":["fastapi","web"],"k8s":["kubernetes"],"database":["sql"]}
def tokenize(x):return re.findall(r"[a-z0-9]+",x.lower())
def expand(q):
    terms=tokenize(q);return " ".join(terms+[x for t in terms for x in SYNONYMS.get(t,[])])
def rrf(*rankings,k=60):
    scores={}
    for ranking in rankings:
        for rank,idx in enumerate(ranking,1):scores[idx]=scores.get(idx,0)+1/(k+rank)
    return [idx for idx,_ in sorted(scores.items(),key=lambda x:x[1],reverse=True)]
def snippet(text,query,width=180):
    terms=tokenize(query); lower=text.lower(); starts=[lower.find(t) for t in terms if lower.find(t)>=0]; start=max(0,(min(starts) if starts else 0)-30); value=html.escape(text[start:start+width])
    for term in sorted(set(terms),key=len,reverse=True):value=re.sub(f"(?i)({re.escape(term)})",r"<mark>\1</mark>",value)
    return value
class SearchEngine:
    def __init__(self,docs=CORPUS):
        self.docs=docs; self.tokens=[tokenize(d['title']+' '+d['text']) for d in docs]; self.bm=BM25Okapi(self.tokens); self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True); self.matrix=self.vectorizer.fit_transform([d['title']+' '+d['text'] for d in docs])
    def search(self,q,category=None,limit=10,mode='hybrid'):
        query=expand(q); bm_order=list(self.bm.get_scores(tokenize(query)).argsort()[::-1]); dense_order=list(cosine_similarity(self.vectorizer.transform([query]),self.matrix)[0].argsort()[::-1]); order=bm_order if mode=='bm25' else dense_order if mode=='dense' else rrf(bm_order,dense_order)
        results=[]
        for idx in order:
            d=self.docs[idx]
            if category and d['category']!=category:continue
            results.append(d|{'snippet':snippet(d['text'],q)})
            if len(results)>=limit:break
        return results
def evaluate(engine):
    labels={"persist container data":{"docker-vol"},"validate API JSON":{"fastapi-validation"},"rolling pod updates":{"k8s-deploy"},"running total SQL":{"sql-window"}}
    out={}
    for mode in ['bm25','dense','hybrid']:
        hits=[]
        for q,relevant in labels.items():hits.append(len({x['id'] for x in engine.search(q,limit=3,mode=mode)}&relevant)/len(relevant))
        out[mode]={'recall@3':sum(hits)/len(hits),'precision@3':sum(h/3 for h in hits)/len(hits)}
    return out
