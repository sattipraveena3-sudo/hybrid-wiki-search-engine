from app.search import SearchEngine,evaluate,rrf,snippet
def test_known_queries():
    e=SearchEngine();assert e.search('persist container data')[0]['id']=='docker-vol';assert e.search('running total rows')[0]['id']=='sql-window'
def test_fusion(): assert rrf([1,2,3],[2,1,4])[:2]==[1,2]
def test_snippet(): assert '<mark>volume</mark>' in snippet('Docker volume storage','volume')
def test_evaluation():
    m=evaluate(SearchEngine());assert m['hybrid']['recall@3']>=.75
