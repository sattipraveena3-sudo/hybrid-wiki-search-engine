from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_ready():
    assert client.get('/health').status_code == 200
    assert client.get('/ready').json()['status'] == 'ready'


def test_hybrid_search_api():
    response = client.get('/api/search', params={'q':'rolling pod updates','mode':'hybrid','explain':'true'})
    assert response.status_code == 200
    body = response.json()
    assert body['results']
    assert body['results'][0]['id'] == 'k8s-deploy'
    assert 'explain' in body['results'][0]


def test_suggestions_stats_and_metrics():
    assert client.get('/api/suggest', params={'q':'dock'}).json()['suggestions']
    assert client.get('/api/stats').json()['index']['documents'] >= 16
    metrics = client.get('/metrics')
    assert metrics.status_code == 200
    assert 'wiki_search_documents' in metrics.text


def test_document_ingestion():
    payload = {'id':'api-doc','title':'Hybrid retrieval pipeline','category':'Information Retrieval','section':'Architecture','text':'Hybrid retrieval combines lexical and semantic rankers with reciprocal rank fusion.','url':'','tags':['hybrid','search']}
    assert client.post('/api/documents', json=payload).status_code == 201
    result = client.get('/api/search', params={'q':'lexical semantic rankers'}).json()['results']
    assert any(row['id'] == 'api-doc' for row in result)
