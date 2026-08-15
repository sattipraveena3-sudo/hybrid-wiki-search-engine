from pathlib import Path
from fastapi import FastAPI,Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.search import SearchEngine,evaluate
engine=SearchEngine();app=FastAPI(title="Hybrid Documentation Search");static=Path(__file__).parent/'static';app.mount('/static',StaticFiles(directory=static),name='static')
@app.get('/')
def home():return FileResponse(static/'index.html')
@app.get('/health')
def health():return {'status':'ok','documents':len(engine.docs),'evaluation':evaluate(engine)}
@app.get('/search')
def search(q:str=Query(min_length=2),category:str|None=None,page:int=1,page_size:int=5):return {'query':q,'results':engine.search(q,category,limit=page*page_size)[(page-1)*page_size:],'facets':sorted({d['category'] for d in engine.docs})}
