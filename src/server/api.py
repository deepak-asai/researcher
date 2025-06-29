from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .researcher import Researcher

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UrlsRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    query: str

@app.post("/add-urls")
async def add_urls(request: UrlsRequest):
    researcher = Researcher()
    researcher.add_urls(request.urls)
    return {"status": "success", "received_urls": request.urls}

@app.post("/query")
async def query(request: QueryRequest):
    researcher = Researcher()
    results = researcher.query(request.query)

    return {"status": "success", "query": request.query, "result": results}