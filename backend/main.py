from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import generate_answer

app = FastAPI(title="RAG-Medibot API (HuggingFace)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/test_llm")
async def query_api(req: QueryRequest):
    response = generate_answer(req.query, req.top_k)
    return {
        "query": req.query,
        "answer": response["answer"],
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}