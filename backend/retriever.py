import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("/Users/dhanushrajaa/Desktop/untitled folder/backend/medquad_index.faiss")

# Load metadata
with open("/Users/dhanushrajaa/Desktop/untitled folder/backend/medquad_qa.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def search(query: str, top_k: int = 5):
    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    # Search FAISS
    D, I = index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx]
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "source": item["source"],
            "url": item["url"],
            "score": float(score)
        })
    return results