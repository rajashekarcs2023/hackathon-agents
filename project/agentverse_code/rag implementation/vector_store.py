import os
import requests
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings

API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_HOST = os.environ["INDEX_HOST"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    dimensions=1536,
)


def query_vector_db(query: str, top_k: int = 8) -> List[Dict]:
    vec = embeddings.embed_query(query)

    headers = {
        "Api-Key": API_KEY,
        "Content-Type": "application/json",
        "X-Pinecone-API-Version": "2025-04"
    }
    payload = {
        "namespace": "__default__",
        "vector": vec,
        "topK": 8,
        "includeMetadata": "true"
    }

    resp = requests.post(f"https://{INDEX_HOST}/query", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json().get("matches")
