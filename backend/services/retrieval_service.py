import pinecone
from sentence_transformers import SentenceTransformer
from core.config import key

model = SentenceTransformer("all-MiniLM-L6-v2")
index = pinecone.Index(key.pinecone_index)

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
