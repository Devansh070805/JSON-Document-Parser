import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model (you can replace this with your own)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# File to persist index and metadata
INDEX_FILE = "faiss_index.index"
META_FILE = "faiss_metadata.pkl"

# Initialize FAISS index (dimension must match embedding size)
embedding_dim = 384  # for 'all-MiniLM-L6-v2'
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []

def add_texts(texts: list[str]):
    """Add list of texts to the FAISS index"""
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index.add(np.array(embeddings))
    metadata.extend(texts)

    # Save to disk
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

def search(query: str, k: int = 5):
    """Search for similar texts"""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_embedding), k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "text": metadata[idx],
                "distance": float(distances[0][i])
            })
    return results

# Example usage
if __name__ == "__main__":
    add_texts(["AI is the future", "Python is great", "Cats are cute"])
    res = search("What about artificial intelligence?")
    for r in res:
        print(r)
