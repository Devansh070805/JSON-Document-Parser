from sentence_transformers import SentenceTransformer
import pinecone
from core.config import key

# Initialize Pinecone (for v2.2.4)
pinecone.init(api_key=key.pinecone_api_key, environment=key.pinecone_env)

# Initialize the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create index if not exists
if key.pinecone_index not in pinecone.list_indexes():
    pinecone.create_index(
        name=key.pinecone_index,
        dimension=384,
        metric="cosine"
    )

# Connect to index
index = pinecone.Index(key.pinecone_index)

def embed_and_store_chunks(chunks: list[str], source_url: str):
    vectors = model.encode(chunks).tolist()
    payload = [
        {
            "id": f"{source_url}_{i}",
            "values": vec,
            "metadata": {"text": chunk}
        }
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    index.upsert(payload)
