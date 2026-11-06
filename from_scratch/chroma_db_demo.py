import chromadb 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The canteen serves masala dosa and vada pav.",
    "Students love eating samosas in the cafeteria.",
    "Electrons move through a conductor under a potential difference."
]

client = chromadb.Client()
collection = client.create_collection("demo")

collection.add(
    ids=["a", "b", "c"],
    documents=sentences,
    embeddings=model.encode(sentences).tolist(),
    metadatas=[{"page": 1}, {"page": 2}, {"page": 3}]
)

question = "Where can I get south indian food?"
results = collection.query(
    query_embeddings=model.encode([question]).tolist(),
    n_results=2
)

print(results)