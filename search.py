import chromadb 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")

question = "What is the capital of France?"
question_embedding = model.encode(question).tolist()

results = collection.query(query_embeddings=[question_embedding], n_results=3)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print("page", meta["page"], "| distance", round(dist, 3))
    print(doc[:150])
    print("-----")
