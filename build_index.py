import chromadb
from sentence_transformers import SentenceTransformer
from load_pdf import load_pdf 
from chunker import chunk_pages

model = SentenceTransformer("all-MiniLM-L6-v2")

pages = load_pdf("data/attention_is_all_you_need.pdf")
chunks = chunk_pages(pages, "attention_is_all_you_need.pdf")

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks]
)

print(collection.count())