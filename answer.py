import chromadb 
from sentence_transformers import SentenceTransformer
from llm import ask_llm

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")

SYSTEM_PROMPT = (
    "Answer only from the context given."
    "If the context does not contain the answer, you say you don't know."
)

def retrieve(question, n_results=3):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding], n_results=n_results
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            {"text": doc, "page": meta["page"], "source": meta["source"], "distance":dist}
        )

    return chunks

def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join([c["text"] for c in chunks])
    user_prompt = f"Context: \n{context}\n\nQuestion: {question}"
    return ask_llm(SYSTEM_PROMPT, user_prompt), chunks

if  __name__ == "__main__":
    question = "What is the capital of France?"
    reply, chunks = answer(question)
    print(reply)
    print("Pages used: ", [c["page"] for c in chunks])