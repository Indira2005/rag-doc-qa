import chromadb 
from sentence_transformers import SentenceTransformer
from llm import ask_llm

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")

SYSTEM_PROMPT = (
    "Answer only from the context given. "
    "Each context passage is numbered like [1], [2]. "
    "After every claim in your answer, cite the passage number(s) that support it, "
    "like [1] or [1][3]. "
    "Write citations with plain square brackets exactly like [1], never any other bracket style. "
    "If the context does not contain the answer, say you don't know and cite nothing."
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

DISTANCE_THRESHOLD = 0.76

def answer(question):
    chunks = retrieve(question)
    if chunks[0]["distance"] > DISTANCE_THRESHOLD:
        return "I couldn't find this in the document.", []
    context = build_context(chunks)
    user_prompt = f"Context: \n{context}\n\nQuestion: {question}"
    return ask_llm(SYSTEM_PROMPT, user_prompt), chunks

def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (page {c['page']})\n{c['text']}")
    return "\n\n".join(parts) 

if  __name__ == "__main__":
    question = "Why do they divide by the square root of d_k?"
    reply, chunks = answer(question)
    print(reply)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] attention_is_all_you_need.pdf, {c['page']}")
    
    # good = [
    #     "What is multi-head attention?",
    #     "What does positional encoding mean?",
    #     "What does the softmax function do?",
    #     "What exactly does the transformer model compose of?",
    #     "What were models based on before the transformer architecture was proposed?",
    #     "How has transformer changed ai models?"
    # ]
    # bad = [
    #     "What is the capital of France?",
    #     "Who was the first guy to reach the moon?",
    #     "What is ARM cortex?",
    #     "What does an IPO mean? What happens when a company announces IPO?",
    #     "Does animation involve graphic programming?",
    #     "Where can I buy the best formal clothes?"
    # ]

    # for label, group in [("GOOD", good), ("BAD", bad)]:
    #     for q in group:
    #         print(label, round(retrieve(q)[0]["distance"], 3), q)