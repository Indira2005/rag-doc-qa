from llm import ask_llm 

context = "My favourite color is blue. I like the sky and the ocean."
question = "Do I like the ocean?"

answer = ask_llm(
    "Answer only from the context given. If the context does not contain the answer, say you don't know.",
    f"Context:\n{context}\n\nQuestion: {question}"
)

print(answer)