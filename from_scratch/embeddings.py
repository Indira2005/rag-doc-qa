from sentence_transformers import SentenceTransformer, util 

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The king and the queen both were very kind.",
    "The manager was very rude and unhelpful.",
    "Electrons move through a conductor under a potential difference."
]

question = "How does current flow in a wire?"

embeddings = model.encode(sentences)
question_embedding = model.encode(question)

print(embeddings.shape)
print(embeddings[0][:5])
print(util.cos_sim(question_embedding, embeddings))