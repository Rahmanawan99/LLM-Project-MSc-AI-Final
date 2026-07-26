from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


text = [
    "Humans cannot breathe underwater naturally."
]


embeddings = embedding_model.encode(text)


print(embeddings.shape)