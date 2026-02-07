# backend/app/embeddings.py
from sentence_transformers import SentenceTransformer
import os

class Embeddings:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list):
        """
        texts: list[str] -> returns list[list[float]]
        """
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
