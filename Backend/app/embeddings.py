# backend/app/embeddings.py
from groq import Groq
import os
import numpy as np

class Embeddings:
    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize embeddings with Groq by default.
        
        Args:
            model_name: The embedding model to use. Defaults to llama-3.1-8b-instant
            provider: 'groq' or None (defaults to groq)
        """
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "llama-3.1-8b-instant")
        
        self.model_name = model_name
        self.provider = provider or "groq"
        
        # Configure Groq API
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def embed_texts(self, texts: list):
        """
        texts: list[str] -> returns list[list[float]]
        """
        if self.provider == "groq" or self.provider is None:
            return self._embed_with_groq(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _embed_with_groq(self, texts: list):
        """
        Embed texts using Groq's embedding API.
        """
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.model_name,
                input=text
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        
        return np.array(embeddings)
