import chromadb
from groq import Groq
import os
import uuid

class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize Groq client for embeddings
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.groq_client = Groq(api_key=api_key)

    def _get_embedding(self, text: str) -> list:
        """Get embedding using Groq's embeddings API."""
        response = self.groq_client.embeddings.create(
            model="embed-multilingual-v3-mqa",
            input=text,
            timeout=30  # 30 second timeout to prevent hanging
        )
        return response.data[0].embedding

    def add_document(self, text: str, metadata: dict):
        embedding = self._get_embedding(text)

        # FIX: Always unique ID — prevents overwrite
        doc_id = str(uuid.uuid4())

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def search(self, query: str, n_results: int = 10):
        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results

    def clear_collection(self):
        results = self.collection.get()
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])

    def list_sources(self):
        results = self.collection.get(include=["metadatas"])
        source_counts = {}
        if results and "metadatas" in results:
            for meta in results["metadatas"]:
                source = meta.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "sources": list(source_counts.keys()),
            "counts": source_counts
        }


