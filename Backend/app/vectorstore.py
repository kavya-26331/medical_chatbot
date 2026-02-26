import chromadb
from app.embeddings import Embeddings
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(self):
        logger.info("Initializing VectorStore...")
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        logger.info(f"ChromaDB path: {chroma_path}")
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection initialized")
        
        # Initialize local embeddings model (SentenceTransformer)
        logger.info("Initializing local embeddings model...")
        self.embeddings_model = Embeddings()
        logger.info("Local embeddings model initialized successfully")

    def _get_embedding(self, text: str) -> list:
        """Get embedding using local SentenceTransformer model."""
        logger.info(f"Getting embedding for text of length: {len(text)}")
        try:
            embedding = self.embeddings_model.embed_texts([text])[0]
            logger.info("Embedding received successfully")
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {type(e).__name__}: {str(e)}")
            raise

    def add_document(self, text: str, metadata: dict):
        logger.info(f"Adding document, text length: {len(text)}, metadata: {metadata}")
        embedding = self._get_embedding(text)

        # FIX: Always unique ID — prevents overwrite
        doc_id = str(uuid.uuid4())
        logger.info(f"Generated document ID: {doc_id}")

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.info("Document added to collection successfully")

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


