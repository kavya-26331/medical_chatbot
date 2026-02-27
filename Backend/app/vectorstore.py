import chromadb
from chromadb.utils import embedding_functions
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

        # ✅ Use proper ChromaDB embedding class (SentenceTransformer)
        # This is the CORRECT way to configure embeddings in ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("Using SentenceTransformer embedding function (all-MiniLM-L6-v2)")

        # Delete existing collection if it exists (to avoid the old embedding function issue)
        try:
            self.client.delete_collection(name="medical_docs")
            logger.info("Deleted existing collection to avoid embedding function conflict")
        except:
            pass

        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        logger.info("ChromaDB collection initialized with SentenceTransformer embedding")

    def add_document(self, text: str, metadata: dict):
        logger.info(f"Adding document, text length: {len(text)}, metadata: {metadata}")

        # FIX: Always unique ID — prevents overwrite
        doc_id = str(uuid.uuid4())
        logger.info(f"Generated document ID: {doc_id}")

        # Let ChromaDB handle embeddings automatically via embedding_function
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.info("Document added to collection successfully")

    def search(self, query: str, n_results: int = 10):
        # Let ChromaDB handle query embedding automatically via embedding_function
        results = self.collection.query(
            query_texts=[query],
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
