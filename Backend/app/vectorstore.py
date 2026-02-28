import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(self):
        logger.info("Initializing VectorStore...")
        try:
            chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
            logger.info(f"ChromaDB path: {chroma_path}")

            # Use PersistentClient for data persistence
            self.client = chromadb.PersistentClient(path=chroma_path)
            logger.info("Using PersistentClient for ChromaDB")

            # Use SentenceTransformer embedding function
            # This loads the model and may take time on first call
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Using SentenceTransformer embedding function (all-MiniLM-L6-v2)")

            # Get or create collection - DON'T delete at startup
            self.collection = self.client.get_or_create_collection(
                name="medical_docs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection initialized")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _embed_text(self, text: str):
        """Embed text using the embedding function."""
        return self.embedding_function([text])[0]

    def add_document(self, text: str, metadata: dict):
        logger.info(f"Adding document, text length: {len(text)}, metadata: {metadata}")

        # Generate unique ID
        doc_id = str(uuid.uuid4())
        logger.info(f"Generated document ID: {doc_id}")

        # Manually embed the text
        embedding = self._embed_text(text)
        
        # Add to collection with pre-computed embedding
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding]
        )
        logger.info("Document added to collection successfully")

    def search(self, query: str, n_results: int = 10):
        # Manually embed the query
        query_embedding = self._embed_text(query)
        
        # Search with pre-computed embedding
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
