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
            # Use in-memory ChromaDB client (no persistence = less memory)
            # This is the fastest quick fix for Render's 512MB limit
            self.client = chromadb.Client()
            logger.info("Using in-memory ChromaDB client (no persistence)")

            # Load embedding function at startup (not lazy!)
            # Using all-MiniLM-L6-v2 which is smaller, faster and more stable on low RAM
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2) at startup...")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # Smaller model = less RAM, faster loading
            )
            logger.info("✅ SentenceTransformer model loaded at startup")

            # Get or create collection with embedding function
            self.collection = self.client.get_or_create_collection(
                name="medical_docs",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            logger.info("ChromaDB collection initialized with embedding function")
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
        # Delete the collection entirely and recreate it fresh
        try:
            self.client.delete_collection(name="medical_docs")
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.info(f"Collection may not exist, proceeding to create: {e}")
        
        # Create a new empty collection
        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        logger.info("Created new empty collection")

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
