import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
import logging
import tempfile

logger = logging.getLogger(__name__)

class VectorStore:
    # Class-level model cache to avoid reloading
    _embedding_function = None

    def __init__(self):
        logger.info("Initializing VectorStore...")
        try:
            # Create a unique path for each instance using PID
            self.persist_directory = os.path.join(tempfile.gettempdir(), f"chroma_db_{os.getpid()}")
            logger.info(f"Using ChromaDB path: {self.persist_directory}")

            # Use PersistentClient for data persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("Using PersistentClient for ChromaDB")

            # Lazy load the embedding function (class-level cache)
            if VectorStore._embedding_function is None:
                logger.info("Loading SentenceTransformer model...")
                VectorStore._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("✅ SentenceTransformer model loaded and cached")
            else:
                logger.info("Using cached SentenceTransformer model")
            
            self.embedding_function = VectorStore._embedding_function

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
