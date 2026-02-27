import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(self):
        logger.info("Initializing VectorStore...")
        
        # ✅ Use /tmp/chroma_db for cloud deployment compatibility
        # This works on Streamlit Cloud, Render, and other ephemeral filesystems
        chroma_path = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
        logger.info(f"ChromaDB path: {chroma_path}")

        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)

        # ✅ Use proper ChromaDB embedding class (SentenceTransformer)
        # This is the CORRECT way to configure embeddings in ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("Using SentenceTransformer embedding function (all-MiniLM-L6-v2)")

        # 🔥 PRE-WARM THE MODEL: Force download/load on startup to avoid cold start issues
        # This helps on cloud deployments where the model isn't pre-cached
        try:
            logger.info("Pre-warming embedding model...")
            test_embedding = self.embedding_function(["warmup"])
            logger.info(f"Embedding model warmed up successfully, test embedding shape: {len(test_embedding)}")
        except Exception as e:
            logger.warning(f"Model warmup warning (non-fatal): {type(e).__name__}: {str(e)}")

        # ✅ Get or create collection WITHOUT deleting existing one
        # This prevents issues with persistent storage on cloud deployments
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
        """Clear all documents from the collection by deleting and recreating it."""
        try:
            # Delete the entire collection and recreate it fresh
            # This is more robust than trying to delete individual IDs
            logger.info("Deleting the entire collection...")
            self.client.delete_collection(name="medical_docs")
            logger.info("Collection deleted successfully")
            
            # Recreate the collection with the same settings
            logger.info("Recreating collection with fresh state...")
            self.collection = self.client.get_or_create_collection(
                name="medical_docs",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            logger.info("Collection recreated successfully with fresh state")
            return True
                
        except Exception as e:
            logger.error(f"Error clearing collection: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try to recreate collection as fallback
            try:
                self.collection = self.client.get_or_create_collection(
                    name="medical_docs",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )
                logger.info("Recreated collection after clear error")
                return True
            except Exception as recreate_error:
                logger.error(f"Failed to recreate collection: {recreate_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False

    def list_sources(self):
        try:
            # Try to get the collection, if it doesn't exist, create it fresh
            try:
                self.collection = self.client.get_collection(name="medical_docs")
            except Exception:
                # Collection doesn't exist, create it fresh
                self.collection = self.client.get_or_create_collection(
                    name="medical_docs",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )

            results = self.collection.get(include=["metadatas"])
            source_counts = {}
            if results and "metadatas" in results and results["metadatas"]:
                for meta in results["metadatas"]:
                    if meta:  # Check if meta is not None
                        source = meta.get("source", "unknown")
                        source_counts[source] = source_counts.get(source, 0) + 1

            return {
                "sources": list(source_counts.keys()),
                "counts": source_counts
            }
        except Exception as e:
            logger.error(f"Error listing sources: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"sources": [], "counts": {}}
