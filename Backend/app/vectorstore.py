import chromadb
from groq import Groq
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
        
        # Use Groq as the embedding function for ChromaDB
        embed_model_name = os.getenv("EMBEDDING_MODEL", "llama-3.1-8b-instant")
        logger.info(f"Loading embedding model: {embed_model_name}")
        
        # Configure Groq API
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embed_model_name = embed_model_name
        
        # Create a custom embedding function for Groq
        self.embedding_fn = self._create_groq_embedding_function()
        
        # Delete existing collection if it exists (to avoid the old embedding function issue)
        try:
            self.client.delete_collection(name="medical_docs")
            logger.info("Deleted existing collection to avoid embedding function conflict")
        except:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )
        logger.info("ChromaDB collection initialized with Groq embedding")

    def _create_groq_embedding_function(self):
        """
        Create a custom embedding function for Groq API.
        """
        def groq_embedding_function(input_texts: list) -> list:
            """
            Embed texts using Groq's embedding API.
            """
            embeddings = []
            for text in input_texts:
                response = self.groq_client.embeddings(
                    model=self.embed_model_name,
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            return embeddings
        
        return groq_embedding_function

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
