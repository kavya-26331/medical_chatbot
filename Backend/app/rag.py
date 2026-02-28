from .vectorstore import VectorStore
from .utils import chunk_text

class RAG:
    def __init__(self, vectorstore: VectorStore = None):
        # Accept vectorstore as parameter (passed from main.py at startup)
        # Fallback to creating a new one if not provided (for backward compatibility)
        if vectorstore is None:
            self._vectorstore = VectorStore()
        else:
            self._vectorstore = vectorstore

    @property
    def vectorstore(self):
        """Return the vectorstore instance."""
        return self._vectorstore

    def ingest_document(self, text: str, source_name: str):
        """
        Ingest a document by chunking it and adding to vector store.
        """
        chunks = chunk_text(text)
        for chunk in chunks:
            self.vectorstore.add_document(chunk, {"source": source_name})

    def retrieve_context(self, query: str, n_results: int = 5):
        """
        Retrieve relevant context and sources from vector store based on query.
        """
        results = self.vectorstore.search(query, n_results)
        if results and results['documents']:
            context = ' '.join(results['documents'][0])
            sources = list(set([meta['source'] for meta in results['metadatas'][0]] if results['metadatas'] else []))
            return context, sources
        return "", []
