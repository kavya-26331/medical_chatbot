# Medical Chatbot Project Overview

This project is an AI-powered Medical Chatbot that allows users to upload medical documents and ask medical-related questions. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on ingested documents, ensuring responses are grounded in the provided context.

## Tech Stacks

### Backend (Python/FastAPI)

- **FastAPI**: Asynchronous web framework for building REST APIs.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **ChromaDB**: Vector database for storing document embeddings and metadata.
- **Sentence Transformers**: Library for generating text embeddings (using 'all-MiniLM-L6-v2' model).
- **Transformers & Torch**: Hugging Face libraries for natural language processing and machine learning.
- **Ollama**: Local LLM inference service (connects via HTTP API).
- **Python-Multipart**: For handling file uploads in FastAPI.
- **Requests**: For making HTTP requests to external services.

### Frontend (Streamlit)

- **Streamlit**: Framework for building interactive web UIs with Python.
- **Requests**: For communicating with the backend API.

### Other Technologies

- **Chroma DB Persistence**: SQLite-based storage for vector data.
- **UUID**: For generating unique document IDs to prevent overwrites.

## Architecture

- **Ingestion**: Users upload .txt files via the frontend, which are chunked, embedded, and stored in ChromaDB.
- **Query Processing**: Medical questions are classified; non-medical queries are rejected. Relevant context is retrieved from the vector store.
- **Answer Generation**: Context is passed to the LLM (via Ollama) to generate concise, accurate answers.
- **Clear Functionality**: Allows clearing the vector database before new uploads to ensure only current documents are used.

## Key Features

- Medical-only responses to ensure safety and relevance.
- Source attribution in answers.
- File upload with automatic clearing of previous data.
- Timeout handling for LLM requests.

---

## TODO

- [x] Rename /clear_collection endpoint to /clear in Backend/App/main.py
- [x] Modify Frontend/ui.py to call /clear before ingesting files
- [ ] Test the ingestion process to ensure only new files are used for answers
