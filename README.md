# Medical RAG Chatbot

An AI-powered Medical Chatbot that allows users to upload medical documents and ask medical-related questions. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on ingested documents, ensuring responses are grounded in the provided context.

## Features

- **Document Ingestion**: Upload .txt files containing medical information.
- **Medical-Only Responses**: Classifies queries and only answers medical-related questions.
- **Source Attribution**: Provides sources for answers based on ingested documents.
- **Vector Search**: Uses ChromaDB for efficient retrieval of relevant context.
- **Local LLM Integration**: Connects to Ollama for language model inference.

## Tech Stack

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

## Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running (with a model like llama3.1)

### Backend Setup

1. Navigate to the Backend directory:
   ```
   cd Backend
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the server:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. Navigate to the Frontend directory:
   ```
   cd Frontend
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run ui.py
   ```

## API Endpoints

- `GET /`: Health check
- `GET /list_sources`: List ingested document sources
- `POST /upload_doc`: Upload and ingest a document
- `POST /clear`: Clear the vector database
- `POST /chat`: Send a query and get a response

## Usage

1. Start the backend server.
2. Start the frontend app.
3. Upload medical documents via the frontend.
4. Ask medical-related questions in the chat interface.

## License

This project is licensed under the MIT License.
