# Medical RAG Chatbot

An AI-powered Medical Chatbot that allows users to upload medical documents and ask medical-related questions. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on ingested documents, ensuring responses are grounded in the provided context.

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [How It Works (RAG Logic)](#how-it-works-rag-logic)
- [Architecture](#architecture)
- [Component Breakdown](#component-breakdown)
- [API Endpoints](#api-endpoints)
- [Setup](#setup)
- [Usage](#usage)

---

## Project Overview

This is a full-stack Medical Chatbot application that:

1. **Accepts Document Uploads**: Users can upload `.txt` medical documents
2. **Processes & Stores Documents**: Documents are chunked, embedded, and stored in a vector database
3. **Retrieves Relevant Context**: When a user asks a question, the system finds the most relevant document sections
4. **Generates Answers**: Uses an LLM (Large Language Model) to generate accurate answers based on the retrieved context
5. **Filters Non-Medical Queries**: Only responds to medical-related questions
6. **Provides Source Attribution**: Shows which documents were used to generate the answer

---

## Tech Stack

### Backend Technologies

| Technology                | Purpose                     | Why Used                                                            |
| ------------------------- | --------------------------- | ------------------------------------------------------------------- |
| **FastAPI**               | Web framework for REST APIs | High-performance, async support, automatic API documentation        |
| **Uvicorn**               | ASGI server                 | Runs the FastAPI application efficiently                            |
| **ChromaDB**              | Vector database             | Stores document embeddings for semantic search                      |
| **Sentence Transformers** | Text embeddings             | Converts text into numerical vectors using 'all-MiniLM-L6-v2' model |
| **Transformers & Torch**  | NLP/ML libraries            | Required for sentence-transformers to work                          |
| **Groq**                  | LLM API                     | Provides fast LLM inference for answer generation                   |
| **Python-Multipart**      | File handling               | Handles file uploads in FastAPI                                     |
| **Requests**              | HTTP client                 | For making API calls within the app                                 |
| **python-dotenv**         | Environment variables       | Manages API keys and configuration                                  |

### Frontend Technologies

| Technology    | Purpose          | Why Used                                            |
| ------------- | ---------------- | --------------------------------------------------- |
| **Streamlit** | Web UI framework | Quick creation of interactive data apps with Python |
| **Requests**  | HTTP client      | Communicates with the backend API                   |

### Other Technologies

- **SQLite**: ChromaDB's persistence layer for storing vector data
- **UUID**: Generates unique IDs for each document chunk to prevent overwrites

---

## How It Works (RAG Logic)

### What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:

1. **Retrieval** - Finding relevant information from a knowledge base
2. **Augmentation** - Adding that information to the LLM's prompt
3. **Generation** - Using the LLM to generate a response

This approach ensures the LLM's answers are grounded in your documents, reducing hallucinations.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER UPLOAD FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐
  │  .txt    │───▶│  FastAPI     │───▶│  chunk_text │───▶│  Sentence     │
  │  File    │    │  /upload_doc │    │  (utils.py) │    │  Transformer  │
  └──────────┘    └──────────────┘    └─────────────┘    │  (Embedding)  │
                                                            └───────────────┘
                                                                     │
                                                                     ▼
                                                            ┌───────────────┐
                                                            │   ChromaDB    │
                                                            │  (Vector Store)│
                                                            │  - documents  │
                                                            │  - embeddings │
                                                            │  - metadata   │
                                                            └───────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐
  │  User    │───▶│  FastAPI     │───▶│  Semantic   │───▶│  Retrieve top │
  │  Query   │    │  /chat       │    │  Search     │    │  K documents  │
  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘
                                                                     │
                                                                     ▼
                                                            ┌───────────────┐
                                                            │  Combine      │
                                                            │  context +    │
                                                            │  prompt       │
                                                            └───────────────┘
                                                                     │
                                                                     ▼
                                                            ┌───────────────┐
                                                            │  Groq LLM     │
                                                            │  (llama-3.3)  │
                                                            └───────────────┘
                                                                     │
                                                                     ▼
                                                            ┌───────────────┐
                                                            │  Return       │
                                                            │  answer +     │
                                                            │  sources      │
                                                            └───────────────┘
```

### Step-by-Step Explanation

#### Step 1: Document Ingestion

1. **File Upload**: User uploads a `.txt` file via the Streamlit frontend
2. **Encoding Detection**: The system tries multiple encodings (UTF-8, Windows-1252, etc.) to decode the file
3. **Text Chunking**: The document is split into chunks of ~1000 characters using `chunk_text()` in `utils.py`
4. **Embedding Generation**: Each chunk is converted to a numerical vector using the Sentence Transformer model (`all-MiniLM-L6-v2`)
5. **Vector Storage**: The chunks, embeddings, and source metadata are stored in ChromaDB with unique UUIDs

#### Step 2: Query Processing

1. **Medical Classification**: The system checks if the query contains medical keywords (patient, symptom, diagnosis, treatment, medical, health, doctor, medicine)
2. **Non-Medical Rejection**: If no medical keywords are found, the system returns: "I'm a medical assistant. Please ask medical-related questions."
3. **Semantic Search**: The query is embedded and compared against stored document embeddings using cosine similarity
4. **Context Retrieval**: Top K (default 20) most relevant document chunks are retrieved

#### Step 3: Answer Generation

1. **Prompt Construction**: A prompt is built with:
   - Instructions for the LLM
   - The retrieved context
   - The user's question
2. **LLM Processing**: The prompt is sent to Groq's LLM API
3. **Response Extraction**: The model's response is extracted and returned
4. **Source Attribution**: The source document names are included in the response

---

## Architecture

### Directory Structure

```
MedicalChatbot/
├── Backend/
│   ├── app/
│   │   ├── __init__.py       # Package initialization
│   │   ├── main.py           # FastAPI app & endpoints
│   │   ├── rag.py            # RAG orchestrator
│   │   ├── vectorstore.py    # ChromaDB wrapper
│   │   ├── llm.py            # Groq LLM integration
│   │   ├── utils.py          # Text chunking utility
│   │   └── schemas.py        # Pydantic models
│   ├── chroma_db/            # Vector database files
│   ├── requirements.txt      # Python dependencies
│   └── .env                  # Environment variables (API keys)
├── Frontend/
│   ├── ui.py                 # Streamlit UI
│   └── requirements.txt      # Python dependencies
├── sample_medical_doc.txt   # Example document
└── README.md                 # This file
```

### Component Breakdown

#### 1. `Backend/app/main.py` (API Layer)

- **Purpose**: FastAPI application that handles all HTTP requests
- **Key Functions**:
  - `GET /` - Health check
  - `GET /list_sources` - List all ingested document sources
  - `POST /clear` - Clear the vector database
  - `POST /upload_doc` - Upload and ingest documents
  - `POST /chat` - Process chat queries

#### 2. `Backend/app/rag.py` (RAG Orchestrator)

- **Purpose**: Coordinates the RAG pipeline
- **Key Functions**:
  - `ingest_document()` - Chunks text and adds to vector store
  - `retrieve_context()` - Searches vector store and returns context + sources

#### 3. `Backend/app/vectorstore.py` (Vector Database)

- **Purpose**: Manages document embeddings
- **Key Functions**:
  - `add_document()` - Embeds and stores document chunks
  - `search()` - Performs semantic search
  - `clear_collection()` - Deletes all stored documents
  - `list_sources()` - Lists all document sources
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Similarity Metric**: Cosine similarity

#### 4. `Backend/app/llm.py` (LLM Integration)

- **Purpose**: Handles communication with Groq's LLM API
- **Key Functions**:
  - `generate_answer()` - Sends prompt to LLM and returns response
- **Model**: `llama-3.3-70b-versatile` (configurable via .env)
- **Safety**: Only uses provided context, refuses to hallucinate

#### 5. `Backend/app/utils.py` (Utilities)

- **Purpose**: Helper functions
- **Key Functions**:
  - `chunk_text()` - Splits text into manageable chunks (~1000 chars)

#### 6. `Frontend/ui.py` (User Interface)

- **Purpose**: Streamlit web interface
- **Features**:
  - Document upload with drag-and-drop
  - Chat interface with message history
  - Source attribution display
  - Custom dark/cyber medical theme

---

## API Endpoints

| Method | Endpoint        | Description              | Parameters                                |
| ------ | --------------- | ------------------------ | ----------------------------------------- |
| GET    | `/`             | Health check             | None                                      |
| GET    | `/list_sources` | List ingested documents  | None                                      |
| POST   | `/clear`        | Clear vector database    | None                                      |
| POST   | `/upload_doc`   | Upload & ingest document | `file` (UploadFile), `source_name` (Form) |
| POST   | `/chat`         | Send query, get response | `query` (JSON body)                       |

### Request/Response Examples

#### Upload Document

```
POST /upload_doc
Content-Type: multipart/form-data

file: <binary>
source_name: "medical_notes.txt"

Response:
{
  "status": "success",
  "message": "medical_notes.txt ingested"
}
```

#### Chat

```
POST /chat
Content-Type: application/json

{
  "query": "What are the symptoms of diabetes?"
}

Response:
{
  "answer": "Based on the uploaded documents, common symptoms of diabetes include...",
  "sources": ["medical_notes.txt", "symptoms_guide.txt"]
}
```

---

## Setup

### Prerequisites

- **Python 3.8+**
- **Groq API Key** - Get one at [groq.com](https://groq.com)

### Backend Setup

1. Navigate to the Backend directory:

```
bash
   cd Backend

```

2. Create a `.env` file in the Backend folder:

```
env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.3-70b-versatile

```

3. Install dependencies:

```
bash
   pip install -r requirements.txt

```

4. Run the server:

```
bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

### Frontend Setup

1. Navigate to the Frontend directory:

```
bash
   cd Frontend

```

2. Install dependencies:

```
bash
   pip install -r requirements.txt

```

3. Run the Streamlit app:

```
bash
   streamlit run ui.py

```

4. Open your browser at `http://localhost:8501`

---

## Usage

### Step 1: Start the Application

1. Start the backend server (terminal 1):

```
bash
   cd Backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

2. Start the frontend (terminal 2):

```
bash
   cd Frontend
   streamlit run ui.py

```

### Step 2: Upload Documents

1. Open `http://localhost:8501` in your browser
2. Use the sidebar to upload `.txt` medical documents
3. Click "Ingest Files" to process the documents
4. The vector database will be populated with embedded chunks

### Step 3: Ask Questions

1. Type your medical question in the chat input
2. Click "Ask AI"
3. The system will:
   - Verify it's a medical question
   - Search for relevant context
   - Generate an answer using the LLM
   - Display sources used

### Example Documents

Use the provided `sample_medical_doc.txt` to test the system:

```
txt
Diabetes Management Guide

Type 2 Diabetes Symptoms:
- Increased thirst
- Frequent urination
- Extreme hunger
- Unexplained weight loss
- Fatigue
- Blurred vision

Treatment Options:
- Metformin
- Lifestyle changes
- Regular exercise
- Healthy diet
```

---

## License

This project is licensed under the MIT License.
#   M e d i c a l _ C h a t b o t  
 #   m e d i c a l _ c h a t b o t  
 #   A i _ m e d i c a l _ c h a t b o t  
 