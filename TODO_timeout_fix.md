# TODO: Fix ReadTimeout Error in MedicalChatbot

## Problem Description

The MedicalChatbot application was experiencing a ReadTimeout error when users submitted chat queries. The error occurred in the Streamlit frontend (Frontend/ui.py) at line 40, where a POST request to the backend's /chat endpoint timed out after 120 seconds. This timeout was insufficient because the backend's LLM generation process, which calls the Ollama API, could take longer than 120 seconds, especially with retry logic (up to 3 attempts).

## Technologies Used

- **Frontend:** Streamlit (Python web app framework) for the user interface, located in Frontend/ui.py.
- **Backend:** FastAPI (Python web framework) for the API server, located in Backend/app/main.py.
- **HTTP Client:** requests library (Python) for making HTTP calls from frontend to backend.
- **Vector Database:** ChromaDB for storing and retrieving document embeddings, located in Backend/app/vectorstore.py.
- **Embeddings:** SentenceTransformer (all-MiniLM-L6-v2 model) for generating text embeddings.
- **LLM Service:** Ollama API (running locally on port 11434) for generating answers using the llama3.1 model.
- **RAG System:** Custom RAG class (Backend/app/rag.py) that combines retrieval from vector store with LLM generation.

## Process Flow

1. User enters a query in the Streamlit frontend.
2. Frontend sends a POST request to Backend /chat endpoint with the query.
3. Backend classifies the query as medical or not.
4. If medical, it retrieves relevant context using RAG.retrieve_context:
   - Encodes the query using SentenceTransformer.
   - Searches ChromaDB collection for similar documents.
5. Then generates an answer using LLM.generate_answer:
   - Truncates context to 4000 characters.
   - Sends prompt to Ollama API with 120-second timeout and up to 3 retries.
6. Response is sent back to frontend, which displays it.

## Root Cause

The frontend timeout (120 seconds) matched the Ollama API timeout, but with retries, the total time could exceed 120 seconds, causing the ReadTimeout before the backend could respond.

## Solution

Increased the frontend timeout from 120 to 300 seconds to allow sufficient time for the LLM response, including retries.

## Steps to Complete

- [x] Increase timeout in Frontend/ui.py from 120 to 300 seconds for the /chat POST request
