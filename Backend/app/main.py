import os
import sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .llm import LLM
from .rag import RAG

# Print startup info for debugging
print(f"Python version: {sys.version}")
print(f"PORT environment variable: {os.getenv('PORT', 'NOT SET')}")

app = FastAPI(title="Medical Chatbot API")

# Lazy initialization to reduce memory usage at startup
_llm = None
_rag = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm

def get_rag():
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/list_sources")
def list_sources():
    sources = get_rag().vectorstore.list_sources()
    return {"sources": sources}

@app.post("/clear")
def clear_db():
    get_rag().vectorstore.clear_collection()
    return {"message": "Vector DB cleared!"}

# ------------------------
#  INGEST ROUTE
# ------------------------
@app.post("/upload_doc")
async def upload_doc(
    file: UploadFile = File(...),
    source_name: str = Form(...)
):
    content = await file.read()

    # Try to decode with UTF-8 first, fallback to other encodings if it fails
    encodings_to_try = ["utf-8", "windows-1252", "iso-8859-1", "latin1"]
    text = None
    for encoding in encodings_to_try:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        return {"status": "error", "message": "Unable to decode file content with supported encodings"}

    get_rag().ingest_document(text, source_name)

    return {"status": "success", "message": f"{source_name} ingested"}

# ------------------------
#  CHAT ROUTE
# ------------------------
@app.post("/chat")
async def chat(payload: dict):
    try:
        query = payload.get("query", "")

        # Classify query as medical or not
        medical_keywords = ["patient", "symptom", "diagnosis", "treatment", "medical", "health", "doctor", "medicine"]
        is_medical = any(keyword in query.lower() for keyword in medical_keywords)

        if not is_medical:
            return {
                "answer": "I'm a medical assistant. Please ask medical-related questions.",
                "sources": []
            }

        context, sources = get_rag().retrieve_context(query)
        answer = get_llm().generate_answer(query, context)

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        }
