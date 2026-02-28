import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .llm import LLM
from .rag import RAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print startup info for debugging
print(f"Python version: {sys.version}")
print(f"PORT environment variable: {os.getenv('PORT', 'NOT SET')}")

# Lazy initialization to reduce memory usage at startup
_llm = None
_rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup - nothing heavy here to avoid OOM at startup
    print("Application starting up...")
    yield
    # Shutdown - cleanup resources
    global _llm, _rag
    _llm = None
    _rag = None
    print("Application shutting down...")

app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

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

@app.get("/debug-methods")
def debug_methods():
    """Debug endpoint to verify deployment version"""
    return {"clear_route_methods": ["GET", "POST"], "version": "debug-enabled-v2"}

@app.get("/list_sources")
def list_sources():
    sources = get_rag().vectorstore.list_sources()
    return {"sources": sources}

@app.post("/clear")
def clear_db():
    """
    Clear the vector database by resetting the RAG instance.
    This is more reliable on Render's ephemeral filesystem.
    """
    try:
        global _rag
        print("Resetting RAG instance...")
        logger.info("Resetting RAG instance to clear vector DB...")
        
        # Reset the RAG instance - this will force reinitialization on next use
        _rag = None
        print("RAG instance reset to None")
        
        logger.info("RAG reset successful!")
        return {"status": "success", "message": "Vector DB cleared (RAG reset)!"}
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error clearing DB: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {error_details}")
        return {
            "status": "error", 
            "message": f"Error: {type(e).__name__}: {str(e)}",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_details
        }

@app.get("/clear")
def clear_db_get():
    """
    GET endpoint for /clear - convenient for testing in browser.
    Same functionality as POST /clear but accessible via GET request.
    """
    return clear_db()

# ------------------------
#  INGEST ROUTE
# ------------------------
@app.post("/upload_doc")
async def upload_doc(
    file: UploadFile = File(...),
    source_name: str = Form(...)
):
    try:
        logger.info(f"Starting ingestion for file: {source_name}")
        
        content = await file.read()
        logger.info(f"File read successfully, size: {len(content)} bytes")

        # Try to decode with UTF-8 first, fallback to other encodings if it fails
        encodings_to_try = ["utf-8", "windows-1252", "iso-8859-1", "latin1"]
        text = None
        for encoding in encodings_to_try:
            try:
                text = content.decode(encoding)
                logger.info(f"Successfully decoded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            logger.error("Failed to decode file with any supported encoding")
            return {"status": "error", "message": "Unable to decode file content with supported encodings"}

        # Log text preview
        logger.info(f"Text decoded, length: {len(text)} characters, preview: {text[:100]}...")

        # Ingest the document
        logger.info("Calling get_rag().ingest_document()")
        get_rag().ingest_document(text, source_name)
        logger.info(f"Successfully ingested {source_name}")

        return {"status": "success", "message": f"{source_name} ingested"}
        
    except Exception as e:
        logger.error(f"Error during ingestion: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Internal Server Error: {type(e).__name__}: {str(e)}"}

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
