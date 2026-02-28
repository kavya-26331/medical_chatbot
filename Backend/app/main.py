import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from .llm import LLM
from .rag import RAG
from .vectorstore import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print startup info for debugging
print(f"Python version: {sys.version}")
print(f"PORT environment variable: {os.getenv('PORT', 'NOT SET')}")

# Cache the model files to avoid re-downloading on every deploy
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/huggingface")
print(f"HF_HOME set to: {os.environ['HF_HOME']}")

# Global instances - loaded once at startup
_llm = None
_rag = None
_vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup - Load vectorstore once at startup (not per request)
    global _vectorstore, _llm, _rag
    print("Application starting up...")
    
    print("Loading VectorStore at startup (this may take 30-120 seconds)...")
    _vectorstore = VectorStore()
    print("VectorStore loaded successfully!")
    
    # Pre-initialize RAG with the vectorstore
    _rag = RAG(vectorstore=_vectorstore)
    print("RAG initialized with VectorStore")
    
    yield
    # Shutdown - cleanup resources
    _llm = None
    _rag = None
    _vectorstore = None
    print("Application shutting down...")

app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm

def get_rag():
    global _rag
    # RAG is already initialized at startup, just return it
    return _rag

def get_vectorstore():
    """Get the pre-loaded vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        from .vectorstore import VectorStore
        _vectorstore = VectorStore()
    return _vectorstore

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
    global _vectorstore
    vectorstore_status = "loaded" if _vectorstore is not None else "not loaded"
    return {
        "clear_route_methods": ["GET", "POST"], 
        "version": "startup-loaded-v2",
        "vectorstore_status": vectorstore_status
    }

@app.get("/clear")
def clear_db_get():
    """Clear the vector database using GET method"""
    print(f"[{datetime.now()}] GET /clear request received")
    try:
        get_vectorstore().clear_collection()
        print(f"[{datetime.now()}] Vector DB cleared (GET)")
        return {"status": "success", "message": "Vector DB cleared (GET)!"}
    except Exception as e:
        print(f"[{datetime.now()}] Error in GET /clear: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.post("/clear")
def clear_db():
    """Clear the vector database using POST method"""
    print(f"[{datetime.now()}] POST /clear request received")
    try:
        get_vectorstore().clear_collection()
        print(f"[{datetime.now()}] Vector DB cleared (POST)")
        return {"status": "success", "message": "Vector DB cleared (POST)!"}
    except Exception as e:
        print(f"[{datetime.now()}] Error in POST /clear: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.get("/list_sources")
def list_sources():
    sources = get_vectorstore().list_sources()
    return {"sources": sources}

# ------------------------
#  INGEST ROUTE
# ------------------------
@app.post("/upload_doc")
async def upload_doc(
    file: UploadFile = File(...),
    source_name: str = Form(...)
):
    print(f"[{datetime.now()}] POST /upload_doc request received for: {source_name}")
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

        # Ingest the document using pre-loaded vectorstore
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
async def chat(request: Request):
    print(f"[{datetime.now()}] POST /chat request received")
    try:
        payload = await request.json()
        query = payload.get("query", "")
        print(f"Query: {query}")

        # Classify query as medical or not
        medical_keywords = ["patient", "symptom", "diagnosis", "treatment", "medical", "health", "doctor", "medicine"]
        is_medical = any(keyword in query.lower() for keyword in medical_keywords)

        if not is_medical:
            return {
                "answer": "I'm a medical assistant. Please ask medical-related questions.",
                "sources": []
            }

        # Use pre-loaded RAG with pre-loaded vectorstore
        print("Retrieving context...")
        context, sources = get_rag().retrieve_context(query)
        print(f"Context retrieved, sources: {sources}")
        
        print("Generating answer...")
        answer = get_llm().generate_answer(query, context)
        print(f"Answer generated")

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error in POST /chat: {type(e).__name__}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }
        )
