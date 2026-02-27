"""
Test script to diagnose embedding and ChromaDB issues.
Run this to verify the SentenceTransformer model works correctly.
"""
import sys
import os

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("DIAGNOSTIC TEST FOR EMBEDDINGS")
print("=" * 50)

# Test 1: Check if sentence-transformers is installed
print("\n[1] Testing sentence-transformers import...")
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers installed successfully")
except ImportError as e:
    print(f"❌ ERROR: sentence-transformers not installed: {e}")
    sys.exit(1)

# Test 2: Load the model
print("\n[2] Testing SentenceTransformer model loading...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test embedding generation
print("\n[3] Testing embedding generation...")
try:
    test_text = "This is a test medical document about patient symptoms."
    embedding = model.encode(test_text)
    print(f"✅ Embedding generated successfully, shape: {embedding.shape}")
except Exception as e:
    print(f"❌ ERROR generating embedding: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test ChromaDB with SentenceTransformerEmbeddingFunction
print("\n[4] Testing ChromaDB with SentenceTransformerEmbeddingFunction...")
try:
    from chromadb.utils import embedding_functions
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print("✅ ChromaDB embedding function created")
    
    # Test it directly
    test_embedding = embedding_function([test_text])
    print(f"✅ ChromaDB embedding test passed, embedding type: {type(test_embedding)}")
    
except Exception as e:
    print(f"❌ ERROR with ChromaDB embedding function: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test full VectorStore initialization
print("\n[5] Testing full VectorStore initialization...")
try:
    # Set a temp path for testing
    os.environ["CHROMA_DB_PATH"] = "/tmp/chroma_test_db"
    
    from app.vectorstore import VectorStore
    vs = VectorStore()
    print("✅ VectorStore initialized successfully")
    
    # Test adding a document
    test_metadata = {"source": "test_doc", "test": "true"}
    vs.add_document(test_text, test_metadata)
    print("✅ Document added successfully")
    
    # Test search
    results = vs.search("medical symptoms", n_results=1)
    print(f"✅ Search completed, results: {results}")
    
    # Test clear
    vs.clear_collection()
    print("✅ Clear collection completed")
    
except Exception as e:
    print(f"❌ ERROR with VectorStore: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED! ✅")
print("=" * 50)
