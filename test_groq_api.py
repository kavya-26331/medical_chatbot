"""
Test script to check if Groq API is working
"""
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Get API key
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

print("=" * 50)
print("GROQ API TEST")
print("=" * 50)

# Check if API key exists
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    print("Please add GROQ_API_KEY to your .env file")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-5:]}")
print(f"✓ Model: {model}")

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
    print("✓ Groq client initialized successfully")
except Exception as e:
    print(f"❌ ERROR initializing Groq client: {e}")
    exit(1)

# Test API with a simple request
print("\nTesting API with a simple prompt...")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello! What is 2+2? Just answer briefly."}
        ],
        temperature=0.3,
        max_tokens=50,
    )
    
    if response and response.choices:
        answer = response.choices[0].message.content
        print(f"✓ SUCCESS! API is working!")
        print(f"✓ Response: {answer}")
    else:
        print("❌ ERROR: Empty response from API")
        exit(1)
        
except Exception as e:
    print(f"❌ ERROR: API call failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("GROQ API TEST COMPLETED SUCCESSFULLY!")
print("=" * 50)
