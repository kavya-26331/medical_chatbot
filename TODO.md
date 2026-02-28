# TODO - Fix API Timeout Issue

## Plan

- [x] Update Backend/app/main.py - Move VectorStore to startup (load once at startup, not per request)
- [x] Update Backend/app/rag.py - Accept vectorstore as parameter instead of lazy loading

## Implementation Steps

1. [x] Modify main.py to load VectorStore in lifespan startup event
2. [x] Modify rag.py to accept vectorstore as a parameter
3. [x] Verify changes work correctly

## Summary of Changes Made:

### Backend/app/main.py:

- Added import for VectorStore
- Created global \_vectorstore instance
- Updated lifespan to load VectorStore at startup (once, not per request)
- Updated get_rag() to return the pre-initialized RAG instance directly

### Backend/app/rag.py:

- Updated **init** to accept vectorstore as a parameter
- Falls back to creating new VectorStore if not provided (for backward compatibility)
- Simplified vectorstore property to just return the instance

## How This Fixes the Issue:

- **Before**: VectorStore (embedding model) was loaded on first request to /chat or /clear, taking 30-120 seconds and causing 502 timeout
- **After**: VectorStore loads once at application startup, so all requests are fast
