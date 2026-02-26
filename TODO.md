# TODO - Fix Internal Server Error on txt file ingestion

## Task

Fix the "Internal Server Error" that occurs when ingesting txt files in deployment.

## Changes Made

- [x] Add proper exception handling in Backend/app/main.py upload_doc endpoint
- [x] Add logging throughout the ingestion flow for debugging in Backend/app/vectorstore.py
- [x] Added logging to identify where the failure occurs

## What to do next

1. **Deploy the updated code** to your deployment server
2. **Try uploading the txt file again** - you should now see a specific error message instead of "Internal Server Error"
3. **Check the deployment logs** to see the detailed error

## Common causes based on the error message:

- **GROQ_API_KEY not found**: Set GROQ_API_KEY in your deployment platform's environment variables
- **Embedding API errors**: Check if the Groq API key has sufficient quota
- **ChromaDB path errors**: The ChromaDB needs writable storage in deployment

## Status: Completed - Deploy and test
