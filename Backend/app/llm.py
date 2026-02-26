import os
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv


# Load environment variables with explicit path from Backend folder
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


class LLM:
    def __init__(self):
        # Load model and API key from environment
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.api_key = os.getenv("GROQ_API_KEY")

        # Debug print
        print("DEBUG - Loaded API Key:", self.api_key)
        print("DEBUG - Loaded Model:", self.model)

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=self.api_key)

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using Groq LLM with medical-safe prompting.
        """

        # Limit context size (avoid token overflow)
        truncated_context = context[:4000] if context else ""

        prompt = f"""
You are a professional medical assistant AI.

Instructions:
- Use ONLY the provided context.
- Do NOT make up information.
- If the context does not contain relevant data, say:
  "I don't have enough information to answer this question."
- Be accurate and concise.

Context:
{truncated_context}

Question:
{query}

Answer:
"""

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800,
                    timeout=120,  # Increased to 120 second timeout for complex queries
                )

                if response and response.choices:
                    return response.choices[0].message.content.strip()

                return "Error: Model returned empty response."

            except Exception as e:
                error_message = str(e)

                # Specific error handling
                if "model_decommissioned" in error_message:
                    return (
                        "Error: The selected Groq model is no longer supported. "
                        "Please update LLM_MODEL in your .env file."
                    )

                # Handle timeout errors specifically
                if "timeout" in error_message.lower() or "timed out" in error_message.lower():
                    if attempt == max_retries - 1:
                        return "Error: The request timed out. Please try again or reduce your query complexity."
                    continue  # Skip the sleep and retry immediately for timeouts

                if attempt == max_retries - 1:
                    return f"Error: Failed after multiple attempts. {error_message}"

                # Wait before retrying
                time.sleep(2)

        return "Error: Unexpected failure."
