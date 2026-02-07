import requests
import os

class LLM:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "llama3.1")
        self.api_url = "http://localhost:11434/api/generate"

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the Ollama API with the provided context.
        """
        # Reduce context to 4000 characters to avoid overloading the model
        truncated_context = context[:4000] if len(context) > 4000 else context
        prompt = f"You are a medical assistant. Use the provided context to answer the question accurately and concisely. If the context does not contain relevant information, say 'I don't have enough information to answer this question.' Do not make up information.\n\nContext: {truncated_context}\n\nQuestion: {query}\n\nAnswer:"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.post(self.api_url, json=data, timeout=120)  # Increased timeout to 120 seconds
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    return answer
                else:
                    return f"Error: Unable to generate answer. API responded with status {response.status_code}."
            except requests.exceptions.Timeout:
                if attempt == 2:  # Last attempt
                    return "Error: The request to the language model timed out after multiple attempts. Please try again later."
                continue  # Retry
            except requests.exceptions.RequestException as e:
                return f"Error: Failed to connect to the language model service. {str(e)}"
            except Exception as e:
                return f"Error: An unexpected error occurred while generating the answer. {str(e)}"

