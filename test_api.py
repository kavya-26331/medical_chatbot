import requests
import time

API_URL = "http://localhost:8000"

def test_chat():
    payload = {"query": "What is diabetes?"}
    try:
        start_time = time.time()
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        if resp.ok:
            res = resp.json()
            print("Success:", res)
        else:
            print("Error:", resp.text)
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    test_chat()
