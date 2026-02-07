import requests

API_URL = "http://localhost:8000"

def list_sources():
    resp = requests.get(f"{API_URL}/list_sources")
    if resp.ok:
        print(resp.json())
    else:
        print("Error:", resp.text)

if __name__ == "__main__":
    list_sources()
