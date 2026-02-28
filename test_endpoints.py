"""
Test file to verify the Medical Chatbot API endpoints
Run this file to test the deployed endpoints on Render
"""
import requests
import sys

# Base URL for the deployed API
BASE_URL = "https://medical-chatbot-hav1.onrender.com"

def test_root():
    """Test the root endpoint"""
    print("\n=== Testing GET / ===")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_debug_methods():
    """Test the debug-methods endpoint"""
    print("\n=== Testing GET /debug-methods ===")
    try:
        response = requests.get(f"{BASE_URL}/debug-methods", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        data = response.json()
        assert response.status_code == 200
        assert "GET" in data.get("clear_route_methods", [])
        assert "POST" in data.get("clear_route_methods", [])
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clear_get():
    """Test the /clear endpoint using GET method"""
    print("\n=== Testing GET /clear ===")
    try:
        response = requests.get(f"{BASE_URL}/clear", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clear_post():
    """Test the /clear endpoint using POST method"""
    print("\n=== Testing POST /clear ===")
    try:
        response = requests.post(f"{BASE_URL}/clear", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat():
    """Test the /chat endpoint"""
    print("\n=== Testing POST /chat ===")
    try:
        payload = {"query": "What are the symptoms of diabetes?"}
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        assert response.status_code == 200
        assert "answer" in data
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Medical Chatbot API Endpoint Tests")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("GET /", test_root()))
    results.append(("GET /debug-methods", test_debug_methods()))
    results.append(("GET /clear", test_clear_get()))
    results.append(("POST /clear", test_clear_post()))
    results.append(("POST /chat", test_chat()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
