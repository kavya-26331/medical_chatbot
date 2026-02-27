"""
Test file to check the /clear endpoint on the deployed Render backend.
Run this after deploying the updated code to Render.
"""

import requests
import json

# The deployed backend URL
BASE_URL = "https://medical-chatbot-hav1.onrender.com"

def test_debug_methods():
    """Test the debug endpoint to verify deployment version"""
    print("\n" + "="*50)
    print("Testing /debug-methods endpoint")
    print("="*50)
    
    url = f"{BASE_URL}/debug-methods"
    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def test_clear_endpoint():
    """Test the /clear endpoint to check for errors"""
    print("\n" + "="*50)
    print("Testing /clear endpoint")
    print("="*50)
    
    url = f"{BASE_URL}/clear"
    try:
        response = requests.post(url, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        try:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Check for detailed error info
            if data.get("status") == "error":
                print("\n" + "-"*50)
                print("ERROR DETECTED!")
                print("-"*50)
                if "error_type" in data:
                    print(f"Error Type: {data.get('error_type')}")
                if "error_message" in data:
                    print(f"Error Message: {data.get('error_message')}")
                if "traceback" in data:
                    print(f"Traceback:\n{data.get('traceback')}")
            
            return data
            
        except json.JSONDecodeError:
            print(f"Raw Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def test_root_endpoint():
    """Test the root endpoint to check if service is running"""
    print("\n" + "="*50)
    print("Testing / (root) endpoint")
    print("="*50)
    
    url = f"{BASE_URL}/"
    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MEDICAL CHATBOT BACKEND TEST")
    print("="*60)
    
    # Test 1: Check if service is running
    root_ok = test_root_endpoint()
    
    # Test 2: Check debug methods (to verify new deployment)
    debug_ok = test_debug_methods()
    
    # Test 3: Test the clear endpoint
    clear_result = test_clear_endpoint()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Root endpoint (/): {'✅ OK' if root_ok else '❌ FAILED'}")
    print(f"Debug endpoint (/debug-methods): {'✅ OK' if debug_ok else '❌ FAILED'}")
    
    if clear_result:
        if clear_result.get("status") == "success":
            print(f"Clear endpoint: ✅ SUCCESS - {clear_result.get('message')}")
        else:
            print(f"Clear endpoint: ❌ FAILED - {clear_result.get('message')}")
            print("\n💡 The error above is the actual issue! Check the traceback for details.")
    else:
        print("Clear endpoint: ❌ FAILED (no response)")
    
    print("="*60)
