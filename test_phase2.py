import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

def test_phase2_endpoints():
    print("--- Starting Phase 2 Smoke Test ---")
    
    # 1. Check History
    print("\n1. Testing GET /analyze/history...")
    try:
        resp = requests.get(f"{BASE_URL}/analyze/history")
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            print(f"Found {len(history)} items in history.")
            if history:
                print(f"First item: {history[0]['analysis_id']} - {history[0]['status']}")
    except Exception as e:
        print(f"Error testing history: {e}")

    # 2. Test specific analysis ID (using a dummy one if history empty or first one)
    if 'history' in locals() and history:
        analysis_id = history[0]['analysis_id']
        print(f"\n2. Testing GET /analyze/{analysis_id}...")
        try:
            resp = requests.get(f"{BASE_URL}/analyze/{analysis_id}")
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print("Successfully retrieved analysis details.")
        except Exception as e:
            print(f"Error testing detail: {e}")

    print("\n--- Smoke Test Finished ---")

if __name__ == "__main__":
    # Note: This assumes the server is running on localhost:8000
    test_phase2_endpoints()
