import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_sse_flow():
    print("--- Starting Phase 3 SSE Smoke Test ---")
    
    # 1. Plan
    print("\n1. Requesting Plan...")
    plan_payload = {
        "question": "What is the average age?",
        "data": [{"name": "A", "age": 25}, {"name": "B", "age": 30}],
        "fileName": "test.csv"
    }
    plan_resp = requests.post(f"{BASE_URL}/analyze/plan", json=plan_payload)
    if plan_resp.status_code != 200:
        print(f"Plan failed: {plan_resp.text}")
        return
        
    analysis_id = plan_resp.json()["analysis_id"]
    print(f"Plan created with ID: {analysis_id}")
    
    # 2. Approve
    print("\n2. Approving Plan...")
    approve_resp = requests.post(f"{BASE_URL}/analyze/approve", json={"analysis_id": analysis_id})
    if approve_resp.status_code != 200:
        print(f"Approve failed: {approve_resp.text}")
        return
    print("Plan approved.")
        
    # 3. Execute (Background)
    print("\n3. Triggering Execution (Background)...")
    exec_resp = requests.post(f"{BASE_URL}/analyze/execute", json={"analysis_id": analysis_id})
    if exec_resp.status_code != 200:
        print(f"Execute start failed: {exec_resp.text}")
        return
        
    print(f"Execution started in background: {exec_resp.json()}")

    # 4. Listen to SSE Stream
    print("\n4. Listening to SSE stream...")
    try:
        # We use stream=True to read chunks as they arrive
        stream_resp = requests.get(f"{BASE_URL}/analyze/stream/{analysis_id}", stream=True, timeout=120)
        
        for line in stream_resp.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                print(f"SSE Payload -> {decoded}")
                if "event: complete" in decoded or "event: error" in decoded:
                    print("Stream finished.")
                    break
    except Exception as e:
        print(f"Stream error: {e}")

    # 5. Get Final Result
    time.sleep(1) # wait a tiny bit for DB save
    print("\n5. Fetching final analysis document...")
    final_resp = requests.get(f"{BASE_URL}/analyze/{analysis_id}")
    if final_resp.status_code == 200:
        doc = final_resp.json()
        status = doc.get("status")
        print(f"Final status in DB: {status}")
        if status == "completed":
            print(f"Artifacts generated: {len(doc.get('run', {}).get('artifacts', []))}")
            
    print("\n--- Phase 3 SSE Smoke Test Finished ---")

if __name__ == "__main__":
    test_sse_flow()
