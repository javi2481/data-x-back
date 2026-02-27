import os
import requests
import json
import time

def run_tests():
    # If the user has a Railway URL, they can run `set API_URL=https://...` before executing
    base_url = os.environ.get("API_URL", "http://localhost:8000")
    print(f"=== Starting Functional Validation for Data-X Backend at {base_url} ===")
    
    # Common test data representing a mix of numeric and categorical values
    data_payload = [
        {"id": 1, "product": "Laptop", "price": 1200, "category": "Electronics", "date": "2023-01-15", "sales_qty": 5},
        {"id": 2, "product": "Mouse", "price": 25, "category": "Electronics", "date": "2023-01-16", "sales_qty": 150},
        {"id": 3, "product": "Desk", "price": 300, "category": "Furniture", "date": "2023-01-17", "sales_qty": 20},
        {"id": 4, "product": "Chair", "price": 150, "category": "Furniture", "date": "2023-01-18", "sales_qty": 45},
        {"id": 5, "product": "Headphones", "price": 80, "category": "Electronics", "date": "2023-01-19", "sales_qty": 80}
    ]

    # Pre-calculated statistical metadata (Smart Sampling feature test)
    stats_summary = {
        "price": {"min": 25, "max": 1200, "mean": 351, "count": 5},
        "sales_qty": {"min": 5, "max": 150, "mean": 60, "count": 5}
    }
    
    category_summary = {
        "category": {"top": [{"value": "Electronics", "count": 3}, {"value": "Furniture", "count": 2}], "uniqueCount": 2}
    }

    # 1. /health
    print("\n[*] Test 1: Health Check (Readiness & Auth)")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")
        assert r.status_code == 200, "Health check failed."
    except Exception as e:
        print(f"  [ERROR] Cannot connect to {base_url}. Error: {e}")
        return

    # 2. Error Handling
    print("\n[*] Test 2: Error Handling (Missing data on /analyze)")
    r = requests.post(f"{base_url}/analyze", json={"question": "hello"})
    print(f"  Status: {r.status_code}")
    if r.status_code != 422:
        print(f"  Response: {r.text}")
    assert r.status_code == 422, "Should return 422 Unprocessable Entity for missing required data payload."

    # 3. Standard Analysis
    print("\n[*] Test 3: Standard Analysis (/analyze) with Smart Sampling & Visualizations")
    payload = {
        "question": "What is the total sales quantity per category? Also show me a chart.",
        "data": data_payload,
        "fileName": "test_sales.csv",
        "model_size": "M",
        "stats_summary": stats_summary,
        "category_summary": category_summary,
        "total_rows": 5,
        "sampled_rows": 5
    }
    
    t0 = time.time()
    r = requests.post(f"{base_url}/analyze", json=payload)
    print(f"  Status: {r.status_code} - Time: {time.time()-t0:.2f}s")
    if r.status_code == 200:
        resp = r.json()
        print(f"  [Output Validations]")
        parsed_content = json.loads(resp.get('content', '{}'))
        text_content = parsed_content.get('content', '')
        print(f"  - Passed isMock flag: {resp.get('isMock')}")
        print(f"  - Parsed Content Length: {len(text_content)} chars")
        
        vis = parsed_content.get('visualizations', [])
        print(f"  - Visualizations built: {len(vis)}")
        if vis:
            print(f"  - First vis type: {vis[0].get('type')}")
    else:
        print(r.text)

    # 4. Deep Analysis
    print("\n[*] Test 4: Deep Analysis (/analyze/deep) for Exec Reports and Markdown Sections")
    payload_deep = {
        "question": "Genera el análisis profundo.",
        "data": data_payload,
        "fileName": "test_sales_deep.csv",
        "model_size": "L"
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/analyze/deep", json=payload_deep)
    print(f"  Status: {r.status_code} - Time: {time.time()-t0:.2f}s")
    if r.status_code == 200:
        resp = r.json()
        parsed_content = json.loads(resp.get('content', '{}'))
        deep_content = parsed_content.get("content", "")
        print(f"  - Total length: {len(deep_content)} chars")
        
        sections = [line for line in deep_content.split('\n') if line.strip().startswith('#')]
        print("  - Markdown Sections Found:")
        for s in sections[:5]:
            print(f"      {s.strip()}")
        if len(sections) == 0:
            print("  [WARN] No markdown sections found. Expected an executive report structure.")
    else:
        print(r.text)

    # 5. Semantic Search
    print("\n[*] Test 5: Semantic Search (/analyze/semantic)")
    payload_semantic = {
        "query": "algo para sentarse", # Testing language robustness natively (Spanish query -> Furniture match)
        "data": data_payload
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/analyze/semantic", json=payload_semantic)
    print(f"  Status: {r.status_code} - Time: {time.time()-t0:.2f}s")
    if r.status_code == 200:
        res = r.json().get('results', [])
        print(f"  Returned {len(res)} results.")
        if res:
            print(f"  - Top match score: {res[0].get('score'):.4f}")
            print(f"  - Top match data: {res[0].get('row', {}).get('product')}")
    else:
        print(r.text)
        
    print("\n=== Validation Scenario Generated ===")

if __name__ == '__main__':
    run_tests()
