import requests
import json

url = "http://localhost:8000/analyze"
payload = {
    "question": "Analyze the sales trends.",
    "data": [
        {"ventas": 100, "region": "Norte"},
        {"ventas": 200, "region": "Sur"}
    ],
    "stats_summary": {
        "ventas": {
            "min": 100, "max": 95000, "mean": 12340, "count": 5000,
            "p5": 250, "p25": 3200, "p50": 8900, "p75": 18500, "p95": 52000
        }
    },
    "category_summary": {
        "region": {
            "top": [
                {"value": "Norte", "count": 1200},
                {"value": "Sur", "count": 980}
            ],
            "othersCount": 340,
            "uniqueCount": 15
        }
    },
    "total_rows": 5000,
    "sampled_rows": 2
}

print(f"Sending payload to {url}...")
try:
    # We are just testing the payload validation and local logic here.
    # The actual LLM call might fail if keys are missing but that's okay for verifying the payload schema.
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
