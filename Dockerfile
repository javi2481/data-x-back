FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify sphinx-cli is available after install
RUN sphinx-cli -h > /dev/null 2>&1 && echo "sphinx-cli OK" || echo "sphinx-cli NOT FOUND"

COPY . .

# Railway injects $PORT dynamically
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
