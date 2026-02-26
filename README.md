# Data-X Back

Backend mínimo en **Python (FastAPI)** que ejecuta **SphinxAI** directamente (sin Jupyter/Gateway) y devuelve JSON compatible con el frontend **Lovable**.

---

## Estructura

```
main.py            ← Aplicación FastAPI + wrapper SphinxAI
requirements.txt   ← Dependencias pip
.env.example       ← Variables de entorno documentadas
Dockerfile         ← Deploy a Railway / Docker
.gitignore
README.md
```

## Requisitos

- Python 3.11+
- (Opcional) `sphinx-cli` en PATH **ó** paquete `sphinxai` instalado.
- Variable de entorno `SPHINX_API_KEY` con la clave de API.

---

## Ejecución Local

```bash
# 1. Clonar
git clone https://github.com/javi2481/data-x-back.git
cd data-x-back

# 2. Entorno virtual
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Copiar y llenar variables de entorno
cp .env.example .env
# editar .env con tu SPHINX_API_KEY, MAX_ROWS, CORS_ORIGINS

# 5. Arrancar servidor
uvicorn main:app --reload
```

El servidor inicia en `http://localhost:8000`.

---

## Variables de Entorno

| Variable         | Requerida          | Default | Descripción                                                                                                       |
| ---------------- | ------------------ | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `SPHINX_API_KEY` | Sí (para análisis) | —       | Clave de API de SphinxAI. Se pasa al SDK como parámetro explícito o como env var al CLI. **Nunca se hard-codea.** |
| `MAX_ROWS`       | No                 | `5000`  | Máximo de filas por petición `/analyze`.                                                                          |
| `CORS_ORIGINS`   | No                 | `*`     | Orígenes CORS permitidos, separados por coma. En producción restringir a `https://tu-app.lovable.app`.            |

---

## Endpoints

### `GET /health`

Liveness check.

```bash
curl http://localhost:8000/health
```

Respuesta:

```json
{ "ok": true, "sphinx_available": false, "max_rows": 5000 }
```

### `POST /analyze`

Ejecuta SphinxAI sobre los datos enviados.

**Request:**

```json
{
  "question": "¿Cuál es la venta promedio?",
  "data": [
    { "producto": "A", "ventas": 100 },
    { "producto": "B", "ventas": 250 }
  ],
  "columns": ["producto", "ventas"],
  "fileName": "ventas.csv"
}
```

**Response (siempre):**

```json
{
  "content": "La venta promedio es 175.",
  "visualizations": [],
  "isMock": false
}
```

**Errores:**
| Código | Cuándo |
|---|---|
| 400 | Faltan campos o `data` está vacío |
| 413 | `data` excede `MAX_ROWS` |
| 500 | Falla SphinxAI (mensaje claro) |

**Ejemplo `curl`:**

```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "question": "¿Cuál es la venta promedio?",
       "data": [
         {"producto": "A", "ventas": 100},
         {"producto": "B", "ventas": 250}
       ]
     }'
```

---

## Integración Lovable

El frontend / edge function debe:

1. **Dejar de usar** `JUPYTER_SERVER_URL` y `JUPYTER_TOKEN`.
2. **Configurar** `BACKEND_BASE_URL` (ej: `https://<railway-app>.up.railway.app`).
3. **Llamar** `POST {BACKEND_BASE_URL}/analyze` con el body `{ question, data, columns?, fileName? }`.

---

## Deploy en Railway

1. Conectar el repo GitHub a Railway.
2. Railway detectará el `Dockerfile` automáticamente.
3. En la pestaña **Variables** del servicio, agregar:
   - `SPHINX_API_KEY`
   - `MAX_ROWS` (opcional, default 5000)
   - `CORS_ORIGINS` (ej: `https://tu-app.lovable.app`)
4. Railway asigna `$PORT` dinámicamente; el Dockerfile ya lo usa.
5. Start command (si no se usa Docker): `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## SphinxAI — Estrategia de integración

El backend intenta dos caminos en orden:

1. **Python SDK** (`import sphinxai`): si el paquete está instalado, se instancia el cliente pasando `SPHINX_API_KEY` explícitamente al constructor.
2. **CLI fallback** (`sphinx-cli`): si el SDK no está disponible, se invoca `sphinx-cli` como subproceso con `SPHINX_API_KEY` inyectado en el entorno del proceso hijo (nunca impreso en logs).

Si ninguno está disponible, `/health` reporta `"sphinx_available": false` y `/analyze` devuelve un error 500 claro.

---

## Evidence

> Verificación local ejecutada el 2026-02-25.

### `GET /health`

```
$ curl http://localhost:8000/health

{"ok":true,"sphinx_available":true,"max_rows":5000}
```

### `POST /analyze` — error 400 (question vacía)

```
$ curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"question":"","data":[]}'

HTTP 400 → {"detail":"Missing or empty 'question'."}
```

### `POST /analyze` — error 400 (data vacío)

```
$ curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"question":"test","data":[]}'

HTTP 400 → {"detail":"'data' is required and must not be empty."}
```

### `POST /analyze` — error 500 (SphinxAI no configurado)

```
$ curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"question":"avg sales?","data":[{"producto":"A","ventas":100},{"producto":"B","ventas":250}]}'

HTTP 500 → {"detail":"SphinxAI is not available. Install the sphinxai Python package or ensure sphinx-cli is in PATH. Errors: CLI: SPHINX_API_KEY is required but not set"}
```

> **Nota:** En producción con `SPHINX_API_KEY` configurada y `sphinx-cli` / SDK disponible, el endpoint devuelve la respuesta normalizada `{ content, visualizations, isMock }`.
