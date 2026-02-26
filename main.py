"""
Data-X Backend — Minimal FastAPI server that executes SphinxAI and returns
JSON compatible with the Lovable frontend.

SphinxAI integration strategy (in order of priority):
  1. Python SDK: `import sphinxai` — if the pip package is installed.
     The API key is passed explicitly via the constructor or set as env var.
  2. CLI fallback: `sphinx-cli` — invoked via subprocess with SPHINX_API_KEY
     injected into the child-process environment. Expects JSON output.
  3. If neither is available the endpoint returns a clear 503 error.
"""

import json
import os
import shutil
import subprocess
import tempfile
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load .env (no-op if file doesn't exist) ──────────────────────────
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
MAX_ROWS: int = int(os.getenv("MAX_ROWS", "5000"))
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
SPHINX_API_KEY: Optional[str] = os.getenv("SPHINX_API_KEY")  # Never hard-coded

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Data-X Backend",
    description="Minimal backend that runs SphinxAI and returns Lovable-compatible JSON.",
)

# CORS — in dev defaults to "*"; in prod set CORS_ORIGINS env var.
_origins = (
    ["*"]
    if CORS_ORIGINS == "*"
    else [o.strip() for o in CORS_ORIGINS.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Attempt to import the SphinxAI Python SDK (optional) ─────────────
try:
    import sphinxai as _sphinxai_sdk  # type: ignore
except ImportError:
    _sphinxai_sdk = None


# ── Pydantic models ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    question: str
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None
    fileName: Optional[str] = None


class AnalyzeResponse(BaseModel):
    content: str
    visualizations: List[Any] = Field(default_factory=list)
    isMock: bool = False


# ── SphinxAI wrapper ─────────────────────────────────────────────────

def _run_via_sdk(question: str, df: pd.DataFrame) -> AnalyzeResponse:
    """Try the Python SDK first (sphinxai package)."""
    if _sphinxai_sdk is None:
        raise RuntimeError("SDK not installed")

    # Strategy A: sphinxai.Sphinx class
    if hasattr(_sphinxai_sdk, "Sphinx"):
        client = _sphinxai_sdk.Sphinx(api_key=SPHINX_API_KEY)
        fn = getattr(client, "chat", None) or getattr(client, "query", None)
        if fn is None:
            raise RuntimeError("Sphinx client has no chat/query method")
        res = fn(question, dataset=df)
        content = str(getattr(res, "content", res))
        visualizations = getattr(res, "visualizations", [])
        return AnalyzeResponse(content=content, visualizations=visualizations)

    # Strategy B: sphinxai.chat top-level function
    if hasattr(_sphinxai_sdk, "chat"):
        # Some SDKs read the key from the env var directly
        if SPHINX_API_KEY:
            os.environ["SPHINX_API_KEY"] = SPHINX_API_KEY
        res = _sphinxai_sdk.chat(question, df)
        if isinstance(res, dict):
            return AnalyzeResponse(
                content=str(res.get("content", str(res))),
                visualizations=res.get("visualizations", []),
            )
        return AnalyzeResponse(content=str(res))

    raise RuntimeError("Unknown sphinxai SDK interface")


def _run_via_cli(question: str, df: pd.DataFrame) -> AnalyzeResponse:
    """Fallback: invoke sphinx-cli as a subprocess."""
    cli_path = shutil.which("sphinx-cli")
    if cli_path is None:
        raise RuntimeError("sphinx-cli not found in PATH")

    if not SPHINX_API_KEY:
        raise RuntimeError("SPHINX_API_KEY is required but not set")

    # Write DataFrame to a temp CSV so the CLI can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    try:
        df.to_csv(tmp, index=False)
        tmp.close()

        env = {**os.environ, "SPHINX_API_KEY": SPHINX_API_KEY}
        result = subprocess.run(
            [cli_path, "--file", tmp.name, "--question", question],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sphinx-cli exited with code {result.returncode}: {result.stderr.strip()}"
            )

        # Attempt to parse JSON output from stdout
        try:
            parsed = json.loads(result.stdout)
            return AnalyzeResponse(
                content=str(parsed.get("content", result.stdout)),
                visualizations=parsed.get("visualizations", []),
            )
        except json.JSONDecodeError:
            # CLI returned plain text
            return AnalyzeResponse(content=result.stdout.strip())
    finally:
        os.unlink(tmp.name)


def run_sphinxai(question: str, df: pd.DataFrame) -> AnalyzeResponse:
    """Unified entry-point: try SDK → CLI → error."""
    errors: list[str] = []

    # 1) Python SDK
    if _sphinxai_sdk is not None:
        try:
            return _run_via_sdk(question, df)
        except Exception as exc:
            errors.append(f"SDK: {exc}")

    # 2) CLI fallback
    try:
        return _run_via_cli(question, df)
    except Exception as exc:
        errors.append(f"CLI: {exc}")

    # 3) Nothing worked
    raise RuntimeError(
        "SphinxAI is not available. Install the sphinxai Python package or "
        "ensure sphinx-cli is in PATH. Errors: " + " | ".join(errors)
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple liveness check."""
    sphinx_available = (
        _sphinxai_sdk is not None or shutil.which("sphinx-cli") is not None
    )
    return {
        "ok": True,
        "sphinx_available": sphinx_available,
        "max_rows": MAX_ROWS,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # ── Validation ────────────────────────────────────────────────────
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'question'.")

    if not req.data:
        raise HTTPException(status_code=400, detail="'data' is required and must not be empty.")

    if len(req.data) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Payload exceeds the maximum allowed rows ({MAX_ROWS}).",
        )

    # ── Build DataFrame ───────────────────────────────────────────────
    try:
        df = pd.DataFrame(req.data)
        if req.columns:
            available = [c for c in req.columns if c in df.columns]
            if available:
                df = df[available]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not convert data to DataFrame: {exc}",
        )

    # ── Execute SphinxAI ──────────────────────────────────────────────
    try:
        return run_sphinxai(req.question, df)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis: {exc}",
        )
