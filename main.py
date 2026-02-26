"""
Data-X Backend — Minimal FastAPI server that executes SphinxAI and returns
JSON compatible with the Lovable frontend.

SphinxAI integration:
  The CLI `sphinx-cli` (installed via `pip install sphinx-ai-cli`) is the
  primary execution path.  It requires a .ipynb notebook, so the wrapper:
    1. Writes the incoming DataFrame to a temp CSV.
    2. Creates a minimal .ipynb that loads that CSV into a DataFrame.
    3. Invokes `sphinx-cli chat --notebook-filepath <nb> --prompt <question>`
       with SPHINX_API_KEY injected in the subprocess environment.
    4. Parses stdout into the normalised {content, visualizations, isMock} shape.

  Auth: SPHINX_API_KEY must be set as an environment variable (never hard-coded).
  Docs: https://docs.sphinx.ai/cli-documentation/installation
"""

import json
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
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
    description="Minimal backend that runs SphinxAI via CLI and returns "
                "Lovable-compatible JSON.",
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


# ── Helpers ───────────────────────────────────────────────────────────

def _sphinx_cli_available() -> bool:
    """Return True if sphinx-cli is reachable."""
    return shutil.which("sphinx-cli") is not None


def _build_notebook(csv_path: str, file_name: str) -> nbformat.NotebookNode:
    """Create a minimal .ipynb that loads the CSV into `df`."""
    nb = nbformat.v4.new_notebook()
    code = (
        "import pandas as pd\n"
        f"df = pd.read_csv(r'{csv_path}')\n"
        f"# Source file: {file_name}\n"
        "df.head()"
    )
    nb.cells.append(nbformat.v4.new_code_cell(source=code))
    return nb


def _run_via_cli(question: str, df: pd.DataFrame, file_name: str) -> AnalyzeResponse:
    """
    Execute sphinx-cli chat with a temporary notebook + CSV.
    Auth is via SPHINX_API_KEY env var injected into subprocess.
    """
    if not _sphinx_cli_available():
        raise RuntimeError("sphinx-cli is not installed or not in PATH")
    if not SPHINX_API_KEY:
        raise RuntimeError(
            "SPHINX_API_KEY env var is required but not set. "
            "Get your key at https://dashboard.prod.sphinx.ai"
        )

    tmp_dir = tempfile.mkdtemp(prefix="datax_")
    try:
        # 1. Write CSV
        csv_path = os.path.join(tmp_dir, file_name or "data.csv")
        df.to_csv(csv_path, index=False)

        # 2. Build notebook
        nb = _build_notebook(csv_path, file_name or "data.csv")
        nb_path = os.path.join(tmp_dir, "analysis.ipynb")
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        # 3. Invoke sphinx-cli chat
        env = {**os.environ, "SPHINX_API_KEY": SPHINX_API_KEY}
        result = subprocess.run(
            [
                "sphinx-cli", "chat",
                "--notebook-filepath", nb_path,
                "--prompt", question,
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or "(no stderr)"
            raise RuntimeError(
                f"sphinx-cli exited with code {result.returncode}: {stderr}"
            )

        stdout = result.stdout.strip()

        # 4. Try to parse structured JSON from stdout
        try:
            parsed = json.loads(stdout)
            return AnalyzeResponse(
                content=str(parsed.get("content", stdout)),
                visualizations=parsed.get("visualizations", []),
            )
        except json.JSONDecodeError:
            # CLI returned plain text — wrap it
            return AnalyzeResponse(content=stdout)

    finally:
        # Clean up temp files
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness + readiness check."""
    cli_ok = _sphinx_cli_available()
    key_ok = bool(SPHINX_API_KEY)
    return {
        "ok": True,
        "sphinx_available": cli_ok,
        "sphinx_authenticated": cli_ok and key_ok,
        "max_rows": MAX_ROWS,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # ── Validation ────────────────────────────────────────────────────
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'question'.")

    if not req.data:
        raise HTTPException(
            status_code=400,
            detail="'data' is required and must not be empty.",
        )

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

    # ── Execute SphinxAI via CLI ──────────────────────────────────────
    try:
        return _run_via_cli(req.question, df, req.fileName or "data.csv")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis: {exc}",
        )
