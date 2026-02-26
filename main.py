"""
Data-X Backend — Minimal FastAPI server that executes SphinxAI using the
official `sphinxai` Python library.

Integration strategy:
  Uses the `sphinxai` library directly to call LLMs.
  1. Configures the provider with `SPHINX_API_KEY`.
  2. Submits the question and a sample of the data to the LLM.
  3. Returns the normalized response: {content, visualizations, isMock}.

  Reference: https://docs.sphinx.ai
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import sphinxai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load .env ────────────────────────────────────────────────────────
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
MAX_ROWS: int = int(os.getenv("MAX_ROWS", "5000"))
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
SPHINX_API_KEY: Optional[str] = os.getenv("SPHINX_API_KEY")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

logger = logging.getLogger("datax")
logging.basicConfig(level=logging.INFO)

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Data-X Backend",
    description="Backend using sphinxai library for data analysis.",
)

# CORS configuration
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


# ── Internal Helpers ──────────────────────────────────────────────────

def _setup_sphinx():
    """Configure sphinxai library pointing to OpenRouter's OpenAI-compatible API."""
    # Prioritize OpenRouter if configured, otherwise fallback to Sphinx (useful for backwards compatibility)
    api_key = OPENROUTER_API_KEY or SPHINX_API_KEY
    if not api_key:
        raise RuntimeError("Neither OPENROUTER_API_KEY nor SPHINX_API_KEY is set.")
    
    # We use the 'openai' provider format points to OpenRouter.
    # We explicitly map S/M/L tiers to minimax/minimax-m2.5.
    sphinxai.set_llm_config(
        provider="openai",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        models={
            "S": "minimax/minimax-m2.5",
            "M": "minimax/minimax-m2.5",
            "L": "minimax/minimax-m2.5"
        }
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness + readiness check."""
    return {
        "ok": True,
        "library": "sphinxai",
        "authenticated": bool(OPENROUTER_API_KEY or SPHINX_API_KEY),
        "using": "openrouter" if OPENROUTER_API_KEY else "sphinx",
        "max_rows": MAX_ROWS,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
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
            detail=f"Payload exceeds limit ({MAX_ROWS} rows).",
        )

    # ── Auth Check ────────────────────────────────────────────────────
    if not SPHINX_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="SPHINX_API_KEY is not configured on the server.",
        )

    # ── Data Processing ───────────────────────────────────────────────
    try:
        df = pd.DataFrame(req.data)
        if req.columns:
            available = [c for c in req.columns if c in df.columns]
            if available:
                df = df[available]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Data conversion error: {exc}",
        )

    # ── SphinxAI Analysis ─────────────────────────────────────────────
    try:
        _setup_sphinx()
        
        # Build prompt with data context. 
        # For small enough datasets, we can include more info.
        # For very large ones, we'd ideally use a different RAG pattern, 
        # but here we keep it simple as a library replacement.
        data_context = df.to_string(index=False, max_rows=20) # Sample 20 rows
        
        prompt = (
            f"Context: Analyze the following data sample from '{req.fileName or 'dataset'}' "
            f"({len(req.data)} total rows).\n\n"
            f"Data:\n{data_context}\n\n"
            f"Question: {req.question}\n\n"
            "Provide a clear, insight-driven response."
        )

        logger.info(f"Submitting to sphinxai.llm: question='{req.question[:50]}...'")
        
        # Call the library
        response_text = await sphinxai.llm(
            prompt=prompt,
            model_size="M",  # Medium tier as requested in docs
            timeout=60.0
        )

        return AnalyzeResponse(
            content=response_text,
            visualizations=[], # Currently returning empty as lib doesn't generate them directly
            isMock=False
        )

    except Exception as exc:
        logger.error(f"SphinxAI Error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}",
        )
