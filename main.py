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
import json
import re
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
    # Attempt to find the API key in all possible variants
    # Note: SPHINX_LLM_API_KEY is the internal name used by the library for OpenAI-format keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    sphinx_llm_key = os.getenv("SPHINX_LLM_API_KEY")
    sphinx_api_key = os.getenv("SPHINX_API_KEY")
    
    api_key = openrouter_key or sphinx_llm_key or sphinx_api_key
    
    if not api_key:
        raise RuntimeError(
            "No API Key found. Please set OPENROUTER_API_KEY, "
            "SPHINX_LLM_API_KEY, or SPHINX_API_KEY in environment."
        )
    
    # Log which variable we are using (safe for security as it doesn't log the value)
    if openrouter_key:
        logger.info("Using auth token from OPENROUTER_API_KEY")
    elif sphinx_llm_key:
        logger.info("Using auth token from SPHINX_LLM_API_KEY")
    else:
        logger.info("Using auth token from SPHINX_API_KEY")
    
    # We use the 'openai' provider format pointing to OpenRouter.
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
            f"Question: {req.question}\n\n"
            f"Data Context (sample of {len(req.data)} rows from {req.fileName or 'dataset'}):\n{data_context}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the data and answer the question.\n"
            "2. If appropriate, create one or more visualizations using Vega-Lite schema.\n"
            "3. MANDATORY: Respond ONLY with a valid JSON object in this format:\n"
            "{\n"
            "  \"content\": \"Your detailed analysis here (markdown supported)\",\n"
            "  \"visualizations\": [\n"
            "    {\n"
            "      \"type\": \"vega-lite\",\n"
            "      \"specification\": { ... Vega-Lite JSON spec ... }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "If no visualization is needed, return an empty array for \"visualizations\"."
        )

        logger.info(f"Submitting to sphinxai.llm: question='{req.question[:50]}...'")
        
        # Call the library
        response_text = await sphinxai.llm(
            prompt=prompt,
            model_size="M",
            timeout=60.0
        )

        # ── Parse and Re-encode for Lovable ──────────────────────────
        # Lovable's Edge Function expects a "double-encoded" response:
        # The 'content' field should be a JSON string containing {content, visualizations}.
        
        try:
            # 1. Try to extract clean JSON from LLM response
            # (LLM might wrap it in ```json ... ```)
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            clean_json_str = json_match.group(1) if json_match else response_text
            
            # 2. Validate it's valid JSON
            parsed = json.loads(clean_json_str)
            
            # 3. Ensure we have the required structure
            if "content" not in parsed:
                parsed = {"content": response_text, "visualizations": []}
            
            # Return it stringified in the 'content' field
            final_content = json.dumps(parsed)
            
        except Exception as exc:
            # Fallback: if parsing fails, wrap the raw text
            logger.warning(f"Failed to parse LLM response as JSON: {exc}")
            final_content = json.dumps({
                "content": response_text,
                "visualizations": []
            })

        return AnalyzeResponse(
            content=final_content,
            visualizations=[], # The Edge Function will populate this from 'content'
            isMock=False
        )

    except Exception as exc:
        logger.error(f"SphinxAI Error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}",
        )
