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

import numpy as np
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
    model_size: Optional[str] = "M"  # Default to Balanced


class SemanticRequest(BaseModel):
    query: str
    data: List[Dict[str, Any]]


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
    
    # Configure LLM with tiers
    sphinxai.set_llm_config(
        provider="openai", # OpenRouter uses OpenAI format
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        models={
            "S": "google/gemini-2.0-flash-lite-preview-02-05", # Fast (User suggested flash-lite)
            "M": "google/gemini-2.0-flash",                   # Balanced
            "L": "google/gemini-2.0-pro-exp-02-05"            # Deep (User suggested gemini-2.5 pro, using latest stable/exp)
        }
    )
    
    # Configure Embeddings
    sphinxai.set_embedding_config(
        provider="openai",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="text-embedding-3-small"
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
        raise HTTPException(status_code=400, detail="'data' is required.")

    # ── Data Processing ───────────────────────────────────────────────
    try:
        df = pd.DataFrame(req.data)
        if req.columns:
            available = [c for c in req.columns if c in df.columns]
            if available:
                df = df[available]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data conversion error: {exc}")

    # ── SphinxAI Analysis ─────────────────────────────────────────────
    try:
        _setup_sphinx()
        
        # Enriched prompt with describe() and more context
        data_context = df.to_string(index=False, max_rows=100)
        stats = df.describe().to_string()
        
        prompt = (
            f"Question: {req.question}\n\n"
            f"Data Context (sample of {len(req.data)} rows from {req.fileName or 'dataset'}):\n{data_context}\n\n"
            f"Statistics Summary:\n{stats}\n\n"
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
            "If no visualization is needed, return [] for \"visualizations\"."
        )

        logger.info(f"Submitting to sphinxai.llm (Tier: {req.model_size}): {req.question[:50]}...")
        
        response_text = await sphinxai.llm(
            prompt=prompt,
            model_size=req.model_size or "M",
            timeout=90.0
        )

        # ── Parse and Re-encode for Lovable ──────────────────────────
        try:
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            clean_json_str = json_match.group(1) if json_match else response_text
            parsed = json.loads(clean_json_str)
            if "content" not in parsed:
                parsed = {"content": response_text, "visualizations": []}
            final_content = json.dumps(parsed)
        except Exception:
            final_content = json.dumps({"content": response_text, "visualizations": []})

        return AnalyzeResponse(content=final_content, isMock=False)

    except Exception as exc:
        logger.error(f"SphinxAI Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@app.post("/analyze/deep", response_model=AnalyzeResponse)
async def deep_analyze(req: AnalyzeRequest):
    """Deep analysis using batch_llm to process multiple aspects in parallel."""
    try:
        _setup_sphinx()
        df = pd.DataFrame(req.data)
        
        sub_questions = [
            f"Describe la distribución de cada columna numérica basándote en estas estadísticas:\n{df.describe()}",
            f"Identifica outliers y anomalías en esta muestra de datos:\n{df.to_string(max_rows=50)}",
            f"Encuentra correlaciones interesantes entre las columnas del dataset.",
            f"Identifica tendencias temporales si hay columnas que parezcan fechas.",
            f"Proporciona 3 recomendaciones accionables basadas en los insights del negocio encontrados."
        ]
        
        logger.info("Starting Parallel Batch Analysis (5 tasks)...")
        results = await sphinxai.batch_llm(
            sub_questions,
            model_size="M",
            max_concurrent=5
        )
        
        # Consolidation step
        logger.info("Consolidating batch results into Executive Report...")
        consolidated_prompt = (
            f"Consolida estos análisis individuales sobre el archivo {req.fileName or 'dataset'} en un reporte ejecutivo estructurado en markdown:\n\n"
            f"{chr(10).join(results)}\n\n"
            "IMPORTANTE: El reporte debe ser coherente, evitar redundancias y terminar con una sección de conclusiones."
        )
        
        final_report = await sphinxai.llm(consolidated_prompt, model_size="L")
        
        # Wrap for Lovable
        final_content = json.dumps({
            "content": final_report,
            "visualizations": [] # Batch consolidation usually doesn't return unified charts yet
        })
        
        return AnalyzeResponse(content=final_content, isMock=False)

    except Exception as exc:
        logger.error(f"Deep Analysis Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(exc)}")


@app.post("/analyze/semantic")
async def semantic_search(req: SemanticRequest):
    """Semantic search over data rows using embeddings."""
    try:
        _setup_sphinx()
        
        # Vectorize text representation of each row
        texts = [json.dumps(row) for row in req.data]
        logger.info(f"Generating embeddings for {len(texts)} rows...")
        
        embeddings = await sphinxai.batch_embed_text(texts, model_size="S")
        query_embedding = await sphinxai.embed_text(req.query, model_size="S")
        
        # Calculate cosine similarity using numpy
        # Note: We assume embeddings are normalized (standard for OpenAI/OpenRouter)
        embeddings_np = np.array(embeddings)
        query_np = np.array(query_embedding)
        
        similarities = np.dot(embeddings_np, query_np)
        top_indices = np.argsort(similarities)[-10:][::-1]
        
        results = [
            {"row": req.data[i], "score": float(similarities[i])} 
            for i in top_indices if i < len(req.data)
        ]
        
        return {"results": results}

    except Exception as exc:
        logger.error(f"Semantic Search Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(exc)}")
