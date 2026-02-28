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
import hashlib
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import sphinxai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.db import db_manager
from engine.repository import repository

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

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Data-X Application...")
    db_manager.connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Data-X Application...")
    db_manager.close_db()

# ── Pydantic models ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    question: str
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None
    fileName: Optional[str] = None
    model_size: Optional[str] = "M"  # Default to Balanced
    stats_summary: Optional[Dict[str, Any]] = None
    category_summary: Optional[Dict[str, Any]] = None
    total_rows: Optional[int] = None
    sampled_rows: Optional[int] = None


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
            "ROUTER_PLANNER_COMPLEX": "anthropic/claude-3.7-sonnet", # Deep reasoning 
            "ROUTER_PLANNER_FAST": "google/gemini-2.5-pro",          # Fast planning
            "ROUTER_EXEC_BATCH": "google/gemini-2.0-flash-001",      # Concurrent code gen
            "ROUTER_EXEC_SURGEON": "anthropic/claude-3.5-sonnet",    # Complex code gen
            "ROUTER_AUDITOR_FAST": "qwen/qwen3.5-flash-02-23",       # Cheap & fast guardrail
            "ROUTER_SMART_FIXER": "openai/gpt-4.5-preview"           # Self-correction king
        }
    )
    
    # Configure Embeddings
    sphinxai.set_embedding_config(
        provider="openai",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        models={"S": "text-embedding-3-small"}
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
        
        # Prepare sample context and basic stats first
        data_context = df.to_string(index=False, max_rows=100)
        stats = df.describe().to_string()
        
        # --- Phase 2: Enriched Context ---
        context_parts = []
        effective_total = req.total_rows if req.total_rows is not None else len(req.data)
        effective_sampled = req.sampled_rows if req.sampled_rows is not None else len(req.data)
        
        if req.stats_summary:
            context_parts.append(f"Dataset has {effective_total} rows (sample of {effective_sampled}).")
            context_parts.append(f"Detailed Statistics (percentiles): {json.dumps(req.stats_summary, indent=2)}")
        
        if req.category_summary:
            context_parts.append(f"Categorical Distribution: {json.dumps(req.category_summary, indent=2)}")
        
        extra_context = "\n".join(context_parts)
        
        prompt = (
            f"Question: {req.question}\n\n"
            f"Data Context (sample of {effective_sampled} rows from {req.fileName or 'dataset'}):\n{data_context}\n\n"
            f"Quick Stats Summary:\n{stats}\n\n"
        )
        
        if extra_context:
            prompt += f"EXTENDED CONTEXT:\n{extra_context}\n\n"
            
        prompt += (
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


# ── Phase 1 Integration: Engine Orchestration ───────────────────────

# Note: Old /analyze/deep logic was removed in favor of the new modal engine.

from engine.contracts import AnalysisRequest, DatasetMetadata, AnalysisPlanResponse, ExecuteRequest
from engine.planner import planner
from engine.executor import executor

@app.post("/analyze/plan", response_model=AnalysisPlanResponse)
async def analyze_plan(req: AnalyzeRequest):
    """
    Fase 1 del Motor Modal: Genera un plan estructurado basado en la solicitud.
    No ejecuta el código, solo propone el camino para revisión o frontend.
    """
    try:
        _setup_sphinx()
        
        # 1. Adaptar el Request de FastAPI al Contrato del Motor
        # Esto previene que el engine dependa de detalles de implementación HTTP
        
        # Fast conversion for sample context
        try:
            df = pd.DataFrame(req.data)
            sample_data = df.head(10).to_markdown() # Using markdown for leaner LLM context
            
            # Extract datatypes
            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Data parsing error: {exc}")

        analysis_id = str(uuid.uuid4())
        
        # Save exact dataset to local temporal vault for future secure execution
        dataset_path = os.path.join(os.getcwd(), "_datasets", f"{analysis_id}.csv")
        df.to_csv(dataset_path, index=False)
        
        dataset_meta = DatasetMetadata(
            filename=dataset_path,
            total_rows=req.total_rows if req.total_rows is not None else len(req.data),
            columns=req.columns if req.columns else list(df.columns),
            dtypes=dtypes_dict
        )
        
        engine_request = AnalysisRequest(
            query=req.question,
            dataset_metadata=dataset_meta,
            sample_data=sample_data,
            analysis_type="deep", # Inherited from the old context
            language="es"
        )
        
        # 2a. Crear tracking de persistencia en DB (Estado: planning)
        await repository.create_analysis(analysis_id, engine_request)
        
        # 2b. Orquestar llamado al Motor
        logger.info(f"Orquestando /analyze/plan vía motor (ID {analysis_id}) para query: {req.question[:30]}...")
        plan_response = await planner.create_plan(engine_request)
        
        # 2c. Guardar el plan propuesto (Estado: planned)
        await repository.update_plan(analysis_id, plan_response)
        
        # 3. Retornar plan al frontend, inyectándole el analysis_id para seguimiento
        response_dict = plan_response.model_dump()
        response_dict["analysis_id"] = analysis_id
        return response_dict

    except Exception as exc:
        logger.error(f"Plan Orchestration Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(exc)}")


class ApprovePlanRequest(BaseModel):
    analysis_id: str


@app.post("/analyze/approve")
async def approve_plan(req: ApprovePlanRequest):
    """
    Fase 2.5: El usuario revisó el plan y lo aprueba.
    Cambiamos estado a 'approved'.
    """
    doc = await repository.get_analysis(req.analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis ID no encontrado")
    
    if doc.get("status") != "planned":
        raise HTTPException(status_code=400, detail="El plan debe estar en estado 'planned' para ser aprobado")
        
    await repository.mark_approved(req.analysis_id)
    return {"status": "approved", "analysis_id": req.analysis_id}


class ExecutePlanRequest(BaseModel):
    analysis_id: str
    options: Optional[Dict[str, Any]] = None


@app.post("/analyze/execute")
async def execute_plan(req_data: ExecutePlanRequest):
    """
    Fase 3 del Motor Modal: Pone en marcha la ejecución. Lee el plan desde la BD,
    lo pasa al motor de ejecución, y guarda los artefactos generados.
    """
    try:
        analysis_id = req_data.analysis_id
        
        # Re-hidratar request desde DB
        doc = await repository.get_analysis(analysis_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Analysis ID no encontrado")
            
        status = doc.get("status")
        # if status not in ["planned", "approved"]: 
        # By default we can enforce approval, but for seamless UI maybe we can auto-execute if 'planned'
        # Let's enforce it strictly for agentic safety:
        if status not in ["planned", "approved"]:
            raise HTTPException(status_code=400, detail=f"Estado invalido para ejecución: {status}. Requiere 'planned' o 'approved'.")

        await repository.mark_running(analysis_id)
        
        # Model hydration
        from engine.contracts import ExecuteRequest
        original_req = AnalysisRequest(**doc["request"])
        exec_req = ExecuteRequest(
            analysis_id=analysis_id,
            approved_plan=AnalysisPlanResponse(**doc["plan"]),
            options=req_data.options
        )
        
        logger.info(f"Orquestando /analyze/execute para ID: {analysis_id}")
        
        execution_result = await executor.execute_plan(original_req, exec_req)
        
        # Guardar en BD (Estado final 'completed' o 'failed')
        await repository.save_execution_result(analysis_id, execution_result)
        
        return execution_result.model_dump()
        
    except Exception as exc:
        logger.error(f"Execute Orchestration Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Plan execution failed: {str(exc)}")



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
