from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Input Contracts ---

class DatasetMetadata(BaseModel):
    filename: Optional[str] = None
    total_rows: int
    columns: List[str]
    dtypes: Dict[str, str]

class CapabilitiesConfig(BaseModel):
    """
    Controla las capacidades extendidas del agente (Sprint 4).
    """
    web_enabled: bool = False
    pdf_parsing_enabled: bool = False
    file_search_enabled: bool = False

class AnalysisRequest(BaseModel):
    """
    Petición estandarizada que el Motor de Data-X entiende.
    Abstrae el origen (HTTP, CLI, Queue).
    """
    query: str
    dataset_metadata: DatasetMetadata
    sample_data: Optional[str] = None # Muestra en texto (ej. markdown head)
    analysis_type: str = "deep" # quick, deep, semantic
    language: str = "es"
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)

# --- Plan Modality Contracts ---

class PlanStep(BaseModel):
    step_number: int
    description: str
    expected_artifact: str = Field(description="e.g., 'filtered_df', 'sales_chart.png'")

class AnalysisPlanResponse(BaseModel):
    """Contrato de salida para POST /analyze/plan"""
    analysis_id: str
    profile_summary: str
    hypothesis: List[str]
    steps: List[PlanStep]
    warnings: List[str]
    estimated_complexity: str = Field(description="Low, Medium, High")

# --- Execution Modality Contracts ---

class ExecutionOptions(BaseModel):
    capabilities: Dict[str, bool] = Field(default_factory=lambda: {"web_enabled": False, "pdf_enabled": False})
    
class ExecuteRequest(BaseModel):
    """Contrato de entrada para POST /analyze/execute"""
    analysis_id: str
    approved_plan: AnalysisPlanResponse
    options: ExecutionOptions

# --- Result & Artifact Contracts ---

class AnalysisArtifact(BaseModel):
    artifact_type: str = Field(description="code, table, chart, report, log")
    content: Any # Could be string code, base64 image, or JSON data

class ExecutionResult(BaseModel):
    """Contrato de salida final para POST /analyze/execute"""
    analysis_id: str
    status: str = Field(description="completed or failed")
    final_answer: str
    artifacts: List[AnalysisArtifact]
    error_log: Optional[str] = None
