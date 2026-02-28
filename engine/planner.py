import json
import uuid
from typing import Dict, Any, List
# from sphinxai import llm # Se usará Sphinx nativo aquí en el futuro real
from .contracts import AnalysisRequest, AnalysisPlanResponse, PlanStep
from .context_builder import context_builder
import logging

logger = logging.getLogger(__name__)

class Planner:
    """
    Recibe el contexto provisto por context_builder y llama a SphinxAI 
    para estructurar un plan de análisis (Modalidad Plan).
    """
    
    async def create_plan(self, request: AnalysisRequest) -> AnalysisPlanResponse:
        """Genera un plan de acción basado en la solicitud del usuario (sin ejecutar código)."""
        logger.info(f"Iniciando generación de plan para: '{request.query}'")
        
        # 1. Construir el mega-prompt guiado por reglas
        planning_context = context_builder.build_planning_context(request)
        
        # 2. Instrucciones de tipado JSON fuerte para Sphinx
        json_schema_prompt = """
        DEBES responder EXCLUSIVAMENTE con un objeto JSON válido que siga esta estructura exacta:
        {
            "profile_summary": "Un breve diagnostico técnico de los datos según la muestra",
            "hypothesis": ["Hipótesis 1 a validar con los datos", "Hipótesis 2..."],
            "steps": [
                {"step_number": 1, "description": "Acción específica", "expected_artifact": "df_limpio"},
                {"step_number": 2, "description": "Otra acción", "expected_artifact": "grafico.png"}
            ],
            "warnings": ["Advertencias de riesgo sobre los datos (ej: faltantes, PII) o políticas"],
            "estimated_complexity": "Baja, Media o Alta"
        }
        """
        
        full_system_prompt = f"{planning_context}\n\n{json_schema_prompt}"

        # 3. Llamada al motor Sphinx (Mockeado momentáneamente para estructurar el backend, luego se usa sphinxai.llm())
        # En el código final esto sería: response = await sphinxai.llm(prompt=request.query, system=full_system_prompt, format="json")
        logger.debug(f"Prompt enviado a Sphinx (Simulado):\n{full_system_prompt}")
        
        # FIXME: Reemplazar este mock fijo con la llamada real a la librería sphinxai
        mocked_sphinx_response = {
            "profile_summary": f"Dataset de {request.dataset_metadata.total_rows} filas con columnas {request.dataset_metadata.columns}",
            "hypothesis": ["Existen valores atípicos", "Existen tendencias temporales"],
            "steps": [
                {"step_number": 1, "description": f"Limpieza de datos y casting tipo según {request.dataset_metadata.dtypes}", "expected_artifact": "df_cleaned"},
                {"step_number": 2, "description": f"Análisis y agrupación para responder a: {request.query}", "expected_artifact": "df_grouped"},
                {"step_number": 3, "description": "Generación de gráfico o reporte final", "expected_artifact": "final_visualization.png"}
            ],
            "warnings": ["La muestra provista es pequeña", "Verificar posibles nulos en la ejecución"],
            "estimated_complexity": "Media"
        }

        # 4. Parsear y Validar con el Contrato Pydantic
        steps = [PlanStep(**step_data) for step_data in mocked_sphinx_response["steps"]]
        
        plan_response = AnalysisPlanResponse(
            analysis_id=str(uuid.uuid4()), # Generamos un ID de análisis único para la persistencia futura
            profile_summary=mocked_sphinx_response["profile_summary"],
            hypothesis=mocked_sphinx_response["hypothesis"],
            steps=steps,
            warnings=mocked_sphinx_response["warnings"],
            estimated_complexity=mocked_sphinx_response["estimated_complexity"]
        )

        logger.info(f"Plan generado exitosamente con ID: {plan_response.analysis_id} y {len(plan_response.steps)} pasos.")
        return plan_response

planner = Planner()
