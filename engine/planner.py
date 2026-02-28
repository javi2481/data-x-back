import json
import uuid
import re
from typing import Dict, Any, List

import sphinxai
from .contracts import AnalysisRequest, AnalysisPlanResponse, PlanStep
from .context_builder import context_builder
from .model_router import model_router
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

        # 3. Elección Dinámica del Modelo Planificador
        planner_model = model_router.get_planner_model(request)
        
        # 4. Llamada REAL al motor Sphinx
        logger.debug(f"Prompt enviado a Sphinx (Modelo: {planner_model}):\n{full_system_prompt[:200]}...")
        
        try:
            response_text = await sphinxai.llm(
                prompt=request.query, 
                system=full_system_prompt, 
                model_size=planner_model,
                timeout=120.0 # Planning can take a while 
            )
            
            # Limpiar posible markdown en la respuesta
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            clean_json_str = json_match.group(1) if json_match else response_text
            parsed_response = json.loads(clean_json_str)

            # 5. Parsear y Validar con el Contrato Pydantic
            steps = [PlanStep(**step_data) for step_data in parsed_response.get("steps", [])]
            
            plan_response = AnalysisPlanResponse(
                analysis_id=str(uuid.uuid4()), # Generamos un ID de análisis único para la persistencia futura
                profile_summary=parsed_response.get("profile_summary", "Diagnóstico no provisto"),
                hypothesis=parsed_response.get("hypothesis", []),
                steps=steps,
                warnings=parsed_response.get("warnings", []),
                estimated_complexity=parsed_response.get("estimated_complexity", "Media")
            )
            
        except Exception as e:
            logger.error(f"Falla grave al planificar con el modelo {planner_model}: {e}\nResponse: {response_text if 'response_text' in locals() else 'N/A'}")
            raise RuntimeError(f"El LLM Planificador falló: {e}")

        logger.info(f"Plan generado exitosamente con ID: {plan_response.analysis_id} y {len(plan_response.steps)} pasos.")
        return plan_response

planner = Planner()
