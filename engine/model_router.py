"""
Enrutador dinámico de modelos de Inteligencia Artificial (LLM).
Analiza el contexto, el tamaño de los datos y la intención del paso para elegir
el modelo más eficiente (en costo y velocidad) y preciso para el momento del ciclo de vida.
"""
import logging
from typing import Dict, Any, Optional
from .contracts import AnalysisRequest

logger = logging.getLogger(__name__)

class ModelRouter:
    """
    Decide en tiempo de ejecución de entre los modelos configurados en main.py.
    """
    
    def get_planner_model(self, request: AnalysisRequest) -> str:
        """El Arquitecto: Determina el mejor modelo para entender el problema y crear el plan."""
        meta = request.dataset_metadata
        if not meta:
            return "M"
            
        # Heurísticas de complejidad: muchas columnas o pregunta analítica muy larga
        num_cols = len(meta.columns) if meta.columns else 0
        if num_cols > 20 or len(request.query) > 150:
            logger.info(f"ModelRouter: Seleccionando planificador COMPLEX (Columnas: {num_cols}, Longitud Query: {len(request.query)})")
            return "M"
            
        logger.info("ModelRouter: Seleccionando planificador FAST.")
        return "S"

    def get_executor_model(self, step_description: str) -> str:
        """El Ingeniero: Determina el mejor modelo para escribir código Python según la tarea."""
        complex_keywords = ["machine learning", "modelo", "regresión", "predicción", "pca", "estadística", "correlación", "complejo", "ml"]
        desc_lower = step_description.lower()
        
        if any(kw in desc_lower for kw in complex_keywords):
            logger.info(f"ModelRouter: Seleccionando ejecutor SURGEON (L) para tarea de Stats/ML: {step_description[:30]}...")
            return "L"
            
        # Default para agrupaciones estándar, transformaciones y gráficos
        return "S"

    def get_reviewer_model(self) -> str:
        """El Guardia: Devuelve el modelo más barato/rápido para escaneos semánticos básicos de seguridad."""
        return "S"

    def get_fixer_model(self) -> str:
        """El Cirujano Fixer: Devuelve el modelo más capaz del mundo para entender Tracebacks y reparar código roto."""
        return "L"

model_router = ModelRouter()
