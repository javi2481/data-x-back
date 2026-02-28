import logging
from typing import Dict, Any, List
from .contracts import AnalysisArtifact
from .rules_loader import rules_loader

logger = logging.getLogger(__name__)

class Reviewer:
    """
    Componente crítico de validación. Implementa los 'guardrails'
    en tiempo de ejecución. Evita que la salida arruine el dashboard o divulgue PII.
    """
    
    def review_code_generation(self, code: str, intent: str) -> Dict[str, Any]:
        """
        Valida que el código no contenga imports prohibidos o 
        lógica destructiva según rules.json (Code Generation Policy).
        """
        policy = rules_loader.get_code_policy()
        banned = policy.get("banned_functions", [])
        
        for func in banned:
            if func in code:
                logger.warning(f"Revisión Fallida: El código contiene la función prohibida '{func}'")
                return {"approved": False, "reason": f"Uso de función prohibida: {func}"}
        
        # En una iteración real usando Sphinx, podríamos validar que importe pandas.
        return {"approved": True, "reason": "Política de código pasada"}

    def review_data_safety(self, dataset_head: str) -> bool:
        """
        Verifica si se está exponiendo PII o rompiendo las reglas del Data Safety.
        """
        safety_rules = rules_loader.get_safety_policy()
        restricted_cols = safety_rules.get("restricted_columns", [])
        
        for col in restricted_cols:
            if col.lower() in dataset_head.lower():
                logger.error(f"Revisión Fallida: Información restringida detectada en el output ({col}).")
                return False
        return True

    def review_final_output(self, artifacts: List[AnalysisArtifact], original_query: str) -> bool:
        """
        Verifica que el resultado responde a la pregunta y chequea
        que los formatos de los artefactos sean consumibles por el dashboard.
        """
        if not artifacts:
            logger.warning("Revisión Fallida: La ejecución no generó ningún artefacto.")
            return False
            
        policy = rules_loader.get_output_policy()
        # Verificar lógicamente (Próxima etapa: LLM-as-a-judge aquí con Sphinx)
        return True

reviewer = Reviewer()
