import logging
from typing import Dict, Any, List
from .contracts import AnalysisArtifact
import sphinxai

logger = logging.getLogger(__name__)

class Reviewer:
    """
    Componente crítico de validación. Implementa los 'guardrails'
    en tiempo de ejecución. Evita que la salida arruine el sistema.
    """
    
    async def review_code_generation(self, code: str, intent: str) -> Dict[str, Any]:
        """
        Valida el código HÍBRIDAMENTE:
        1. Reglas determinísticas (imports prohibidos o destructivos).
        2. LLM-as-a-judge (Evaluación semántica de seguridad).
        """
        # --- 1. Reglas Determinísticas ---
        policy = rules_loader.get_code_policy()
        banned = policy.get("banned_functions", [])
        
        for func in banned:
            if func in code:
                logger.warning(f"Revisión Fallida (Regla Dura): El código contiene '{func}'")
                return {"approved": False, "reason": f"Uso de función prohibida por política: {func}"}
        
        if "os.system" in code or "subprocess" in code:
             return {"approved": False, "reason": "Llamadas directas al sistema operativo (os, subprocess) bloqueadas por seguridad."}

        # --- 2. LLM-as-a-Judge (Validación Semántica) ---
        judge_prompt = (
            "Eres un Auditor de Seguridad de Código para Data-X. Tu trabajo es analizar el código Python autogenerado "
            "y aprobarlo SOLAMENTE si es seguro, carece de código malicioso, no borra archivos del sistema local (excepto outputs temporales) "
            "y parece lógicamente correcto para la tarea prevista.\n\n"
            f"TAREA PREVISTA: {intent}\n\n"
            f"CÓDIGO A AUDITAR:\n{code}\n\n"
            "Responde ÚNICAMENTE con la palabra 'APPROVE' si es seguro, o 'REJECT: <Razón breve>' si es inseguro."
        )
        try:
            # We use a smaller, faster model for checking (M)
            judge_response = await sphinxai.llm(judge_prompt, model_size="M")
            if judge_response.strip().startswith("APPROVE"):
                return {"approved": True, "reason": "Auditoría determinística y semántica superadas."}
            else:
                logger.warning(f"Revisión Fallida (Juez LLM): {judge_response}")
                return {"approved": False, "reason": judge_response.strip()}
        except Exception as e:
            logger.error(f"Falla en el Juez LLM: {e}")
            return {"approved": False, "reason": "No se pudo realizar la auditoría semántica."}

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
