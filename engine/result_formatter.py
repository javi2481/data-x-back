from typing import Dict, Any, List
from .contracts import AnalysisArtifact
from .rules_loader import rules_loader

class ResultFormatter:
    """
    Se encarga de estandarizar la salida cruda de la ejecución 
    para que FastAPI devuelva contratos perfectos al frontend dashboard.
    """
    
    def format_artifact(self, artifact_name: str, raw_data: Dict[str, Any]) -> AnalysisArtifact:
        """
        Envuelve el resultado de python/llm en el contrato `AnalysisArtifact`.
        Asegura que el tipo de artefacto coincida con las políticas de interfaz.
        """
        output_policy = rules_loader.get_output_policy()
        
        # Deducción rudimentaria del tipo de artefacto
        artifact_type = "report" # Default
        if raw_data.get("type") == "chart":
            artifact_type = f"chart_{output_policy.get('default_chart_type', 'png')}"
        elif raw_data.get("type") == "table":
            artifact_type = "data_table"
            
        return AnalysisArtifact(
            artifact_type=artifact_type,
            content=raw_data.get("content", {})
        )
        
    def generate_final_summary(self, query: str, artifacts: List[AnalysisArtifact], success: bool) -> str:
        """
        Genera el texto final de la respuesta, resumiendo lo que se generó.
        """
        if not success:
            return "Lo siento, hubo un error crítico durante la ejecución técnica de tu consulta y no pudo ser completada. Revisa los logs de errores."
            
        md_response = f"Análisis completado para: *{query}*\\n\\n"
        md_response += "### Artefactos Generados:\\n"
        for i, art in enumerate(artifacts):
            md_response += f"{i+1}. [{art.artifact_type.upper()}] - Generado correctamente.\\n"
            
        return md_response

result_formatter = ResultFormatter()
