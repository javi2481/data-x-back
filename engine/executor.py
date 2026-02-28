import logging
from typing import List, Dict, Any
from .contracts import ExecuteRequest, AnalysisRequest, AnalysisArtifact, ExecutionResult
from .context_builder import context_builder
from .reviewer import reviewer
from .result_formatter import result_formatter

logger = logging.getLogger(__name__)

class Executor:
    """
    Motor central de ejecución para los planes aprobados.
    Ciclo por cada paso:
      1. Extrae contexto.
      2. Genera código (agente).
      3. Valida el código (reviewer).
      4. Ejecuta el código en el workspace (aislado preferentemente).
      5. Formatea artefactos (result_formatter).
    """

    async def execute_plan(self, original_req: AnalysisRequest, exec_req: ExecuteRequest) -> ExecutionResult:
        logger.info(f"Iniciando ejecución para el análisis: {exec_req.analysis_id}")
        
        artifacts_collected: List[AnalysisArtifact] = []
        execution_errors: List[str] = []
        
        # 1. Preparar prompts para TODOS los pasos (Batch Processing - Sprint 4 Optimization)
        # En lugar de bloquear la red paso a paso, pre-generamos todo el código concurrentemente
        # con sphinxai.batch_llm() para máxima velocidad.
        step_prompts = []
        for step in exec_req.approved_plan.steps:
            prompt = context_builder.build_execution_context(
                request=original_req,
                options=exec_req.options,
                current_step=step.model_dump()
            )
            step_prompts.append(prompt)
            
        logger.info(f"Enviando {len(step_prompts)} prompts a batch_llm para generación concurrente.")
        
        # 2. Llamada concurrente al LLM (Mockeada por ahora)
        # generated_codes = await sphinxai.batch_llm(step_prompts, model="google/gemini-2.5-pro")
        generated_codes = [self._mock_generate_code(step.description) for step in exec_req.approved_plan.steps]
        
        # 3. Iterar sobre el código ya generado para validación y ejecución segura
        for idx, step in enumerate(exec_req.approved_plan.steps):
            logger.debug(f"Ejecutando paso {step.step_number}: {step.description}")
            generated_code = generated_codes[idx]
            
            # 3a. Revisión Funcional (Reviewer Intercept)
            review_status = reviewer.review_code_generation(generated_code, step.description)
            if not review_status.get("approved"):
                error_msg = f"Paso {step.step_number} falló la revisión: {review_status.get('reason')}"
                logger.error(error_msg)
                execution_errors.append(error_msg)
                break # Cancelamos el flujo si un código es inseguro
            
            # 3b. Ejecución del código Python para generar el artefacto 
            # (Aquí iría subprocess/eval seguro sobre la persistencia local de datos)
            raw_artifact = self._mock_execute_python_code(generated_code, step.expected_artifact)
            
            # 3c. Formateo y tipado duro del artefacto devuelto
            formatted_artifact = result_formatter.format_artifact(
                artifact_name=step.expected_artifact, 
                raw_data=raw_artifact
            )
            artifacts_collected.append(formatted_artifact)
            logger.info(f"Artefacto {step.expected_artifact} generado correctamente.")
            
        
        # Revisión Global Final
        final_review = reviewer.review_final_output(artifacts_collected, original_req.query)
        
        final_status = "completed" if not execution_errors and final_review else "failed"
        
        final_answer = result_formatter.generate_final_summary(
            query=original_req.query, 
            artifacts=artifacts_collected,
            success=(final_status == "completed")
        )
        
        logger.info(f"Ejecución del análisis {exec_req.analysis_id} finalizó con estado: {final_status}")

        return ExecutionResult(
            analysis_id=exec_req.analysis_id,
            status=final_status,
            final_answer=final_answer,
            artifacts=artifacts_collected,
            error_log="\\n".join(execution_errors) if execution_errors else None
        )

    # --- Métodos de Simulación Interna (Para estructurar la llamada sin conectar Sphinx) ---
    def _mock_generate_code(self, intent: str) -> str:
        """Simula que Sphinx devolvió un bloque de código Python"""
        return f"import pandas as pd\\n# Código generado para: {intent}\\nprint('Done')"
        
    def _mock_execute_python_code(self, code: str, expected_name: str) -> Dict[str, Any]:
        """Simula la ejecución segura y retorno de datos en crudo"""
        # Dependiendo del sufijo del artefacto, devolvemos un payload mock diferente
        if expected_name.endswith(".png"):
            return {"type": "chart", "content": "base64_encoded_png_mock_data"}
        elif expected_name.startswith("df_") or expected_name.endswith(".csv"):
            return {"type": "table", "content": [{"col1": "val1"}, {"col1": "val2"}]}
        else:
            return {"type": "report", "content": "Markdown text based on analysis."}


executor = Executor()
