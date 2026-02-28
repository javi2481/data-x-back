import os
import sys
import json
import logging
import tempfile
import base64
import subprocess
from typing import List, Dict, Any, Optional

import sphinxai
from .contracts import ExecuteRequest, AnalysisRequest, AnalysisArtifact, ExecutionResult
from .context_builder import context_builder
from .reviewer import reviewer
from .result_formatter import result_formatter
from .model_router import model_router

logger = logging.getLogger(__name__)

class Executor:
    """
    Motor central de ejecución para los planes aprobados con inyección real de SphinxAI
    y sandbox de ejecución (subprocess) para aislamiento.
    """

    async def execute_plan(self, original_req: AnalysisRequest, exec_req: ExecuteRequest) -> ExecutionResult:
        logger.info(f"Iniciando ejecución real para el análisis: {exec_req.analysis_id}")
        
        artifacts_collected: List[AnalysisArtifact] = []
        execution_errors: List[str] = []
        
        # 1. Preparar prompts para TODOS los pasos
        step_prompts = []
        for step in exec_req.approved_plan.steps:
            prompt = context_builder.build_execution_context(
                request=original_req,
                options=exec_req.options,
                current_step=step.model_dump()
            )
            step_prompts.append(prompt)
            
        logger.info(f"Enviando {len(step_prompts)} prompts a sphinxai.batch_llm para generación concurrente.")
        
        # 2. Elegir el modelo adecuado para ejecución generica/batch (La mayoría de los pasos)
        # Podríamos rutear cada paso si sphinxai soportara batch mixto, 
        # pero usamos el Batch Executor general para la cola principal.
        exec_model = "ROUTER_EXEC_BATCH" 
        for step in exec_req.approved_plan.steps:
            if model_router.get_executor_model(step.description) == "ROUTER_EXEC_SURGEON":
                exec_model = "ROUTER_EXEC_SURGEON" # Upgrade whole batch to Surgeon if needed
                break
                
        logger.info(f"Llamada concurrente usando el Ejecutor: {exec_model}")
        try:
            generated_codes = await sphinxai.batch_llm(step_prompts, model_size=exec_model)
        except Exception as e:
            logger.error(f"Fallo grave en generacion de código SphinxAI: {e}")
            return ExecutionResult(
                analysis_id=exec_req.analysis_id,
                status="failed",
                final_answer="La IA no pudo generar el código para este plan.",
                artifacts=[],
                error_log=str(e)
            )
            
        # 3. Iterar sobre el código ya generado para validación y ejecución segura
        for idx, step in enumerate(exec_req.approved_plan.steps):
            logger.info(f"Procesando paso {step.step_number}: {step.description}")
            current_code = self._clean_llm_code(generated_codes[idx])
            
            # 3a. Revisión Funcional (Reviewer Intercept)
            review_status = await reviewer.review_code_generation(current_code, step.description)
            if not review_status.get("approved"):
                error_msg = f"Revisión de seguridad fallida en Paso {step.step_number}: {review_status.get('reason')}"
                logger.error(error_msg)
                execution_errors.append(error_msg)
                break 

            # 3b. Ejecución de Código en Sandbox con Self-Correction Loop
            max_retries = 2
            success = False
            raw_artifact = None
            
            for attempt in range(max_retries + 1):
                raw_artifact, exec_err = self._execute_python_sandbox(
                    code=current_code, 
                    dataset_path=original_req.dataset_metadata.filename,
                    expected_artifact=step.expected_artifact
                )
                
                if not exec_err:
                    success = True
                    break
                    
                logger.warning(f"Paso {step.step_number} falló en ejecución (Intento {attempt+1}/{max_retries+1}): {exec_err[:200]}...")
                
                if attempt < max_retries:
                    # Self-Correction Loop: Enviar el stacktrace al LLM para arreglar el código
                    fix_prompt = (
                        f"Tu código anterior para '{step.description}' falló con este error:\n"
                        f"{exec_err}\n\n"
                        f"Código original:\n```python\n{current_code}\n```\n\n"
                        "Por favor, devuelve ÚNICAMENTE el código Python corregido sin explicaciones."
                    )
                    logger.info(f"Lanzando Self-Correction Loop para paso {step.step_number}...")
                    fixer_model = model_router.get_fixer_model()
                    corrected_code_raw = await sphinxai.llm(fix_prompt, model_size=fixer_model)
                    current_code = self._clean_llm_code(corrected_code_raw)
                else:
                    error_msg = f"Paso {step.step_number} falló tras {max_retries} intentos de corrección. Error: {exec_err}"
                    execution_errors.append(error_msg)
            
            if not success:
                break # Cancelamos el resto del plan si un paso es incobrable
            
            # 3c. Formateo y tipado duro del artefacto devuelto
            formatted_artifact = result_formatter.format_artifact(
                artifact_name=step.expected_artifact, 
                raw_data=raw_artifact
            )
            artifacts_collected.append(formatted_artifact)
            logger.info(f"Artefacto {step.expected_artifact} generado correctamente.")
            
        # 4. Revisión Global Final
        final_review = reviewer.review_final_output(artifacts_collected, original_req.query)
        final_status = "completed" if not execution_errors and final_review else "failed"
        
        final_answer = result_formatter.generate_final_summary(
            query=original_req.query, 
            artifacts=artifacts_collected,
            success=(final_status == "completed")
        )
        
        return ExecutionResult(
            analysis_id=exec_req.analysis_id,
            status=final_status,
            final_answer=final_answer,
            artifacts=artifacts_collected,
            error_log="\n".join(execution_errors) if execution_errors else None
        )

    # --- Métodos de Sandboxing ---

    def _clean_llm_code(self, raw_text: str) -> str:
        """Limpia la respuesta del LLM para extraer solo el código Python"""
        if "```python" in raw_text:
            return raw_text.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_text:
            return raw_text.split("```")[1].split("```")[0].strip()
        return raw_text.strip()

    def _execute_python_sandbox(self, code: str, dataset_path: str, expected_artifact: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Ejecuta el código generado en un proceso aislado local (subprocess).
        Lee el dataset desde el disco. Devuelve (resultado, error).
        """
        # Inyección de variable de entorno DataFrame_PATH
        injected_code = f"import pandas as pd\nimport numpy as np\nDATASET_PATH = r'{dataset_path}'\ndf = pd.read_csv(DATASET_PATH)\n\n" + code
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "agent_exec.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(injected_code)
                
            # Ejecutamos el subprocess. Timeout=30s para evitar loops infinitos.
            try:
                result = subprocess.run(
                    [sys.executable, script_path], 
                    cwd=temp_dir, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                if result.returncode != 0:
                    return None, result.stderr

                # Si el código se ejecutó sin romper, recolectamos el output base
                # Si generó una imagen (esperamos .png) la encodeamos
                if expected_artifact.endswith(".png"):
                    found_img = False
                    for file in os.listdir(temp_dir):
                        if file.endswith(".png"):
                            with open(os.path.join(temp_dir, file), "rb") as img_f:
                                b64 = base64.b64encode(img_f.read()).decode()
                                return {"type": "chart", "content": b64}, None
                    if not found_img:
                        return None, "El código ejecutó correctamente pero no guardó ningún archivo .png"
                        
                # Si el código tenía que devolver datos/json, lo leemos de stdout o archivos csv
                output_str = result.stdout.strip()
                if expected_artifact.startswith("df_") or expected_artifact.endswith(".csv"):
                    return {"type": "table", "content": [{"sandbox_output": output_str}]}, None
                
                return {"type": "report", "content": output_str or "Execution success without standard output"}, None
                
            except subprocess.TimeoutExpired:
                return None, "El código excedió el tiempo límite de ejecución (30s)."
            except Exception as e:
                return None, f"Error del sistema en el sandbox: {str(e)}"

executor = Executor()
