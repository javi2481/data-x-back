from typing import Dict, Any
from .contracts import AnalysisRequest, ExecutionOptions
from .rules_loader import rules_loader

class ContextBuilder:
    """
    Construye el string de contexto unificado (el 'System Prompt' extendido) 
    que el motor Sphinx consumirá. 
    Crucial para inyectar políticas, estado y datos reales en cada paso.
    """
    def build_planning_context(self, request: AnalysisRequest) -> str:
        """Contexto específico para la fase de creación de planes."""
        rules = rules_loader.get_rules()
        
        context = f"""
Eres Data-X Planner, un agente senior de análisis de datos impulsado por SphinxAI.
Tu tarea actual es DISEÑAR UN PLAN de ejecución paso a paso para responder a la siguiente consulta del usuario.
NO debes ejecutar código todavía. Solo planificar.

## [CONSULTA DEL USUARIO]
{request.query}

## [METADATOS DEL DATASET]
* Nombre: {request.dataset_metadata.filename}
* Filas Totales: {request.dataset_metadata.total_rows}
* Columnas: {', '.join(request.dataset_metadata.columns)}
* Tipos de Datos: {request.dataset_metadata.dtypes}

## 3. Capabilities Previstas (Sprint 4)
Este requerimiento tiene los siguientes módulos externos activados:
- Web Search (Internet): {'Habilitado' if request.capabilities.web_enabled else 'Deshabilitado'}
- PDF Parsing: {'Habilitado' if request.capabilities.pdf_parsing_enabled else 'Deshabilitado'}
- File Search: {'Habilitado' if request.capabilities.file_search_enabled else 'Deshabilitado'}

## [MUESTRA REAL]
{request.sample_data}

## [POLÍTICAS OBLIGATORIAS DE DATA-X]
(Debes asegurar que tu plan respete estas reglas operativas)

* Políticas de Seguridad: {rules.get('data_safety', {})}
* Políticas de Análisis: {rules.get('analysis_policy', {})}
* Políticas de Generación de Código: {rules.get('code_generation_policy', {})}
* Políticas de Formato: {rules.get('output_formatting_policy', {})}

## [DICCIONARIO DE ARTEFACTOS ESPERADOS]
En tu plan, debes listar qué artefactos generará cada paso.
Ejemplos válidos: 'filtered_df' (datos tabulares), 'trend_chart.png' (gráfico visual), 'summary_markdown' (reporte de texto).

## [RESTRICCIÓN DE MODALIDAD]
Operas en MODO PLAN. Elabora pasos lógicos, verificables y orientados a resultados basados en la muestra provista, las reglas cargadas y las capacidades de las librerías permitidas.
"""
        return context

    def build_execution_context(self, request: AnalysisRequest, options: ExecutionOptions, current_step: Dict[str, Any]) -> str:
        """Contexto específico para la generación de código/ejecución de un paso aprobado."""
        rules = rules_loader.get_rules()
        capabilities_str = "Habilitado: " + ", ".join([k for k, v in options.capabilities.items() if v]) if any(options.capabilities.values()) else "Ninguna capacidad externa habilitada."

        context = f"""
Eres Data-X Executor, un agente impulsado por SphinxAI.
Tu tarea es EJECUTAR UN PASO ESPECÍFICO de un plan de análisis ya aprobado.

## [PASO ACTUAL A EJECUTAR]
Paso {current_step.get('step_number')}: {current_step.get('description')}
Artefacto Esperado: {current_step.get('expected_artifact')}

## [METADATOS DEL ENTORNO]
* Filas: {request.dataset_metadata.total_rows}
* Tipos de Datos: {request.dataset_metadata.dtypes}
* Capacidades Extra del Engine: {capabilities_str}

## [POLÍTICAS ESTRICTAS DE GENERACIÓN (rules.json)]
* Librerías Permitidas: {rules.get('code_generation_policy', {}).get('allowed_libraries')}
* Funciones Prohibidas: {rules.get('code_generation_policy', {}).get('banned_functions')}
* Tema Gráfico: {rules.get('output_formatting_policy', {}).get('chart_theme')}

Escribe el código Python puro necesario para completar el PASO ACTUAL y generar el '{current_step.get('expected_artifact')}'. 

## [REGLAS DEL SANDBOX DE EJECUCIÓN OBLIGATORIAS]
1. La variable `df` YA ESTÁ DECLARADA y cargada en memoria con todo tu dataset. NO intentes leer archivos CSV, simplemente asume que `df` existe.
2. NO modifiques el DataFrame `df` original en memoria.
3. Si el Artefacto Esperado termina en '.png' (es un gráfico), tu código DEBE guardar la imagen usando `plt.savefig()` o similar en el directorio actual.
4. Si el Artefacto Esperado demanda texto json o datos resumidos, simplemente impímelos a consola con `print()`.
5. Envuelve de forma estricta todo tu código Python final en bloques ```python\n ... \n```
"""
        return context

# Singleton para uso en el motor
context_builder = ContextBuilder()
