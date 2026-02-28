# Sugerencias y Plan de Mejoras Futuras para Data-X

Basado en la arquitectura centralizada de SphinxAI, este es el roadmap ordenado por prioridad arquitectónica y de entrega de valor real.

El plan se divide en cuatro fases estrictas para evitar dispersión, asegurando primero el "valor duro" de la IA y luego construyendo la experiencia de producto a su alrededor.

## Fase 1 — Valor Real del Motor (Prioridad Máxima)

El objetivo de esta fase es que el backend entregue inteligencia verdadera y confiable antes de pulir la experiencia del usuario final.

1.  **Inyección Real de SphinxAI (Crítico)**
    - **Acción:** Reemplazar los métodos `_mock_generate_code` y `_mock_execute_python_code` en `executor.py` por llamadas reales a la suite de `sphinxai`.
    - **Impacto:** Es el paso fundamental. Sin esto, Data-X no entrega su valor central. Conecta el motor con Gemini 2.5 Flash / Pro para razonamiento profundo real.
2.  **Entorno de Ejecución Seguro (Crítico / Prerequisito)**
    - **Acción:** Definir e implementar el aislamiento para la ejecución del código Python generado por el agente (ej. proceso aislado, contenedor efímero tipo Docker, filesystem restringido, timeouts, límites de memoria).
    - **Impacto:** Si vamos a ejecutar código generado por IA, el aislamiento no es una mejora futura, es una obligación de diseño del núcleo del sistema.
3.  **Reviewer Híbrido (Prioridad Alta)**
    - **Acción:** En `reviewer.py`, mantener las _reglas determinísticas_ duras (ej. prohibir `os.system`, bloqueo de red) pero sumar una capa semántica de _LLM-as-a-judge_ que evalúe si el código cumple exactamente el intent o asume cosas riesgosas.
    - **Impacto:** Máxima confiabilidad. Lo determinístico frena lo catastrófico rápido; el Juez LLM frena las alucinaciones lógicas sutiles y previene que código inseguro entre al entorno de ejecución.
4.  **Self-Correction Loop (Prioridad Alta)**
    - **Acción:** En `executor.py`, si la ejecución del código falla (StackTrace Error), el motor auto-envía ese error al LLM para que corrija su propio código (con un límite de intentos, ej. 3) antes de fallarle al usuario.
    - **Impacto:** Menos fallos duros, mejor tasa de éxito y sensación de "inteligencia real". Al estar detrás del reviewer, los ciclos de corrección son seguros.

## Fase 2 — Producto Usable (Prioridad Alta)

Convertir el motor inteligente en un producto con identidad propia.

1.  **Dashboards Persistentes y Análisis Antiguos**
    - **Acción:** Explotar la colección `AnalysisRun` almacenada en Atlas para que el frontend pueda recuperar gráficos y resultados pasados instantáneamente.
    - **Impacto:** Ahorro directo de dinero (tokens) y una mejora inmediata y tangible en el valor del producto para el usuario recurrente.
2.  **Aprobación Granular del Plan**
    - **Acción:** La Fase 1 (Plan) no debe ser un botón de "Aceptar Todo". Permitir al usuario en el frontend editar, borrar o ajustar los pasos propuestos por la IA antes de enviarlos a `execute`.
    - **Impacto:** Pone al usuario al volante, dándole identidad al producto. Data-X deja de ser una "caja negra" y se vuelve un copiloto transparente.

## Fase 3 — Experiencia en Tiempo Real (Prioridad Media)

Comunicar el progreso de la ejecución de forma fluida.

1.  **Server-Sent Events (SSE)**
    - **Acción:** Implementar SSE para streamear el progreso de la Fase 3 del motor ("Iniciando paso 1...", "Corrigiendo error...", "Artefacto listo").
    - **Impacto:** Se eligen SSE sobre WebSockets porque es más simple de implementar y alinea perfecto con el caso de uso (el backend habla empujando progreso, no necesita canal bidireccional constante durante la ejecución). Previene timeouts HTTP en el browser para tareas largas.

## Fase 4 — Capacidades Avanzadas y Seguridad (Prioridad Media/Baja)

Extensiones corporativas, de alcance, procesamiento masivo y protección del servidor.

1.  **Protección de Payload y Límites Duros (Seguridad/Rendimiento)**
    - **Acción:**
      - Limitar el _Body Size_ en FastAPI (ej. 20MB o 50MB) para rechazar peticiones gigantes antes de que saturen la memoria RAM (OOM).
      - Añadir validación en Pydantic (`AnalyzeRequest`) para que si `data` tiene más de X filas (ej. 10,000), devuelva un HTTP 400 amistoso: "Dataset demasiado grande para el plan gratuito".
    - **Impacto:** Evita que el servidor caiga por ataques de inyección masiva de datos y controla los costos.
2.  **Subida de Archivos por Chunks / Multipart (Rendimiento)**
    - **Acción:** A futuro, evitar enviar el CSV completo convertido en un array JSON gigante. Implementar un endpoint `Multipart/form-data` para recibir el archivo binario/texto real y guardarlo a disco directo.
    - **Impacto:** Reduce drásticamente la latencia de subida y el consumo de RAM de FastAPI durante el parseo de Pydantic.

3.  **Políticas Basadas en Roles (RBAC)**
    - **Acicón:** Cruzar JWTs con permisos en `rules.json` (Ej. Manager = Análisis profundos; Analista = Solo lectura). Para implementar únicamente cuando haya adopción real, costos relevantes y el producto base sea estable.
4.  **Ripgrep sobre Datasets Históricos**
    - **Acción:** Expandir el flag `file_search_enabled` para conectar el motor con bases de datos documentales o S3 usando pipelines abstractos. Interesante, pero no es crítico hasta no dominar el dataset activo en contexto.
5.  **Integración de Sphinx CLI (`sphinx-ai-cli`) para Trabajos Asíncronos (Batch Jobs)**
    - **Acción:** Instalar y utilizar el Sphinx CLI para tareas de fondo pesadas. Mientras que la _SphinxAI Library_ se usa para el dashboard en tiempo real por su control estricto, el CLI se utilizará para "Analistas de Madrugada": scripts automatizados que corran sobre servidores de Jupyter (ej: `sphinx-cli chat --notebook-filepath /datasets/ventas_febrero.ipynb ...`) durante la noche para limpiar o explorar sets de datos gigantes sin bloquear hilos HTTP de FastAPI, inyectándole nuestras políticas mediante `--sphinx-rules-path`.
    - **Impacto:** Permite escalar Data-X a procesamiento offline puro o habilitar un modo "Jupyter Asistido" para data scientists expertos que quieran el control total sobre un notebook autogestionado en vez de un dashboard empaquetado.

---

_Este roadmap asegura no dispersar el enfoque técnico construyendo features de lujo sobre una máquina apagada. El objetivo inmediato es encender el motor LLM y encapsularlo en acero hermético._
