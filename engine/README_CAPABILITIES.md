# SphinxAI Capabilities: Guía de Integración (Sprint 4)

El motor de **Data-X** fue diseñado para soportar _capabilities_ extendidas de SphinxAI. Estas se activan enviando flags booleanos en el payload a `/analyze/plan`.

## 1. Web Search (`web_enabled`)

**Estado:** Integrado a nivel contrato.
**Uso:** Cuando `capabilities.web_enabled = True` en el request, `context_builder.py` inyecta en el prompt que el agente tiene permiso para acceder a Internet.
**Integración futura:** Cuando conectes la librería `sphinxai` real en `executor.py`, debes inicializar el agente con la tool de web search de Parallel AI o Phind (dependiendo de la suite de Sphinx).

## 2. PDF Parsing (`pdf_parsing_enabled`)

**Estado:** Soportado en el contrato.
**Casos de uso para Data-X:**
Si un usuario sube un PDF (ej. un reporte trimestral) junto a un CSV, el backend necesita extraer texto o tablas sucias.
**Integración técnica (Reducto):**
En el endpoint de FastAPI de upload de archivos, si detecta un `.pdf`, llamas a la capability de parseo de Sphinx (Reducto) _antes_ de enviarle la tabla a pandas. Lo convertido se inyectará en el `sample_data` o como un `DatasetMetadata` secundario.

## 3. File Search (`file_search_enabled`)

**Estado:** Soportado en el contrato.
**Casos de uso para Data-X:**
Si el usuario sube un archivo `.zip` con 50 CSVs o archivos JSON anidados.
**Integración técnica (Ripgrep):**
En `executor.py`, si este flag está activo, la tool de File Search debe ser inyectada al agente para que pueda correr comandos de búsqueda, localizar qué archivo tiene qué métrica y solo cargar en memoria el pandas DataFrame que necesita.
