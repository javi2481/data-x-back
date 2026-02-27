# Data-X: Manual de Políticas y Procedimientos

Este documento establece las directrices permanentes para el desarrollo de Data-X.

## Contexto del Proyecto

Data-X es un producto con:

- **backend en FastAPI**
- **frontend dashboard**
- **Sphinx** como motor de análisis
- Objetivo de **migrar y eliminar edge functions de Lovable**, reabsorbiendo esa lógica dentro del backend propio.

## Objetivo Arquitectónico

Construir una **arquitectura productiva, mantenible y confiable**.

- **FastAPI es la capa central de orquestación**.
- **Sphinx debe integrarse detrás de un adapter/provider backend**.
- **La lógica de negocio no debe quedar en el frontend ni en edge funciones**.
- **Los MCP servers son herramientas de desarrollo, no parte del runtime productivo**.
- **Toda entrada y salida importante debe pasar por contratos y validación explícita**.

---

## Política de uso de MCP Servers

**Principio Permanente**: Operar proactivamente con los servidores MCP para mejorar el código y la información.

### 1. Filesystem MCP

- **Uso**: Inspección de estructura, lectura/edición de archivos, revisión de prompts, adapters y logs.
- **Restricción**: Solo directorios autorizados, sin acceso innecesario a secretos, sin cambios fuera de scope.

### 2. HTTP MCP

- **Uso**: Verificación de endpoints de FastAPI, `/health`, smoke tests, validación de contratos para migración de edge functions.
- **Restricción**: Priorizar local/staging, evitar pruebas destructivas.

### 3. Python MCP

- **Uso**: Prototipado de Sphinx, validación de structured outputs, parsers y adaptadores.
- **Restricción**: No usar scripts temporales como solución final; la lógica debe vivir en el repo.

---

## Regla de Migración de Lovable Edge Functions

1. **Identificar**: Inputs, outputs, responsabilidades y dependencias.
2. **Diseñar en FastAPI**: Endpoint equivalente, esquemas de entrada/salida, manejo de errores.
3. **Validar**: Probar el endpoint con HTTP MCP y comparar comportamiento.
4. **Obsolescencia**: Eliminar la edge function solo tras validación exitosa.

---

## Política de Integración con Sphinx

- Integración solo a través de abstracciones en el backend.
- Desbloqueo del dominio del formato bruto del motor.
- Esquemas de validación para structured outputs.
- Mapeo de errores técnicos a errores de aplicación.

---

## Principios Operativos Obligatorios

- **Arquitectura Productiva**: Preferencia por el backend propio.
- **No improvisación**: Diferenciar explícitamente entre prototipo, herramienta de desarrollo e implementación final.
- **Contratos y Validación**: Nada se da por correcto sin validación explícita.

---

## Contexto Compartido del Proyecto

Estamos migrando **Data-X** hacia una arquitectura donde:

- el **backend FastAPI** será la única capa de orquestación y análisis
- **Sphinx** será el motor de análisis detrás de una abstracción backend
- el **frontend dashboard** dejará de depender de edge functions de Lovable
- la edge function `sphinx-analyze` debe quedar obsoleta y luego eliminarse
- el frontend debe llamar **directamente** al backend desplegado en Railway
- la implementación debe priorizar **arquitectura productiva**, no parches temporales

### Principios obligatorios

- No dejar lógica crítica en el frontend ni en edge functions
- No acoplar el dominio al formato bruto de respuesta de Sphinx
- Toda integración con Sphinx debe pasar por un **provider/adapter backend**
- Toda entrada y salida importante debe tener **schemas y validación**
- El backend debe devolver al frontend un contrato final estable y listo para renderizar
- Si hay dudas entre “solución rápida” y “arquitectura correcta”, priorizar arquitectura correcta

### Contrato funcional objetivo

El sistema debe soportar estos endpoints backend:

- `POST /analyze`
- `POST /analyze/deep`
- `POST /analyze/semantic`

El frontend enviará:

- `question`
- `data`
- `columns`
- `fileName`
- `modelSize`
- `mode`

El backend deberá encargarse de:

- preprocesamiento y smart sampling
- localización de idioma / instruction wrapping
- llamada a Sphinx
- normalización de respuesta
- visualizaciones en formato compatible con frontend
- secciones en modo deep
- fallback controlado
- metadata técnica útil

### Regla de trabajo entre agentes

- **Antigravity** define contratos, backend, servicios, schemas y estrategia de migración
- **Lovable** adapta el frontend al contrato backend y elimina dependencias de edge functions
- Ninguno debe inventar contratos distintos sin alinear primero con el otro
- Si detectan un problema en el contrato, deben proponer ajuste explícito antes de implementar algo incompatible
