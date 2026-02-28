import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorCollection
from .db import db_manager
from .contracts import AnalysisRequest, AnalysisPlanResponse, ExecutionResult

logger = logging.getLogger(__name__)

class AnalysisRepository:
    """
    Gestiona el ciclo de vida del estado agéntico (Persistencia).
    Modelos Lógicos guardados en MongoDB:
      - Analysis (El pedido original y su estado global)
      - AnalysisPlan (El plan propuesto por Sphinx)
      - AnalysisRun (El intento de ejecución y sus artefactos)
    
    Estados válidos: 
    created -> planning -> planned -> approved -> running -> reviewing -> completed | failed
    """
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        return db_manager.get_db()["analyses"]

    async def create_analysis(self, analysis_id: str, request: AnalysisRequest) -> None:
        """Paso 1: Guarda la solicitud original con estado 'planning'."""
        doc = {
            "analysis_id": analysis_id,
            "status": "planning",
            "request": request.model_dump(),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "plan": None,
            "run": None
        }
        await self.collection.insert_one(doc)
        logger.debug(f"Análisis {analysis_id} persistido en estado 'planning'.")

    async def update_plan(self, analysis_id: str, plan: AnalysisPlanResponse) -> None:
        """Paso 2: Guarda el plan propuesto y pasa a estado 'planned'."""
        update_data = {
            "$set": {
                "status": "planned",
                "plan": plan.model_dump(),
                "updated_at": datetime.now(timezone.utc)
            }
        }
        await self.collection.update_one({"analysis_id": analysis_id}, update_data)
        logger.debug(f"Plan guardado para {analysis_id}. Estado: 'planned'.")

    async def mark_approved(self, analysis_id: str) -> None:
        """Paso 3: El frontend aprueba el plan."""
        await self.collection.update_one(
            {"analysis_id": analysis_id},
            {"$set": {"status": "approved", "updated_at": datetime.now(timezone.utc)}}
        )

    async def mark_running(self, analysis_id: str) -> None:
        """Paso 4: El motor de ejecución arranca."""
        await self.collection.update_one(
            {"analysis_id": analysis_id},
            {"$set": {"status": "running", "updated_at": datetime.now(timezone.utc)}}
        )

    async def save_execution_result(self, analysis_id: str, result: ExecutionResult) -> None:
        """Paso 5: Guarda el resultado final y los artefactos. Estado pasa a completed/failed."""
        update_data = {
            "$set": {
                "status": result.status, # 'completed' or 'failed'
                "run": result.model_dump(),
                "updated_at": datetime.now(timezone.utc)
            }
        }
        await self.collection.update_one({"analysis_id": analysis_id}, update_data)
        logger.info(f"Ciclo terminado para {analysis_id}. Estado: {result.status}.")

    async def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Recupera el estado completo de un análisis."""
        doc = await self.collection.find_one({"analysis_id": analysis_id}, {"_id": 0})
        return doc

repository = AnalysisRepository()
