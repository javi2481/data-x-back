import asyncio
import json
from typing import Dict, List

class EventStreamManager:
    """
    Gestor de Eventos Pub/Sub en memoria para Server-Sent Events (SSE).
    Mantiene una lista de colas asyncio por cada analysis_id para que el 
    frontend pueda suscribirse y recibir actualizaciones de progreso en tiempo real.
    """
    def __init__(self):
        self.listeners: Dict[str, List[asyncio.Queue]] = {}

    def listen(self, analysis_id: str) -> asyncio.Queue:
        if analysis_id not in self.listeners:
            self.listeners[analysis_id] = []
        q = asyncio.Queue()
        self.listeners[analysis_id].append(q)
        return q

    def stop_listening(self, analysis_id: str, q: asyncio.Queue):
        if analysis_id in self.listeners:
            if q in self.listeners[analysis_id]:
                self.listeners[analysis_id].remove(q)
            if not self.listeners[analysis_id]:
                del self.listeners[analysis_id]

    async def emit(self, analysis_id: str, event: str, data: dict):
        """
        Emite un mensaje a todos los clientes HTTP conectados a este analysis_id.
        El payload se formatea de acuerdo a la especificacion SSE.
        """
        if analysis_id in self.listeners:
            # Formato estándar de Server-Sent Events
            payload = json.dumps(data)
            message = f"event: {event}\ndata: {payload}\n\n"
            for q in self.listeners[analysis_id]:
                await q.put(message)

event_manager = EventStreamManager()
