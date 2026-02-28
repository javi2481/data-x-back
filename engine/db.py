import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "data_x_db")

logger = logging.getLogger(__name__)

class DatabaseManager:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    def connect_db(cls):
        """Creates the database connection using the URI."""
        try:
            cls.client = AsyncIOMotorClient(MONGO_URI)
            cls.db = cls.client[MONGO_DB_NAME]
            logger.info("Conectado exitosamente a MongoDB Atlas (Motor).")
        except Exception as e:
            logger.error(f"Fallo al conectar a MongoDB: {e}")

    @classmethod
    def close_db(cls):
        """Closes the database connection."""
        if cls.client:
            cls.client.close()
            logger.info("Conexión a MongoDB cerrada.")

    @classmethod
    def get_db(cls):
        """Returns the active database instance."""
        if cls.db is None:
            cls.connect_db()
        return cls.db

db_manager = DatabaseManager()
