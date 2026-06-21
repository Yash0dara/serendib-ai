# System health check endpoints

from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health_check():
    """
    Basic health check.
    Returns system status.
    """
    return {
        "status": "healthy",
        "app": "Serendib AI",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/services")
async def services_check():
    """
    Checks all connected services.
    """
    services = {}

    # Check MongoDB
    try:
        from backend.database.mongodb import mongodb
        mongodb.client.admin.command("ping")
        services["mongodb"] = "✅ connected"
    except Exception as e:
        services["mongodb"] = f"❌ {str(e)}"

    # Check ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        services["chromadb"] = f"✅ connected ({len(collections)} collections)"
    except Exception as e:
        services["chromadb"] = f"❌ {str(e)}"

    # Check Groq
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            services["groq_llm"] = "✅ key configured"
        else:
            services["groq_llm"] = "❌ key missing"
    except Exception as e:
        services["groq_llm"] = f"❌ {str(e)}"

    # Check Weather API
    try:
        weather_key = os.getenv("OPENWEATHER_API_KEY")
        if weather_key:
            services["weather_api"] = "✅ key configured"
        else:
            services["weather_api"] = "❌ key missing"
    except Exception as e:
        services["weather_api"] = f"❌ {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }