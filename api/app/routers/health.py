from fastapi import APIRouter

from api.app.core.config import PROVIDER
from api.app.services import legacy

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gamatrain AI (Production)",
        "provider": PROVIDER,
        "model": legacy.model_name(),
        "rag_enabled": legacy.rag_ready(),
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": PROVIDER,
        "model": legacy.model_name(),
        "rag_ready": legacy.rag_ready(),
    }

