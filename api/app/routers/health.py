from fastapi import APIRouter

from ..core.config import (
    ENABLE_RAG,
    GROQ_MODEL,
    OLLAMA_MODEL,
    OPENROUTER_MODEL,
    PROVIDER,
)

router = APIRouter(tags=["health"])

def _model_name() -> str:
    if PROVIDER == "ollama":
        return OLLAMA_MODEL
    if PROVIDER == "groq":
        return GROQ_MODEL
    return OPENROUTER_MODEL


@router.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gamatrain AI (Production)",
        "provider": PROVIDER,
        "model": _model_name(),
        "rag_enabled": ENABLE_RAG,
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": PROVIDER,
        "model": _model_name(),
        "rag_ready": False if not ENABLE_RAG else None,
    }

