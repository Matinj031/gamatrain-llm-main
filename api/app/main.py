import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.core.config import PROVIDER
from api.app.routers.chat import router as chat_router
from api.app.routers.health import router as health_router
from api.app.services import legacy

logger = logging.getLogger("GamatrainAPI")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Gamatrain AI Server (Modular App)...")
    logger.info("Using provider: %s", PROVIDER)
    legacy.initialize()
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gamatrain AI API",
        description="RAG-powered educational AI (Production, modularized)",
        version="2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()

