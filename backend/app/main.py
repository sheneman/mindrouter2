############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# main.py: FastAPI application entry point and configuration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import api_router
from backend.app.core.scheduler.policy import init_scheduler, shutdown_scheduler
from backend.app.core.telemetry.registry import init_registry, shutdown_registry
from backend.app.dashboard.routes import dashboard_router
from backend.app.logging_config import (
    bind_request_context,
    clear_request_context,
    get_logger,
    setup_logging,
)
from backend.app.settings import get_settings
from backend.app.storage.artifacts import get_artifact_storage

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    logger.info("Starting MindRouter2...")

    # Initialize components
    await init_registry()
    await init_scheduler()

    # Initialize storage
    storage = get_artifact_storage()
    await storage.initialize()

    logger.info("MindRouter2 started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MindRouter2...")
    await shutdown_scheduler()
    await shutdown_registry()
    logger.info("MindRouter2 shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="LLM Inference Load Balancer for Ollama and vLLM backends",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Add request ID to all requests."""
        import uuid

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        bind_request_context(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        clear_request_context()
        return response

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.exception("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "server_error"}},
        )

    # Include routers
    app.include_router(api_router)
    app.include_router(dashboard_router)

    # Mount static files for dashboard
    import os
    static_path = os.path.join(os.path.dirname(__file__), "dashboard", "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")

    return app


# Create application instance
app = create_app()


def main():
    """Run the application using uvicorn."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
