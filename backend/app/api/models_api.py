############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# models_api.py: Models listing API endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Models listing API endpoint."""

import time
from typing import List

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.canonical_schemas import CanonicalModelInfo, CanonicalModelList
from backend.app.core.telemetry.registry import get_registry
from backend.app.db.session import get_async_db

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
) -> CanonicalModelList:
    """
    List available models (OpenAI-compatible).

    Returns all models available across healthy backends.
    """
    # Authenticate
    user, api_key = await authenticate_request(request, db)

    registry = get_registry()
    backends = await registry.get_healthy_backends()

    # Collect models with their capabilities and backends
    model_data: dict = {}

    for backend in backends:
        backend_models = await registry.get_backend_models(backend.id)

        for model in backend_models:
            if model.name not in model_data:
                model_data[model.name] = {
                    "backends": [],
                    "capabilities": {
                        "vision": False,
                        "embeddings": False,
                        "structured_output": True,
                    },
                    "created": int(model.created_at.timestamp()) if model.created_at else int(time.time()),
                }

            model_data[model.name]["backends"].append(backend.name)

            # Update capabilities
            if model.supports_vision:
                model_data[model.name]["capabilities"]["vision"] = True
            if "embed" in model.name.lower():
                model_data[model.name]["capabilities"]["embeddings"] = True

    # Build response
    models: List[CanonicalModelInfo] = []
    for name, data in sorted(model_data.items()):
        models.append(
            CanonicalModelInfo(
                id=name,
                created=data["created"],
                owned_by="mindrouter",
                capabilities=data["capabilities"],
                backends=data["backends"],
            )
        )

    return CanonicalModelList(data=models)
