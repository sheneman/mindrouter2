############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# admin_api.py: Administrative API endpoints for management
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_admin
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import BackendEngine, BackendStatus, RequestStatus, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response models
class BackendRegisterRequest(BaseModel):
    """Request to register a new backend."""
    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=1)
    engine: BackendEngine
    max_concurrent: int = Field(default=4, ge=1)
    gpu_memory_gb: Optional[float] = None
    gpu_type: Optional[str] = None


class BackendResponse(BaseModel):
    """Backend information response."""
    id: int
    name: str
    url: str
    engine: str
    status: str
    max_concurrent: int
    current_concurrent: int
    gpu_memory_gb: Optional[float]
    gpu_type: Optional[str]
    supports_vision: bool
    supports_embeddings: bool
    version: Optional[str]
    last_health_check: Optional[datetime]

    class Config:
        from_attributes = True


class QueueStats(BaseModel):
    """Queue statistics."""
    total: int
    by_user: Dict[int, int]
    by_model: Dict[str, int]
    average_wait_seconds: float


class AuditSearchRequest(BaseModel):
    """Audit log search parameters."""
    user_id: Optional[int] = None
    model: Optional[str] = None
    status: Optional[RequestStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_text: Optional[str] = None
    skip: int = 0
    limit: int = 100


class AuditRecord(BaseModel):
    """Audit record response."""
    id: int
    request_uuid: str
    user_id: int
    endpoint: str
    model: str
    status: str
    is_streaming: bool
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_time_ms: Optional[int]
    created_at: datetime
    backend_id: Optional[int]

    class Config:
        from_attributes = True


# Backend Management
@router.post("/backends/register", response_model=BackendResponse)
async def register_backend(
    request: BackendRegisterRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Register a new backend server.

    Requires admin role.
    """
    # Check for duplicate name
    existing = await crud.get_backend_by_name(db, request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backend with name '{request.name}' already exists",
        )

    registry = get_registry()

    backend = await registry.register_backend(
        name=request.name,
        url=request.url,
        engine=request.engine,
        max_concurrent=request.max_concurrent,
        gpu_memory_gb=request.gpu_memory_gb,
        gpu_type=request.gpu_type,
        db=db,
    )

    logger.info(
        "backend_registered_by_admin",
        admin_id=admin.id,
        backend_id=backend.id,
        name=backend.name,
    )

    return BackendResponse(
        id=backend.id,
        name=backend.name,
        url=backend.url,
        engine=backend.engine.value,
        status=backend.status.value,
        max_concurrent=backend.max_concurrent,
        current_concurrent=backend.current_concurrent,
        gpu_memory_gb=backend.gpu_memory_gb,
        gpu_type=backend.gpu_type,
        supports_vision=backend.supports_vision,
        supports_embeddings=backend.supports_embeddings,
        version=backend.version,
        last_health_check=backend.last_health_check,
    )


@router.post("/backends/{backend_id}/disable")
async def disable_backend(
    backend_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Disable a backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    await registry.disable_backend(backend_id)

    logger.info(
        "backend_disabled",
        admin_id=admin.id,
        backend_id=backend_id,
    )

    return {"status": "disabled", "backend_id": backend_id}


@router.post("/backends/{backend_id}/enable")
async def enable_backend(
    backend_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Enable a previously disabled backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    await registry.enable_backend(backend_id)

    logger.info(
        "backend_enabled",
        admin_id=admin.id,
        backend_id=backend_id,
    )

    return {"status": "enabled", "backend_id": backend_id}


@router.post("/backends/{backend_id}/refresh")
async def refresh_backend(
    backend_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Force refresh capabilities for a backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    success = await registry.refresh_backend(backend_id)

    if success:
        return {"status": "refreshed", "backend_id": backend_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh backend",
        )


@router.get("/backends", response_model=List[BackendResponse])
async def list_backends(
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all backends."""
    backends = await crud.get_all_backends(db)

    return [
        BackendResponse(
            id=b.id,
            name=b.name,
            url=b.url,
            engine=b.engine.value,
            status=b.status.value,
            max_concurrent=b.max_concurrent,
            current_concurrent=b.current_concurrent,
            gpu_memory_gb=b.gpu_memory_gb,
            gpu_type=b.gpu_type,
            supports_vision=b.supports_vision,
            supports_embeddings=b.supports_embeddings,
            version=b.version,
            last_health_check=b.last_health_check,
        )
        for b in backends
    ]


# Queue Management
@router.get("/queue")
async def get_queue(
    admin: User = Depends(require_admin()),
):
    """Get scheduler queue statistics."""
    scheduler = get_scheduler()
    stats = await scheduler.get_stats()

    return {
        "queue": stats["queue"],
        "fair_share": stats["fair_share"],
        "backend_queues": stats.get("backend_queues", {}),
    }


# Audit Search
@router.get("/audit/search")
async def search_audit(
    user_id: Optional[int] = Query(None),
    model: Optional[str] = Query(None),
    status: Optional[RequestStatus] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    search_text: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Search audit logs.

    Supports filtering by user, model, status, date range, and text search.
    """
    requests, total = await crud.search_requests(
        db=db,
        user_id=user_id,
        model=model,
        status=status,
        start_date=start_date,
        end_date=end_date,
        search_text=search_text,
        skip=skip,
        limit=limit,
    )

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": [
            {
                "id": r.id,
                "request_uuid": r.request_uuid,
                "user_id": r.user_id,
                "endpoint": r.endpoint,
                "model": r.model,
                "status": r.status.value,
                "is_streaming": r.is_streaming,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_time_ms": r.total_time_ms,
                "created_at": r.created_at.isoformat(),
                "backend_id": r.backend_id,
            }
            for r in requests
        ],
    }


@router.get("/audit/{request_id}")
async def get_audit_detail(
    request_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Get full details for an audit record including prompt and response."""
    from sqlalchemy import select
    from backend.app.db.models import Request, Response

    result = await db.execute(
        select(Request).where(Request.id == request_id)
    )
    request = result.scalar_one_or_none()

    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found",
        )

    # Get response
    result = await db.execute(
        select(Response).where(Response.request_id == request_id)
    )
    response = result.scalar_one_or_none()

    return {
        "request": {
            "id": request.id,
            "request_uuid": request.request_uuid,
            "user_id": request.user_id,
            "api_key_id": request.api_key_id,
            "endpoint": request.endpoint,
            "model": request.model,
            "modality": request.modality.value,
            "is_streaming": request.is_streaming,
            "messages": request.messages,
            "prompt": request.prompt,
            "parameters": request.parameters,
            "response_format": request.response_format,
            "status": request.status.value,
            "backend_id": request.backend_id,
            "queued_at": request.queued_at.isoformat() if request.queued_at else None,
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "queue_delay_ms": request.queue_delay_ms,
            "processing_time_ms": request.processing_time_ms,
            "total_time_ms": request.total_time_ms,
            "prompt_tokens": request.prompt_tokens,
            "completion_tokens": request.completion_tokens,
            "error_message": request.error_message,
            "client_ip": request.client_ip,
        },
        "response": {
            "content": response.content if response else None,
            "finish_reason": response.finish_reason if response else None,
            "chunk_count": response.chunk_count if response else 0,
            "structured_output_valid": response.structured_output_valid if response else None,
            "validation_errors": response.validation_errors if response else None,
        } if response else None,
    }


# User Management
@router.get("/users")
async def list_users(
    role: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all users."""
    from backend.app.db.models import UserRole

    role_filter = UserRole(role) if role else None
    users = await crud.get_users(db, skip=skip, limit=limit, role=role_filter)

    return {
        "users": [
            {
                "id": u.id,
                "uuid": u.uuid,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role.value,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            }
            for u in users
        ]
    }


# Quota Request Management
@router.get("/quota-requests")
async def list_quota_requests(
    status: Optional[str] = Query(None),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List pending quota requests."""
    requests = await crud.get_pending_quota_requests(db)

    return {
        "requests": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "requester_name": r.requester_name,
                "requester_email": r.requester_email,
                "affiliation": r.affiliation,
                "request_type": r.request_type,
                "justification": r.justification,
                "requested_tokens": r.requested_tokens,
                "status": r.status.value,
                "created_at": r.created_at.isoformat(),
            }
            for r in requests
        ]
    }


class QuotaReviewRequest(BaseModel):
    """Request to review a quota request."""
    approved: bool
    notes: Optional[str] = None
    granted_tokens: Optional[int] = None


@router.post("/quota-requests/{request_id}/review")
async def review_quota_request(
    request_id: int,
    review: QuotaReviewRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Approve or deny a quota request."""
    from backend.app.db.models import QuotaRequestStatus

    status = QuotaRequestStatus.APPROVED if review.approved else QuotaRequestStatus.DENIED

    quota_request = await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=admin.id,
        status=status,
        review_notes=review.notes,
    )

    if not quota_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quota request not found",
        )

    # If approved and it's a quota increase, update user's quota
    if review.approved and quota_request.user_id and review.granted_tokens:
        quota = await crud.get_user_quota(db, quota_request.user_id)
        if quota:
            quota.token_budget = review.granted_tokens
            await db.flush()

    logger.info(
        "quota_request_reviewed",
        admin_id=admin.id,
        request_id=request_id,
        approved=review.approved,
    )

    return {"status": "reviewed", "approved": review.approved}
