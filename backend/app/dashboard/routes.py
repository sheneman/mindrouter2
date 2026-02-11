############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# routes.py: Dashboard web routes and views
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Dashboard routes for MindRouter2."""

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import BackendEngine, QuotaRequestStatus, UserRole
from backend.app.db.session import get_async_db
from backend.app.security import generate_api_key, hash_password, verify_password
from backend.app.settings import get_settings

dashboard_router = APIRouter(tags=["dashboard"])

# Setup templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)


# Session management helpers
def get_session_user_id(request: Request) -> Optional[int]:
    """Get user ID from session cookie."""
    session_data = request.cookies.get("mindrouter_session")
    if session_data:
        try:
            # Simple session: just store user_id
            # In production, use signed cookies or JWT
            return int(session_data)
        except (ValueError, TypeError):
            pass
    return None


def set_session_cookie(response: Response, user_id: int) -> None:
    """Set session cookie."""
    response.set_cookie(
        key="mindrouter_session",
        value=str(user_id),
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )


def clear_session_cookie(response: Response) -> None:
    """Clear session cookie."""
    response.delete_cookie(key="mindrouter_session")


# Public Dashboard
@dashboard_router.get("/", response_class=HTMLResponse)
async def public_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Public dashboard showing cluster status."""
    settings = get_settings()

    # Get cluster stats
    try:
        registry = get_registry()
        all_backends = await registry.get_all_backends()
        healthy_backends = await registry.get_healthy_backends()
    except Exception:
        all_backends = []
        healthy_backends = []

    # Get models
    models = set()
    for backend in healthy_backends:
        backend_models = await registry.get_backend_models(backend.id)
        for m in backend_models:
            models.add(m.name)

    # Get scheduler stats
    try:
        scheduler = get_scheduler()
        scheduler_stats = await scheduler.get_stats()
        queue_size = scheduler_stats.get("queue", {}).get("total", 0)
    except Exception:
        queue_size = 0

    return templates.TemplateResponse(
        "public/index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "total_backends": len(all_backends),
            "healthy_backends": len(healthy_backends),
            "models": sorted(models),
            "queue_size": queue_size,
            "user_id": get_session_user_id(request),
        },
    )


@dashboard_router.get("/request-api-key", response_class=HTMLResponse)
async def request_api_key_form(request: Request):
    """Display API key request form."""
    return templates.TemplateResponse(
        "public/request_api_key.html",
        {"request": request, "submitted": False},
    )


@dashboard_router.post("/request-api-key", response_class=HTMLResponse)
async def submit_api_key_request(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    affiliation: str = Form(...),
    use_case: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Submit API key request."""
    await crud.create_quota_request(
        db=db,
        request_type="api_key",
        justification=use_case,
        requester_name=name,
        requester_email=email,
        affiliation=affiliation,
    )
    await db.commit()

    return templates.TemplateResponse(
        "public/request_api_key.html",
        {"request": request, "submitted": True},
    )


# Authentication
@dashboard_router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: Optional[str] = None):
    """Display login form."""
    return templates.TemplateResponse(
        "public/login.html",
        {"request": request, "error": error},
    )


@dashboard_router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Handle login."""
    user = await crud.get_user_by_username(db, username)

    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "public/login.html",
            {"request": request, "error": "Invalid username or password"},
        )

    if not user.is_active:
        return templates.TemplateResponse(
            "public/login.html",
            {"request": request, "error": "Account is inactive"},
        )

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, user.id)
    return response


@dashboard_router.get("/logout")
async def logout():
    """Handle logout."""
    response = RedirectResponse(url="/", status_code=302)
    clear_session_cookie(response)
    return response


# User Dashboard
@dashboard_router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """User dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # Get user's API keys
    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=True)

    # Get quota
    quota = await crud.get_user_quota(db, user_id)

    # Calculate quota usage percentage
    usage_percent = 0
    if quota and quota.token_budget > 0:
        usage_percent = min(100, (quota.tokens_used / quota.token_budget) * 100)

    return templates.TemplateResponse(
        "user/dashboard.html",
        {
            "request": request,
            "user": user,
            "api_keys": api_keys,
            "quota": quota,
            "usage_percent": usage_percent,
        },
    )


@dashboard_router.post("/dashboard/create-key")
async def create_api_key(
    request: Request,
    key_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new API key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # Generate new key
    full_key, key_hash, key_prefix = generate_api_key()

    # Store in database
    await crud.create_api_key(
        db=db,
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=key_name,
    )
    await db.commit()

    # Show the key to user (only time they'll see it)
    return templates.TemplateResponse(
        "user/key_created.html",
        {
            "request": request,
            "api_key": full_key,
            "key_name": key_name,
        },
    )


@dashboard_router.post("/dashboard/revoke-key/{key_id}")
async def revoke_key(
    request: Request,
    key_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Revoke an API key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    await crud.revoke_api_key(db, key_id)
    await db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)


@dashboard_router.get("/dashboard/request-quota", response_class=HTMLResponse)
async def request_quota_form(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Display quota request form."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)

    return templates.TemplateResponse(
        "user/request_quota.html",
        {"request": request, "user": user},
    )


@dashboard_router.post("/dashboard/request-quota")
async def submit_quota_request(
    request: Request,
    requested_tokens: int = Form(...),
    justification: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Submit quota increase request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    await crud.create_quota_request(
        db=db,
        user_id=user_id,
        request_type="quota_increase",
        justification=justification,
        requested_tokens=requested_tokens,
    )
    await db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)


# Admin Dashboard
@dashboard_router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get backends
    registry = get_registry()
    backends = await registry.get_all_backends()

    # Get pending requests
    pending_requests = await crud.get_pending_quota_requests(db)

    # Get scheduler stats
    scheduler = get_scheduler()
    scheduler_stats = await scheduler.get_stats()

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "user": user,
            "backends": backends,
            "pending_requests": pending_requests,
            "scheduler_stats": scheduler_stats,
        },
    )


@dashboard_router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin user management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    users = await crud.get_users(db, limit=1000)

    return templates.TemplateResponse(
        "admin/users.html",
        {"request": request, "user": user, "users": users},
    )


@dashboard_router.get("/admin/requests", response_class=HTMLResponse)
async def admin_requests(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin request management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    pending_requests = await crud.get_pending_quota_requests(db)

    return templates.TemplateResponse(
        "admin/requests.html",
        {"request": request, "user": user, "requests": pending_requests},
    )


@dashboard_router.post("/admin/requests/{request_id}/approve")
async def approve_request(
    request: Request,
    request_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Approve a quota/API key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        status=QuotaRequestStatus.APPROVED,
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.post("/admin/requests/{request_id}/deny")
async def deny_request(
    request: Request,
    request_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Deny a quota/API key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        status=QuotaRequestStatus.DENIED,
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.get("/admin/nodes", response_class=HTMLResponse)
async def admin_nodes(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin node management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    nodes = await crud.get_all_nodes(db)

    # Build node data with backends and GPU devices
    node_data = []
    for node in nodes:
        # Get backends on this node
        all_backends = await crud.get_all_backends(db)
        node_backends = [b for b in all_backends if b.node_id == node.id]
        gpu_devices = await crud.get_gpu_devices_for_node(db, node.id)

        node_data.append({
            "node": node,
            "backends": node_backends,
            "gpu_devices": gpu_devices,
        })

    return templates.TemplateResponse(
        "admin/nodes.html",
        {
            "request": request,
            "user": user,
            "nodes": node_data,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/nodes/register")
async def register_node(
    request: Request,
    name: str = Form(...),
    hostname: Optional[str] = Form(None),
    sidecar_url: Optional[str] = Form(None),
    sidecar_key: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Register a new node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        existing = await crud.get_node_by_name(db, name)
        if existing:
            return RedirectResponse(
                url="/admin/nodes?error=Node+name+already+exists", status_code=302
            )

        hostname_val = hostname if hostname else None
        sidecar_url_val = sidecar_url if sidecar_url else None
        sidecar_key_val = sidecar_key if sidecar_key else None

        registry = get_registry()
        await registry.register_node(
            name=name,
            hostname=hostname_val,
            sidecar_url=sidecar_url_val,
            sidecar_key=sidecar_key_val,
        )
        return RedirectResponse(url="/admin/nodes?success=registered", status_code=302)
    except Exception:
        return RedirectResponse(url="/admin/nodes?error=Registration+failed", status_code=302)


@dashboard_router.post("/admin/nodes/{node_id}/edit")
async def edit_node(
    request: Request,
    node_id: int,
    name: str = Form(...),
    hostname: Optional[str] = Form(None),
    sidecar_url: Optional[str] = Form(None),
    sidecar_key: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit an existing node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        kwargs = {"name": name}
        clear_fields = []

        hostname_val = hostname if hostname else None
        if hostname_val is not None:
            kwargs["hostname"] = hostname_val
        else:
            clear_fields.append("hostname")

        sidecar_url_val = sidecar_url if sidecar_url else None
        if sidecar_url_val is not None:
            kwargs["sidecar_url"] = sidecar_url_val
        else:
            clear_fields.append("sidecar_url")

        # Empty sidecar_key means "keep current" — only update if provided
        if sidecar_key and sidecar_key.strip():
            kwargs["sidecar_key"] = sidecar_key.strip()

        if clear_fields:
            kwargs["_clear_fields"] = clear_fields

        registry = get_registry()
        await registry.update_node(node_id, **kwargs)
        return RedirectResponse(url="/admin/nodes?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/nodes?error={error_msg}", status_code=302)


@dashboard_router.post("/admin/nodes/{node_id}/remove")
async def remove_node(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Remove a node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    removed = await registry.remove_node(node_id)
    if removed:
        return RedirectResponse(url="/admin/nodes?success=removed", status_code=302)
    else:
        return RedirectResponse(
            url="/admin/nodes?error=Cannot+remove+node+with+active+backends", status_code=302
        )


@dashboard_router.post("/admin/nodes/{node_id}/refresh")
async def refresh_node(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Refresh node sidecar data."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.refresh_node(node_id)
    return RedirectResponse(url="/admin/nodes?success=refreshed", status_code=302)


@dashboard_router.get("/admin/backends", response_class=HTMLResponse)
async def admin_backends(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin backend management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    backends = await registry.get_all_backends()
    nodes = await crud.get_all_nodes(db)

    # Get telemetry for each backend
    backend_data = []
    for backend in backends:
        telemetry = await registry.get_telemetry(backend.id)
        models = await registry.get_backend_models(backend.id)
        backend_data.append({
            "backend": backend,
            "telemetry": telemetry,
            "models": models,
        })

    return templates.TemplateResponse(
        "admin/backends.html",
        {
            "request": request,
            "user": user,
            "backends": backend_data,
            "nodes": nodes,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/backends/register")
async def register_backend(
    request: Request,
    name: str = Form(...),
    url: str = Form(...),
    engine: str = Form(...),
    max_concurrent: int = Form(4),
    gpu_memory_gb: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    node_id: Optional[str] = Form(None),
    gpu_indices: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Register a new backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        engine_enum = BackendEngine(engine)
        gpu_mem = float(gpu_memory_gb) if gpu_memory_gb else None
        gpu_type_val = gpu_type if gpu_type else None

        # Parse node_id
        node_id_val = int(node_id) if node_id else None

        # Parse gpu_indices (e.g., "0,1,2" -> [0, 1, 2])
        gpu_indices_val = None
        if gpu_indices and gpu_indices.strip():
            gpu_indices_val = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]

        # Check for duplicate name
        existing = await crud.get_backend_by_name(db, name)
        if existing:
            return RedirectResponse(
                url="/admin/backends?error=Backend+name+already+exists", status_code=302
            )

        registry = get_registry()
        await registry.register_backend(
            name=name,
            url=url,
            engine=engine_enum,
            max_concurrent=max_concurrent,
            gpu_memory_gb=gpu_mem,
            gpu_type=gpu_type_val,
            node_id=node_id_val,
            gpu_indices=gpu_indices_val,
        )
        return RedirectResponse(url="/admin/backends?success=registered", status_code=302)
    except Exception:
        return RedirectResponse(url="/admin/backends?error=Registration+failed", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/edit")
async def edit_backend(
    request: Request,
    backend_id: int,
    name: str = Form(...),
    url: str = Form(...),
    engine: str = Form(...),
    max_concurrent: int = Form(4),
    gpu_memory_gb: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    node_id: Optional[str] = Form(None),
    gpu_indices: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit an existing backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        engine_enum = BackendEngine(engine)
        gpu_mem = float(gpu_memory_gb) if gpu_memory_gb else None
        gpu_type_val = gpu_type if gpu_type else None
        node_id_val = int(node_id) if node_id else None

        gpu_indices_val = None
        if gpu_indices and gpu_indices.strip():
            gpu_indices_val = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]

        kwargs = {
            "name": name,
            "url": url,
            "engine": engine_enum,
            "max_concurrent": max_concurrent,
        }
        clear_fields = []
        if gpu_mem is not None:
            kwargs["gpu_memory_gb"] = gpu_mem
        else:
            clear_fields.append("gpu_memory_gb")
        if gpu_type_val is not None:
            kwargs["gpu_type"] = gpu_type_val
        else:
            clear_fields.append("gpu_type")
        if node_id_val is not None:
            kwargs["node_id"] = node_id_val
        else:
            clear_fields.append("node_id")
        if gpu_indices_val is not None:
            kwargs["gpu_indices"] = gpu_indices_val
        else:
            clear_fields.append("gpu_indices")
        if clear_fields:
            kwargs["_clear_fields"] = clear_fields

        registry = get_registry()
        await registry.update_backend(backend_id, **kwargs)
        return RedirectResponse(url="/admin/backends?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/backends?error={error_msg}", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/disable")
async def disable_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Disable a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.disable_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=disabled", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/enable")
async def enable_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Enable a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.enable_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=enabled", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/remove")
async def remove_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Remove/unregister a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    removed = await registry.remove_backend(backend_id)
    if removed:
        return RedirectResponse(url="/admin/backends?success=removed", status_code=302)
    else:
        return RedirectResponse(url="/admin/backends?error=Backend+not+found", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/refresh")
async def refresh_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Refresh backend capabilities."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.refresh_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=refreshed", status_code=302)


@dashboard_router.get("/admin/metrics", response_class=HTMLResponse)
async def admin_metrics(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin GPU metrics dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse(
        "admin/metrics.html",
        {"request": request, "user": user},
    )


@dashboard_router.get("/admin/audit", response_class=HTMLResponse)
async def admin_audit(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin audit log viewer."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get recent requests
    requests, total = await crud.search_requests(db, limit=100)

    return templates.TemplateResponse(
        "admin/audit.html",
        {"request": request, "user": user, "audit_requests": requests, "total": total},
    )
