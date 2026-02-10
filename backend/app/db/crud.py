############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# crud.py: Database CRUD operations for all entities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Database CRUD operations for MindRouter2."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import (
    ApiKey,
    ApiKeyStatus,
    Artifact,
    Backend,
    BackendEngine,
    BackendStatus,
    BackendTelemetry,
    GPUDevice,
    GPUDeviceTelemetry,
    Model,
    Modality,
    Node,
    NodeStatus,
    Quota,
    QuotaRequest,
    QuotaRequestStatus,
    Request,
    RequestStatus,
    Response,
    SchedulerDecision,
    UsageLedger,
    User,
    UserRole,
)


def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (MariaDB returns naive datetimes)."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# User CRUD
async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get user by username."""
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    username: str,
    email: str,
    password_hash: str,
    role: UserRole = UserRole.STUDENT,
    full_name: Optional[str] = None,
) -> User:
    """Create a new user."""
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=role,
        full_name=full_name,
    )
    db.add(user)
    await db.flush()
    return user


async def get_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
) -> List[User]:
    """Get list of users with optional filtering."""
    query = select(User).where(User.deleted_at.is_(None))
    if role:
        query = query.where(User.role == role)
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """Hard-delete a user and all child rows (no CASCADE on FKs).

    Deletion order matters due to foreign key constraints:
    1. scheduler_decisions (FK -> requests)
    2. responses (FK -> requests)
    3. artifacts (FK -> requests)
    4. usage_ledger (FK -> requests, api_keys, users)
    5. requests (FK -> users, api_keys)
    6. api_keys (FK -> users)
    7. quotas (FK -> users)
    8. quota_requests (FK -> users)
    9. users
    """
    # Get request IDs for this user (needed for child tables of requests)
    req_result = await db.execute(
        select(Request.id).where(Request.user_id == user_id)
    )
    request_ids = [r for (r,) in req_result.all()]

    if request_ids:
        # Delete children of requests
        await db.execute(
            delete(SchedulerDecision).where(
                SchedulerDecision.request_id.in_(request_ids)
            )
        )
        await db.execute(
            delete(Response).where(Response.request_id.in_(request_ids))
        )
        await db.execute(
            delete(Artifact).where(Artifact.request_id.in_(request_ids))
        )

    # Delete direct children of user
    await db.execute(
        delete(UsageLedger).where(UsageLedger.user_id == user_id)
    )
    await db.execute(
        delete(Request).where(Request.user_id == user_id)
    )
    await db.execute(
        delete(ApiKey).where(ApiKey.user_id == user_id)
    )
    await db.execute(
        delete(Quota).where(Quota.user_id == user_id)
    )
    await db.execute(
        delete(QuotaRequest).where(QuotaRequest.user_id == user_id)
    )

    # Delete the user
    result = await db.execute(
        delete(User).where(User.id == user_id)
    )
    await db.flush()
    return result.rowcount > 0


# API Key CRUD
async def get_api_key_by_hash(db: AsyncSession, key_hash: str) -> Optional[ApiKey]:
    """Get API key by hash."""
    result = await db.execute(
        select(ApiKey)
        .options(selectinload(ApiKey.user))
        .where(ApiKey.key_hash == key_hash)
    )
    return result.scalar_one_or_none()


async def get_api_key_by_prefix(db: AsyncSession, key_prefix: str) -> Optional[ApiKey]:
    """Get API key by prefix (for identification)."""
    result = await db.execute(
        select(ApiKey)
        .options(selectinload(ApiKey.user))
        .where(ApiKey.key_prefix == key_prefix)
    )
    return result.scalar_one_or_none()


async def create_api_key(
    db: AsyncSession,
    user_id: int,
    key_hash: str,
    key_prefix: str,
    name: str,
    expires_at: Optional[datetime] = None,
) -> ApiKey:
    """Create a new API key."""
    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
        expires_at=expires_at,
        status=ApiKeyStatus.ACTIVE,
    )
    db.add(api_key)
    await db.flush()
    return api_key


async def get_user_api_keys(
    db: AsyncSession, user_id: int, include_revoked: bool = False
) -> List[ApiKey]:
    """Get all API keys for a user."""
    query = select(ApiKey).where(ApiKey.user_id == user_id)
    if not include_revoked:
        query = query.where(ApiKey.status == ApiKeyStatus.ACTIVE)
    result = await db.execute(query)
    return list(result.scalars().all())


async def revoke_api_key(db: AsyncSession, api_key_id: int) -> Optional[ApiKey]:
    """Revoke an API key."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()
    if api_key:
        api_key.status = ApiKeyStatus.REVOKED
        await db.flush()
    return api_key


async def update_api_key_usage(db: AsyncSession, api_key_id: int) -> None:
    """Update API key last used timestamp and usage count.

    Uses an atomic UPDATE to avoid loading the row into the ORM session,
    which prevents holding an exclusive row lock for the entire request.
    """
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key_id)
        .values(
            last_used_at=func.now(),
            usage_count=ApiKey.usage_count + 1,
        )
    )


# Quota CRUD
async def get_user_quota(db: AsyncSession, user_id: int) -> Optional[Quota]:
    """Get quota for a user."""
    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    return result.scalar_one_or_none()


async def create_quota(
    db: AsyncSession,
    user_id: int,
    token_budget: int,
    rpm_limit: int,
    max_concurrent: int,
) -> Quota:
    """Create quota for a user."""
    quota = Quota(
        user_id=user_id,
        token_budget=token_budget,
        rpm_limit=rpm_limit,
        max_concurrent=max_concurrent,
    )
    db.add(quota)
    await db.flush()
    return quota


async def update_quota_usage(
    db: AsyncSession, user_id: int, tokens_used: int
) -> Optional[Quota]:
    """Update quota token usage."""
    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    quota = result.scalar_one_or_none()
    if quota:
        quota.tokens_used += tokens_used
        await db.flush()
    return quota


async def reset_quota_if_needed(db: AsyncSession, user_id: int) -> Optional[Quota]:
    """Reset quota if period has expired."""
    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    quota = result.scalar_one_or_none()
    if quota:
        period_end = _ensure_aware(quota.budget_period_start) + timedelta(days=quota.budget_period_days)
        if datetime.now(timezone.utc) >= period_end:
            quota.budget_period_start = datetime.now(timezone.utc)
            quota.tokens_used = 0
            await db.flush()
    return quota


# Node CRUD
async def create_node(
    db: AsyncSession,
    name: str,
    hostname: Optional[str] = None,
    sidecar_url: Optional[str] = None,
) -> Node:
    """Create a new node."""
    node = Node(
        name=name,
        hostname=hostname,
        sidecar_url=sidecar_url,
        status=NodeStatus.UNKNOWN,
    )
    db.add(node)
    await db.flush()
    return node


async def get_node_by_id(db: AsyncSession, node_id: int) -> Optional[Node]:
    """Get node by ID."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    return result.scalar_one_or_none()


async def get_node_by_name(db: AsyncSession, name: str) -> Optional[Node]:
    """Get node by name."""
    result = await db.execute(select(Node).where(Node.name == name))
    return result.scalar_one_or_none()


async def get_all_nodes(db: AsyncSession) -> List[Node]:
    """Get all nodes."""
    result = await db.execute(select(Node))
    return list(result.scalars().all())


async def update_node_hardware(
    db: AsyncSession,
    node_id: int,
    gpu_count: Optional[int] = None,
    driver_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
) -> Optional[Node]:
    """Update node hardware info from sidecar."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        if gpu_count is not None:
            node.gpu_count = gpu_count
        if driver_version is not None:
            node.driver_version = driver_version
        if cuda_version is not None:
            node.cuda_version = cuda_version
        await db.flush()
    return node


async def update_node_status(
    db: AsyncSession,
    node_id: int,
    status: NodeStatus,
) -> Optional[Node]:
    """Update node status."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        node.status = status
        await db.flush()
    return node


async def delete_node(db: AsyncSession, node_id: int) -> bool:
    """Delete a node. Fails if backends still reference it."""
    # Check for backends referencing this node
    result = await db.execute(
        select(Backend).where(Backend.node_id == node_id).limit(1)
    )
    if result.scalar_one_or_none():
        return False

    # Delete GPU devices on this node (cascade deletes their telemetry)
    gpu_result = await db.execute(
        select(GPUDevice).where(GPUDevice.node_id == node_id)
    )
    for device in gpu_result.scalars().all():
        await db.delete(device)

    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        await db.delete(node)
        await db.flush()
        return True
    return False


# Backend CRUD
async def get_backend_by_id(db: AsyncSession, backend_id: int) -> Optional[Backend]:
    """Get backend by ID."""
    result = await db.execute(
        select(Backend)
        .options(selectinload(Backend.models), selectinload(Backend.node))
        .where(Backend.id == backend_id)
    )
    return result.scalar_one_or_none()


async def get_backend_by_name(db: AsyncSession, name: str) -> Optional[Backend]:
    """Get backend by name."""
    result = await db.execute(select(Backend).where(Backend.name == name))
    return result.scalar_one_or_none()


async def get_healthy_backends(
    db: AsyncSession, engine: Optional[BackendEngine] = None
) -> List[Backend]:
    """Get all healthy backends."""
    query = select(Backend).where(Backend.status == BackendStatus.HEALTHY)
    if engine:
        query = query.where(Backend.engine == engine)
    result = await db.execute(query.options(selectinload(Backend.models), selectinload(Backend.node)))
    return list(result.scalars().all())


async def get_all_backends(db: AsyncSession) -> List[Backend]:
    """Get all backends."""
    result = await db.execute(
        select(Backend).options(selectinload(Backend.models), selectinload(Backend.node))
    )
    return list(result.scalars().all())


async def create_backend(
    db: AsyncSession,
    name: str,
    url: str,
    engine: BackendEngine,
    max_concurrent: int = 4,
    gpu_memory_gb: Optional[float] = None,
    gpu_type: Optional[str] = None,
    node_id: Optional[int] = None,
    gpu_indices: Optional[list] = None,
) -> Backend:
    """Register a new backend."""
    backend = Backend(
        name=name,
        url=url,
        engine=engine,
        max_concurrent=max_concurrent,
        gpu_memory_gb=gpu_memory_gb,
        gpu_type=gpu_type,
        node_id=node_id,
        gpu_indices=gpu_indices,
        status=BackendStatus.UNKNOWN,
    )
    db.add(backend)
    await db.flush()
    return backend


async def update_backend_status(
    db: AsyncSession,
    backend_id: int,
    status: BackendStatus,
    version: Optional[str] = None,
) -> Optional[Backend]:
    """Update backend health status."""
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        backend.status = status
        backend.last_health_check = datetime.now(timezone.utc)
        if version:
            backend.version = version
        if status == BackendStatus.HEALTHY:
            backend.consecutive_failures = 0
            backend.last_success = datetime.now(timezone.utc)
        else:
            backend.consecutive_failures += 1
        await db.flush()
    return backend


async def delete_backend(db: AsyncSession, backend_id: int) -> bool:
    """Delete a backend and its associated models."""
    # Delete associated models first
    models_result = await db.execute(
        select(Model).where(Model.backend_id == backend_id)
    )
    for model in models_result.scalars().all():
        await db.delete(model)

    # Delete the backend
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        await db.delete(backend)
        await db.flush()
        return True
    return False


async def update_backend_concurrency(
    db: AsyncSession, backend_id: int, delta: int
) -> Optional[Backend]:
    """Update backend concurrent request count."""
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        backend.current_concurrent = max(0, backend.current_concurrent + delta)
        await db.flush()
    return backend


async def update_backend_latency_ema(
    db: AsyncSession,
    backend_id: int,
    latency_ema_ms: Optional[float],
    ttft_ema_ms: Optional[float],
    throughput_score: float,
) -> None:
    """Persist latency EMA and derived throughput_score to the backend row."""
    await db.execute(
        update(Backend)
        .where(Backend.id == backend_id)
        .values(
            latency_ema_ms=latency_ema_ms,
            ttft_ema_ms=ttft_ema_ms,
            throughput_score=throughput_score,
        )
    )


async def update_backend_circuit_breaker(
    db: AsyncSession,
    backend_id: int,
    live_failure_count: int,
    circuit_open_until: Optional[datetime] = None,
) -> None:
    """Persist circuit breaker state."""
    await db.execute(
        update(Backend)
        .where(Backend.id == backend_id)
        .values(
            live_failure_count=live_failure_count,
            circuit_open_until=circuit_open_until,
        )
    )


# Model CRUD
async def get_models_for_backend(db: AsyncSession, backend_id: int) -> List[Model]:
    """Get all models for a backend."""
    result = await db.execute(
        select(Model).where(Model.backend_id == backend_id)
    )
    return list(result.scalars().all())


async def get_backends_with_model(
    db: AsyncSession,
    model_name: str,
    modality: Optional[Modality] = None,
) -> List[Backend]:
    """Get backends that have a specific model."""
    query = (
        select(Backend)
        .join(Model)
        .where(
            and_(
                Model.name == model_name,
                Backend.status == BackendStatus.HEALTHY,
            )
        )
    )
    if modality:
        query = query.where(Model.modality == modality)
    result = await db.execute(query.options(selectinload(Backend.models)))
    return list(result.scalars().all())


async def upsert_model(
    db: AsyncSession,
    backend_id: int,
    name: str,
    modality: Modality = Modality.CHAT,
    context_length: Optional[int] = None,
    supports_vision: bool = False,
    supports_structured_output: bool = True,
    is_loaded: bool = False,
) -> Model:
    """Create or update a model record."""
    result = await db.execute(
        select(Model).where(
            and_(Model.backend_id == backend_id, Model.name == name)
        )
    )
    model = result.scalar_one_or_none()

    if model:
        model.modality = modality
        model.context_length = context_length
        model.supports_vision = supports_vision
        model.supports_structured_output = supports_structured_output
        model.is_loaded = is_loaded
    else:
        model = Model(
            backend_id=backend_id,
            name=name,
            modality=modality,
            context_length=context_length,
            supports_vision=supports_vision,
            supports_structured_output=supports_structured_output,
            is_loaded=is_loaded,
        )
        db.add(model)

    await db.flush()
    return model


async def get_all_available_models(db: AsyncSession) -> List[Tuple[str, List[Backend]]]:
    """Get all unique model names with their available backends."""
    result = await db.execute(
        select(Model.name, Backend)
        .join(Backend)
        .where(Backend.status == BackendStatus.HEALTHY)
        .order_by(Model.name)
    )
    # Group by model name
    models_dict: dict[str, List[Backend]] = {}
    for row in result.all():
        model_name, backend = row
        if model_name not in models_dict:
            models_dict[model_name] = []
        models_dict[model_name].append(backend)
    return list(models_dict.items())


# Request/Response CRUD
async def create_request(
    db: AsyncSession,
    user_id: int,
    api_key_id: int,
    endpoint: str,
    model: str,
    modality: Modality,
    is_streaming: bool = False,
    messages: Optional[dict] = None,
    prompt: Optional[str] = None,
    parameters: Optional[dict] = None,
    response_format: Optional[dict] = None,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Request:
    """Create a new request record."""
    request = Request(
        user_id=user_id,
        api_key_id=api_key_id,
        endpoint=endpoint,
        model=model,
        modality=modality,
        is_streaming=is_streaming,
        messages=messages,
        prompt=prompt,
        parameters=parameters,
        response_format=response_format,
        client_ip=client_ip,
        user_agent=user_agent,
        status=RequestStatus.QUEUED,
    )
    db.add(request)
    await db.flush()
    return request


async def update_request_started(
    db: AsyncSession, request_id: int, backend_id: int
) -> Optional[Request]:
    """Update request when processing starts."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.PROCESSING
        request.backend_id = backend_id
        request.started_at = datetime.now(timezone.utc)
        queue_delay = request.started_at - _ensure_aware(request.queued_at)
        request.queue_delay_ms = int(queue_delay.total_seconds() * 1000)
        await db.flush()
    return request


async def update_request_completed(
    db: AsyncSession,
    request_id: int,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    tokens_estimated: bool = False,
) -> Optional[Request]:
    """Update request when completed."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        if request.started_at:
            processing_time = request.completed_at - _ensure_aware(request.started_at)
            request.processing_time_ms = int(processing_time.total_seconds() * 1000)
        total_time = request.completed_at - _ensure_aware(request.queued_at)
        request.total_time_ms = int(total_time.total_seconds() * 1000)
        request.prompt_tokens = prompt_tokens
        request.completion_tokens = completion_tokens
        if prompt_tokens and completion_tokens:
            request.total_tokens = prompt_tokens + completion_tokens
        request.tokens_estimated = tokens_estimated
        await db.flush()
    return request


async def update_request_failed(
    db: AsyncSession, request_id: int, error_message: str, error_code: Optional[str] = None
) -> Optional[Request]:
    """Update request when failed."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.FAILED
        request.completed_at = datetime.now(timezone.utc)
        request.error_message = error_message
        request.error_code = error_code
        if request.started_at:
            processing_time = request.completed_at - _ensure_aware(request.started_at)
            request.processing_time_ms = int(processing_time.total_seconds() * 1000)
        total_time = request.completed_at - _ensure_aware(request.queued_at)
        request.total_time_ms = int(total_time.total_seconds() * 1000)
        await db.flush()
    return request


async def create_response(
    db: AsyncSession,
    request_id: int,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chunk_count: int = 0,
    first_token_time_ms: Optional[int] = None,
    structured_output_valid: Optional[bool] = None,
    validation_errors: Optional[list] = None,
    raw_response: Optional[dict] = None,
) -> Response:
    """Create a response record."""
    response = Response(
        request_id=request_id,
        content=content,
        finish_reason=finish_reason,
        chunk_count=chunk_count,
        first_token_time_ms=first_token_time_ms,
        structured_output_valid=structured_output_valid,
        validation_errors=validation_errors,
        raw_response=raw_response,
    )
    db.add(response)
    await db.flush()
    return response


# Usage Ledger CRUD
async def create_usage_entry(
    db: AsyncSession,
    user_id: int,
    api_key_id: int,
    request_id: int,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    is_estimated: bool = False,
    backend_id: Optional[int] = None,
) -> UsageLedger:
    """Create a usage ledger entry."""
    entry = UsageLedger(
        user_id=user_id,
        api_key_id=api_key_id,
        request_id=request_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        is_estimated=is_estimated,
        backend_id=backend_id,
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_user_usage_in_window(
    db: AsyncSession, user_id: int, window_seconds: int
) -> int:
    """Get total tokens used by user in a time window."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    result = await db.execute(
        select(func.sum(UsageLedger.total_tokens))
        .where(
            and_(
                UsageLedger.user_id == user_id,
                UsageLedger.created_at >= cutoff,
            )
        )
    )
    total = result.scalar_one_or_none()
    return total or 0


# Telemetry CRUD
async def create_telemetry_snapshot(
    db: AsyncSession,
    backend_id: int,
    gpu_utilization: Optional[float] = None,
    gpu_memory_used_gb: Optional[float] = None,
    gpu_memory_total_gb: Optional[float] = None,
    active_requests: int = 0,
    queued_requests: int = 0,
    loaded_models: Optional[list] = None,
    gpu_power_draw_watts: Optional[float] = None,
    gpu_fan_speed_percent: Optional[float] = None,
    gpu_temperature: Optional[float] = None,
) -> BackendTelemetry:
    """Create a telemetry snapshot."""
    telemetry = BackendTelemetry(
        backend_id=backend_id,
        gpu_utilization=gpu_utilization,
        gpu_memory_used_gb=gpu_memory_used_gb,
        gpu_memory_total_gb=gpu_memory_total_gb,
        gpu_temperature=gpu_temperature,
        gpu_power_draw_watts=gpu_power_draw_watts,
        gpu_fan_speed_percent=gpu_fan_speed_percent,
        active_requests=active_requests,
        queued_requests=queued_requests,
        loaded_models=loaded_models,
    )
    db.add(telemetry)
    await db.flush()
    return telemetry


async def get_latest_telemetry(
    db: AsyncSession, backend_id: int
) -> Optional[BackendTelemetry]:
    """Get latest telemetry snapshot for a backend."""
    result = await db.execute(
        select(BackendTelemetry)
        .where(BackendTelemetry.backend_id == backend_id)
        .order_by(BackendTelemetry.timestamp.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# GPU Device CRUD
async def upsert_gpu_device(
    db: AsyncSession,
    node_id: int,
    gpu_index: int,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
    pci_bus_id: Optional[str] = None,
    compute_capability: Optional[str] = None,
    memory_total_gb: Optional[float] = None,
    power_limit_watts: Optional[float] = None,
) -> GPUDevice:
    """Create or update a GPU device record."""
    result = await db.execute(
        select(GPUDevice).where(
            and_(GPUDevice.node_id == node_id, GPUDevice.gpu_index == gpu_index)
        )
    )
    device = result.scalar_one_or_none()

    if device:
        if uuid is not None:
            device.uuid = uuid
        if name is not None:
            device.name = name
        if pci_bus_id is not None:
            device.pci_bus_id = pci_bus_id
        if compute_capability is not None:
            device.compute_capability = compute_capability
        if memory_total_gb is not None:
            device.memory_total_gb = memory_total_gb
        if power_limit_watts is not None:
            device.power_limit_watts = power_limit_watts
    else:
        device = GPUDevice(
            node_id=node_id,
            gpu_index=gpu_index,
            uuid=uuid,
            name=name,
            pci_bus_id=pci_bus_id,
            compute_capability=compute_capability,
            memory_total_gb=memory_total_gb,
            power_limit_watts=power_limit_watts,
        )
        db.add(device)

    await db.flush()
    return device


async def create_gpu_device_telemetry(
    db: AsyncSession,
    gpu_device_id: int,
    utilization_gpu: Optional[float] = None,
    utilization_memory: Optional[float] = None,
    memory_used_gb: Optional[float] = None,
    memory_free_gb: Optional[float] = None,
    temperature_gpu: Optional[float] = None,
    temperature_memory: Optional[float] = None,
    power_draw_watts: Optional[float] = None,
    fan_speed_percent: Optional[float] = None,
    clock_sm_mhz: Optional[int] = None,
    clock_memory_mhz: Optional[int] = None,
) -> GPUDeviceTelemetry:
    """Create a per-GPU telemetry snapshot."""
    telemetry = GPUDeviceTelemetry(
        gpu_device_id=gpu_device_id,
        utilization_gpu=utilization_gpu,
        utilization_memory=utilization_memory,
        memory_used_gb=memory_used_gb,
        memory_free_gb=memory_free_gb,
        temperature_gpu=temperature_gpu,
        temperature_memory=temperature_memory,
        power_draw_watts=power_draw_watts,
        fan_speed_percent=fan_speed_percent,
        clock_sm_mhz=clock_sm_mhz,
        clock_memory_mhz=clock_memory_mhz,
    )
    db.add(telemetry)
    await db.flush()
    return telemetry


async def get_gpu_devices_for_node(
    db: AsyncSession, node_id: int
) -> List[GPUDevice]:
    """Get all GPU devices for a node."""
    result = await db.execute(
        select(GPUDevice)
        .where(GPUDevice.node_id == node_id)
        .order_by(GPUDevice.gpu_index)
    )
    return list(result.scalars().all())


async def get_gpu_devices_for_backend(
    db: AsyncSession, backend_id: int
) -> List[GPUDevice]:
    """Get GPU devices assigned to a backend (via its node + gpu_indices)."""
    # Look up the backend's node_id and gpu_indices
    result = await db.execute(
        select(Backend.node_id, Backend.gpu_indices).where(Backend.id == backend_id)
    )
    row = result.one_or_none()
    if not row or not row[0]:
        return []

    node_id, gpu_indices = row

    query = (
        select(GPUDevice)
        .where(GPUDevice.node_id == node_id)
        .order_by(GPUDevice.gpu_index)
    )
    if gpu_indices:
        query = query.where(GPUDevice.gpu_index.in_(gpu_indices))

    result = await db.execute(query)
    return list(result.scalars().all())


async def get_all_gpu_devices(db: AsyncSession) -> List[GPUDevice]:
    """Get all GPU devices across all nodes."""
    result = await db.execute(
        select(GPUDevice).order_by(GPUDevice.node_id, GPUDevice.gpu_index)
    )
    return list(result.scalars().all())


async def get_backend_telemetry_history(
    db: AsyncSession,
    backend_id: int,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated backend telemetry history with time bucketing."""
    from sqlalchemy import text

    query = text("""
        SELECT
            DATE_FORMAT(timestamp, :bucket_format) as time_bucket,
            AVG(gpu_utilization) as avg_gpu_utilization,
            MIN(gpu_utilization) as min_gpu_utilization,
            MAX(gpu_utilization) as max_gpu_utilization,
            AVG(gpu_memory_used_gb) as avg_gpu_memory_used_gb,
            AVG(gpu_memory_total_gb) as avg_gpu_memory_total_gb,
            AVG(gpu_temperature) as avg_gpu_temperature,
            AVG(gpu_power_draw_watts) as avg_gpu_power_draw_watts,
            AVG(active_requests) as avg_active_requests,
            AVG(queued_requests) as avg_queued_requests,
            AVG(requests_per_second) as avg_requests_per_second
        FROM backend_telemetry
        WHERE backend_id = :backend_id
          AND timestamp >= :start
          AND timestamp <= :end
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    # Choose bucket format based on resolution
    if resolution_minutes <= 1:
        bucket_format = "%Y-%m-%d %H:%i"
    elif resolution_minutes <= 5:
        # Round to 5-minute intervals
        bucket_format = "%Y-%m-%d %H:"
        # Use a more complex expression for 5-min bucketing
        query = text("""
            SELECT
                CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                       LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
                AVG(gpu_utilization) as avg_gpu_utilization,
                MIN(gpu_utilization) as min_gpu_utilization,
                MAX(gpu_utilization) as max_gpu_utilization,
                AVG(gpu_memory_used_gb) as avg_gpu_memory_used_gb,
                AVG(gpu_memory_total_gb) as avg_gpu_memory_total_gb,
                AVG(gpu_temperature) as avg_gpu_temperature,
                AVG(gpu_power_draw_watts) as avg_gpu_power_draw_watts,
                AVG(active_requests) as avg_active_requests,
                AVG(queued_requests) as avg_queued_requests,
                AVG(requests_per_second) as avg_requests_per_second
            FROM backend_telemetry
            WHERE backend_id = :backend_id
              AND timestamp >= :start
              AND timestamp <= :end
            GROUP BY time_bucket
            ORDER BY time_bucket
        """)
    elif resolution_minutes <= 60:
        bucket_format = "%Y-%m-%d %H:00"
    else:
        bucket_format = "%Y-%m-%d"

    params = {
        "backend_id": backend_id,
        "start": start,
        "end": end,
        "bucket_format": bucket_format,
        "res_min": resolution_minutes,
    }

    result = await db.execute(query, params)
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "gpu_utilization": row["avg_gpu_utilization"],
            "gpu_utilization_min": row["min_gpu_utilization"],
            "gpu_utilization_max": row["max_gpu_utilization"],
            "gpu_memory_used_gb": row["avg_gpu_memory_used_gb"],
            "gpu_memory_total_gb": row["avg_gpu_memory_total_gb"],
            "gpu_temperature": row["avg_gpu_temperature"],
            "gpu_power_draw_watts": row["avg_gpu_power_draw_watts"],
            "active_requests": row["avg_active_requests"],
            "queued_requests": row["avg_queued_requests"],
            "requests_per_second": row["avg_requests_per_second"],
        }
        for row in rows
    ]


async def get_gpu_device_telemetry_history(
    db: AsyncSession,
    gpu_device_id: int,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated per-GPU telemetry history with time bucketing."""
    from sqlalchemy import text

    query = text("""
        SELECT
            CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                   LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
            AVG(utilization_gpu) as avg_utilization_gpu,
            MIN(utilization_gpu) as min_utilization_gpu,
            MAX(utilization_gpu) as max_utilization_gpu,
            AVG(utilization_memory) as avg_utilization_memory,
            AVG(memory_used_gb) as avg_memory_used_gb,
            AVG(memory_free_gb) as avg_memory_free_gb,
            AVG(temperature_gpu) as avg_temperature_gpu,
            AVG(temperature_memory) as avg_temperature_memory,
            AVG(power_draw_watts) as avg_power_draw_watts,
            AVG(fan_speed_percent) as avg_fan_speed_percent,
            AVG(clock_sm_mhz) as avg_clock_sm_mhz,
            AVG(clock_memory_mhz) as avg_clock_memory_mhz
        FROM gpu_device_telemetry
        WHERE gpu_device_id = :gpu_device_id
          AND timestamp >= :start
          AND timestamp <= :end
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    result = await db.execute(query, {
        "gpu_device_id": gpu_device_id,
        "start": start,
        "end": end,
        "res_min": resolution_minutes,
    })
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "utilization_gpu": row["avg_utilization_gpu"],
            "utilization_gpu_min": row["min_utilization_gpu"],
            "utilization_gpu_max": row["max_utilization_gpu"],
            "utilization_memory": row["avg_utilization_memory"],
            "memory_used_gb": row["avg_memory_used_gb"],
            "memory_free_gb": row["avg_memory_free_gb"],
            "temperature_gpu": row["avg_temperature_gpu"],
            "temperature_memory": row["avg_temperature_memory"],
            "power_draw_watts": row["avg_power_draw_watts"],
            "fan_speed_percent": row["avg_fan_speed_percent"],
            "clock_sm_mhz": row["avg_clock_sm_mhz"],
            "clock_memory_mhz": row["avg_clock_memory_mhz"],
        }
        for row in rows
    ]


async def get_latest_gpu_device_telemetry(
    db: AsyncSession, node_id: int
) -> List[GPUDeviceTelemetry]:
    """Get the most recent telemetry for each GPU device on a node."""
    # Get device IDs for this node
    device_result = await db.execute(
        select(GPUDevice.id).where(GPUDevice.node_id == node_id)
    )
    device_ids = [row[0] for row in device_result.all()]

    if not device_ids:
        return []

    # Get max timestamp per device
    from sqlalchemy import text
    placeholders = ",".join(str(d) for d in device_ids)
    subq = text(f"""
        SELECT gpu_device_id, MAX(timestamp) as max_ts
        FROM gpu_device_telemetry
        WHERE gpu_device_id IN ({placeholders})
        GROUP BY gpu_device_id
    """)

    result = await db.execute(subq)
    latest_map = {row[0]: row[1] for row in result.all()}

    if not latest_map:
        return []

    # Fetch the actual rows
    conditions = []
    for device_id, max_ts in latest_map.items():
        conditions.append(
            and_(
                GPUDeviceTelemetry.gpu_device_id == device_id,
                GPUDeviceTelemetry.timestamp == max_ts,
            )
        )

    result = await db.execute(
        select(GPUDeviceTelemetry).where(or_(*conditions))
    )
    return list(result.scalars().all())


async def delete_old_telemetry(
    db: AsyncSession, older_than: datetime
) -> int:
    """Delete backend telemetry data older than the given datetime."""
    result = await db.execute(
        delete(BackendTelemetry).where(BackendTelemetry.timestamp < older_than)
    )
    await db.flush()
    return result.rowcount


async def delete_old_gpu_telemetry(
    db: AsyncSession, older_than: datetime
) -> int:
    """Delete per-GPU telemetry data older than the given datetime."""
    result = await db.execute(
        delete(GPUDeviceTelemetry).where(GPUDeviceTelemetry.timestamp < older_than)
    )
    await db.flush()
    return result.rowcount


# Quota Request CRUD
async def create_quota_request(
    db: AsyncSession,
    request_type: str,
    justification: str,
    user_id: Optional[int] = None,
    requester_name: Optional[str] = None,
    requester_email: Optional[str] = None,
    affiliation: Optional[str] = None,
    requested_tokens: Optional[int] = None,
    requested_rpm: Optional[int] = None,
) -> QuotaRequest:
    """Create a quota or API key request."""
    quota_request = QuotaRequest(
        user_id=user_id,
        requester_name=requester_name,
        requester_email=requester_email,
        affiliation=affiliation,
        request_type=request_type,
        justification=justification,
        requested_tokens=requested_tokens,
        requested_rpm=requested_rpm,
        status=QuotaRequestStatus.PENDING,
    )
    db.add(quota_request)
    await db.flush()
    return quota_request


async def get_pending_quota_requests(db: AsyncSession) -> List[QuotaRequest]:
    """Get all pending quota requests."""
    result = await db.execute(
        select(QuotaRequest)
        .where(QuotaRequest.status == QuotaRequestStatus.PENDING)
        .order_by(QuotaRequest.created_at.asc())
    )
    return list(result.scalars().all())


async def review_quota_request(
    db: AsyncSession,
    request_id: int,
    reviewer_id: int,
    status: QuotaRequestStatus,
    review_notes: Optional[str] = None,
) -> Optional[QuotaRequest]:
    """Review a quota request."""
    result = await db.execute(
        select(QuotaRequest).where(QuotaRequest.id == request_id)
    )
    quota_request = result.scalar_one_or_none()
    if quota_request:
        quota_request.status = status
        quota_request.reviewed_by = reviewer_id
        quota_request.reviewed_at = datetime.now(timezone.utc)
        quota_request.review_notes = review_notes
        await db.flush()
    return quota_request


# Scheduler Decision CRUD
async def create_scheduler_decision(
    db: AsyncSession,
    request_id: int,
    selected_backend_id: int,
    candidate_backends: Optional[list] = None,
    scores: Optional[dict] = None,
    user_deficit: Optional[float] = None,
    user_weight: Optional[float] = None,
    user_recent_usage: Optional[int] = None,
    hard_constraints_passed: Optional[list] = None,
    hard_constraints_failed: Optional[list] = None,
) -> SchedulerDecision:
    """Record a scheduler decision."""
    decision = SchedulerDecision(
        request_id=request_id,
        selected_backend_id=selected_backend_id,
        candidate_backends=candidate_backends,
        scores=scores,
        user_deficit=user_deficit,
        user_weight=user_weight,
        user_recent_usage=user_recent_usage,
        hard_constraints_passed=hard_constraints_passed,
        hard_constraints_failed=hard_constraints_failed,
    )
    db.add(decision)
    await db.flush()
    return decision


# Audit Search
async def search_requests(
    db: AsyncSession,
    user_id: Optional[int] = None,
    model: Optional[str] = None,
    status: Optional[RequestStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_text: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> Tuple[List[Request], int]:
    """Search requests with filters."""
    query = select(Request)
    count_query = select(func.count(Request.id))

    conditions = []
    if user_id:
        conditions.append(Request.user_id == user_id)
    if model:
        conditions.append(Request.model == model)
    if status:
        conditions.append(Request.status == status)
    if start_date:
        conditions.append(Request.created_at >= start_date)
    if end_date:
        conditions.append(Request.created_at <= end_date)
    if search_text:
        conditions.append(
            or_(
                Request.prompt.ilike(f"%{search_text}%"),
                func.json_extract(Request.messages, "$").ilike(f"%{search_text}%"),
            )
        )

    if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

    # Get total count
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    # Get paginated results
    query = query.order_by(Request.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query.options(selectinload(Request.response)))
    requests = list(result.scalars().all())

    return requests, total
