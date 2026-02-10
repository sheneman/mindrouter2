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
    Model,
    Modality,
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


# Backend CRUD
async def get_backend_by_id(db: AsyncSession, backend_id: int) -> Optional[Backend]:
    """Get backend by ID."""
    result = await db.execute(
        select(Backend)
        .options(selectinload(Backend.models))
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
    result = await db.execute(query.options(selectinload(Backend.models)))
    return list(result.scalars().all())


async def get_all_backends(db: AsyncSession) -> List[Backend]:
    """Get all backends."""
    result = await db.execute(
        select(Backend).options(selectinload(Backend.models))
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
) -> Backend:
    """Register a new backend."""
    backend = Backend(
        name=name,
        url=url,
        engine=engine,
        max_concurrent=max_concurrent,
        gpu_memory_gb=gpu_memory_gb,
        gpu_type=gpu_type,
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
) -> BackendTelemetry:
    """Create a telemetry snapshot."""
    telemetry = BackendTelemetry(
        backend_id=backend_id,
        gpu_utilization=gpu_utilization,
        gpu_memory_used_gb=gpu_memory_used_gb,
        gpu_memory_total_gb=gpu_memory_total_gb,
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
