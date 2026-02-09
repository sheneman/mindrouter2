############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# inference.py: Core inference service for request routing and proxying
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Inference service - handles request routing and backend proxying."""

import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalMessage,
    CanonicalStreamChunk,
    CanonicalStreamChoice,
    CanonicalStreamDelta,
    MessageRole,
    UsageInfo,
)
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.scheduler.queue import Job, JobModality
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators import OllamaOutTranslator, VLLMOutTranslator
from backend.app.db import crud
from backend.app.db.models import ApiKey, Backend, BackendEngine, Modality, User
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


class InferenceService:
    """
    Handles inference request processing.

    Responsibilities:
    - Create and submit jobs to scheduler
    - Route requests to backends
    - Proxy requests and stream responses
    - Record audit logs
    - Track token usage
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._settings = get_settings()
        self._scheduler = get_scheduler()
        self._registry = get_registry()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._settings.backend_request_timeout,
            )
        return self._http_client

    async def chat_completion(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming chat completion.

        Args:
            request: Canonical chat request
            user: Authenticated user
            api_key: API key used
            http_request: Original HTTP request

        Returns:
            OpenAI-compatible chat completion response
        """
        # Check quota
        await self._check_quota(user, api_key)

        # Create audit record
        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/chat/completions"
        )

        # Create job
        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        # Route to backend
        backend, models = await self._route_request(job, user)

        try:
            # Proxy request
            response = await self._proxy_chat_request(request, backend)

            # Update records
            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except Exception as e:
            await self._fail_request(db_request, backend.id if backend else None, str(e), job)
            raise

    async def stream_chat_completion(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """
        Handle streaming chat completion.

        Yields SSE-formatted chunks.
        """
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/chat/completions"
        )

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        backend, models = await self._route_request(job, user)

        try:
            full_content = ""
            chunk_count = 0
            first_token_time = None

            async for chunk in self._proxy_stream_request(request, backend):
                if first_token_time is None:
                    first_token_time = time.time()

                # Format as SSE
                yield f"data: {chunk.model_dump_json()}\n\n".encode()

                chunk_count += 1

                # Accumulate content
                for choice in chunk.choices:
                    if choice.delta.content:
                        full_content += choice.delta.content

            # Send done signal
            yield b"data: [DONE]\n\n"

            # Update records
            await self._complete_streaming_request(
                db_request, backend.id, full_content, chunk_count, job
            )

        except Exception as e:
            await self._fail_request(db_request, backend.id if backend else None, str(e), job)
            raise

    async def embedding(
        self,
        request: CanonicalEmbeddingRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle embedding request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/embeddings",
            modality=Modality.EMBEDDING
        )

        job = self._scheduler.create_job_from_embedding_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        backend, models = await self._route_request(job, user, Modality.EMBEDDING)

        try:
            response = await self._proxy_embedding_request(request, backend)

            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.EMBEDDING
            )

            return response

        except Exception as e:
            await self._fail_request(db_request, backend.id if backend else None, str(e), job)
            raise

    async def ollama_chat(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama chat request (non-streaming)."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/api/chat"
        )

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        backend, models = await self._route_request(job, user)

        try:
            response = await self._proxy_ollama_chat(request, backend)

            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except Exception as e:
            await self._fail_request(db_request, backend.id if backend else None, str(e), job)
            raise

    async def stream_ollama_chat(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """Handle streaming Ollama chat request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/api/chat"
        )

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        backend, models = await self._route_request(job, user)

        try:
            full_content = ""
            chunk_count = 0

            async for chunk_data in self._proxy_ollama_stream(request, backend):
                yield (json.dumps(chunk_data) + "\n").encode()
                chunk_count += 1

                if "message" in chunk_data:
                    full_content += chunk_data["message"].get("content", "")

            await self._complete_streaming_request(
                db_request, backend.id, full_content, chunk_count, job
            )

        except Exception as e:
            await self._fail_request(db_request, backend.id if backend else None, str(e), job)
            raise

    async def stream_ollama_generate(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """Handle streaming Ollama generate request."""
        # Reuse chat streaming with different endpoint recorded
        async for chunk in self.stream_ollama_chat(request, user, api_key, http_request):
            yield chunk

    async def ollama_generate(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama generate request (non-streaming)."""
        result = await self.ollama_chat(request, user, api_key, http_request)
        # Convert chat format (message) to generate format (response)
        msg = result.pop("message", {})
        result["response"] = msg.get("content", "")
        return result

    async def ollama_embedding(
        self,
        request: CanonicalEmbeddingRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama embedding request."""
        return await self.embedding(request, user, api_key, http_request)

    async def _check_quota(self, user: User, api_key: ApiKey) -> None:
        """Check if user has sufficient quota."""
        # Reset quota if period expired
        await crud.reset_quota_if_needed(self.db, user.id)

        quota = await crud.get_user_quota(self.db, user.id)
        if quota and quota.tokens_used >= quota.token_budget:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Token quota exceeded",
            )

    async def _create_request_record(
        self,
        request: Any,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str,
        modality: Modality = Modality.CHAT,
    ):
        """Create audit record for the request."""
        messages = None
        prompt = None
        parameters = {}

        if hasattr(request, "messages"):
            messages = [m.model_dump() for m in request.messages]
        if hasattr(request, "prompt"):
            prompt = request.prompt if isinstance(request.prompt, str) else str(request.prompt)

        # Extract parameters
        for param in ["temperature", "top_p", "max_tokens", "stop"]:
            if hasattr(request, param) and getattr(request, param) is not None:
                parameters[param] = getattr(request, param)

        response_format = None
        if hasattr(request, "response_format") and request.response_format:
            response_format = request.response_format.model_dump()

        return await crud.create_request(
            db=self.db,
            user_id=user.id,
            api_key_id=api_key.id,
            endpoint=endpoint,
            model=request.model,
            modality=modality,
            is_streaming=getattr(request, "stream", False),
            messages=messages,
            prompt=prompt,
            parameters=parameters,
            response_format=response_format,
            client_ip=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

    async def _route_request(
        self,
        job: Job,
        user: User,
        modality: Optional[Modality] = None,
    ):
        """Route request to a backend."""
        # Get backends that support the model
        backends = await self._registry.get_backends_with_model(
            job.model, modality
        )

        if not backends:
            # Try getting all healthy backends
            backends = await self._registry.get_healthy_backends()

        if not backends:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No healthy backends available",
            )

        # Get models for each backend
        backend_models = {}
        for backend in backends:
            backend_models[backend.id] = await self._registry.get_backend_models(
                backend.id
            )

        # Get GPU utilizations
        gpu_utilizations = await self._registry.get_gpu_utilizations()

        # Submit to scheduler and route
        await self._scheduler.submit_job(job, user.role.value)
        try:
            decision = await self._scheduler.route_job(
                job, backends, backend_models, gpu_utilizations
            )
        except Exception:
            await self._scheduler.cancel_job(job.request_id)
            raise

        if not decision.success:
            await self._scheduler.cancel_job(job.request_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No suitable backend: {decision.reason}",
            )

        return decision.backend, backend_models.get(decision.backend.id, [])

    async def _proxy_chat_request(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy chat request to backend."""
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = False

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Translate response to OpenAI format
        if backend.engine == BackendEngine.OLLAMA:
            canonical = OllamaOutTranslator.translate_chat_response(
                data, request.request_id, request.model
            )
        else:
            canonical = VLLMOutTranslator.translate_chat_response(
                data, request.request_id
            )

        return canonical.model_dump()

    async def _proxy_stream_request(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> AsyncIterator[CanonicalStreamChunk]:
        """Proxy streaming request to backend."""
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = True

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            if backend.engine == BackendEngine.OLLAMA:
                async for chunk in OllamaOutTranslator.translate_chat_stream(
                    response.aiter_bytes(), request.request_id, request.model
                ):
                    yield chunk
            else:
                async for chunk in VLLMOutTranslator.translate_chat_stream(
                    response.aiter_bytes(), request.request_id, request.model
                ):
                    yield chunk

    async def _proxy_embedding_request(
        self,
        request: CanonicalEmbeddingRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy embedding request to backend."""
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/api/embeddings"
        else:
            payload = VLLMOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/v1/embeddings"

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        if backend.engine == BackendEngine.OLLAMA:
            canonical = OllamaOutTranslator.translate_embedding_response(
                data, request.model
            )
        else:
            canonical = VLLMOutTranslator.translate_embedding_response(data)

        return canonical.model_dump()

    async def _proxy_ollama_chat(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy chat request, return in Ollama format."""
        client = await self._get_http_client()

        payload = OllamaOutTranslator.translate_chat_request(request)
        payload["stream"] = False

        if backend.engine == BackendEngine.OLLAMA:
            url = f"{backend.url}/api/chat"
        else:
            # Need to translate through OpenAI and back
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Convert OpenAI format to Ollama format when backend is not Ollama
        if backend.engine != BackendEngine.OLLAMA:
            data = self._openai_response_to_ollama(data)

        return data

    def _openai_response_to_ollama(self, openai_response: Dict) -> Dict:
        """Convert a non-streaming OpenAI response to Ollama format."""
        choices = openai_response.get("choices", [])
        message = {"role": "assistant", "content": ""}
        finish_reason = "stop"
        if choices:
            msg = choices[0].get("message", {})
            message = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content") or "",
            }
            finish_reason = choices[0].get("finish_reason", "stop")

        usage = openai_response.get("usage", {})
        return {
            "model": openai_response.get("model", ""),
            "message": message,
            "done": True,
            "done_reason": finish_reason,
            "total_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    async def _proxy_ollama_stream(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Proxy streaming request, yield Ollama format chunks."""
        client = await self._get_http_client()

        payload = OllamaOutTranslator.translate_chat_request(request)
        payload["stream"] = True

        if backend.engine == BackendEngine.OLLAMA:
            url = f"{backend.url}/api/chat"
        else:
            url = f"{backend.url}/v1/chat/completions"

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            buffer = ""
            async for chunk_bytes in response.aiter_bytes():
                buffer += chunk_bytes.decode()

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    # Handle SSE format from vLLM
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            # Convert OpenAI chunk to Ollama format
                            ollama_chunk = self._openai_chunk_to_ollama(data)
                            yield ollama_chunk
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Native Ollama format
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def _openai_chunk_to_ollama(self, openai_chunk: Dict) -> Dict:
        """Convert OpenAI streaming chunk to Ollama format."""
        choices = openai_chunk.get("choices", [])
        if not choices:
            return {"done": True}

        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        return {
            "model": openai_chunk.get("model", ""),
            "message": {
                "role": delta.get("role", "assistant"),
                "content": delta.get("content", ""),
            },
            "done": finish is not None,
        }

    async def _complete_request(
        self,
        db_request,
        backend_id: int,
        response: Dict,
        job: Job,
        modality: Modality = Modality.CHAT,
    ) -> None:
        """Complete a request with response data."""
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Estimate if not provided
        tokens_estimated = prompt_tokens == 0 and completion_tokens == 0
        if tokens_estimated:
            prompt_tokens = job.estimated_prompt_tokens
            completion_tokens = job.estimated_completion_tokens

        # Update request record
        await crud.update_request_completed(
            self.db, db_request.id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=tokens_estimated,
        )

        # Create response record
        content = None
        if "choices" in response and response["choices"]:
            msg = response["choices"][0].get("message", {})
            content = msg.get("content")

        await crud.create_response(
            self.db, db_request.id,
            content=content,
            finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
        )

        # Update usage ledger
        total_tokens = prompt_tokens + completion_tokens
        await crud.create_usage_entry(
            self.db,
            user_id=db_request.user_id,
            api_key_id=db_request.api_key_id,
            request_id=db_request.id,
            model=db_request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_estimated=tokens_estimated,
            backend_id=backend_id,
        )

        # Update quota
        await crud.update_quota_usage(self.db, db_request.user_id, total_tokens)

        # Notify scheduler
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

        await self.db.commit()

    async def _complete_streaming_request(
        self,
        db_request,
        backend_id: int,
        content: str,
        chunk_count: int,
        job: Job,
    ) -> None:
        """Complete a streaming request."""
        # Estimate tokens
        prompt_tokens = job.estimated_prompt_tokens
        completion_tokens = self._scheduler.estimate_tokens(content)

        await crud.update_request_completed(
            self.db, db_request.id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=True,
        )

        await crud.create_response(
            self.db, db_request.id,
            content=content,
            chunk_count=chunk_count,
            finish_reason="stop",
        )

        total_tokens = prompt_tokens + completion_tokens
        await crud.create_usage_entry(
            self.db,
            user_id=db_request.user_id,
            api_key_id=db_request.api_key_id,
            request_id=db_request.id,
            model=db_request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_estimated=True,
            backend_id=backend_id,
        )

        await crud.update_quota_usage(self.db, db_request.user_id, total_tokens)
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

        await self.db.commit()

    async def _fail_request(
        self,
        db_request,
        backend_id: Optional[int],
        error_message: str,
        job: Job,
    ) -> None:
        """Record a failed request."""
        await crud.update_request_failed(
            self.db, db_request.id,
            error_message=error_message,
        )

        if backend_id:
            await self._scheduler.on_job_failed(job, backend_id)

        await self.db.commit()
