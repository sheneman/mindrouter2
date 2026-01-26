############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# vllm_out.py: Canonical schema to vLLM/OpenAI API format translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical schema to vLLM (OpenAI-compatible) API format translator."""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalMessage,
    CanonicalStreamChunk,
    CanonicalStreamChoice,
    CanonicalStreamDelta,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormatType,
    TextContent,
    UsageInfo,
)


class VLLMOutTranslator:
    """Translate canonical requests to vLLM (OpenAI-compatible) API format."""

    @staticmethod
    def translate_chat_request(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Translate canonical chat request to vLLM/OpenAI format.

        vLLM is fully OpenAI-compatible, so this produces standard OpenAI format.

        Args:
            canonical: Canonical chat request

        Returns:
            OpenAI-compatible API request body
        """
        messages = []
        for msg in canonical.messages:
            messages.append(VLLMOutTranslator._translate_message(msg))

        payload: Dict[str, Any] = {
            "model": canonical.model,
            "messages": messages,
            "stream": canonical.stream,
        }

        # Add optional parameters
        if canonical.temperature is not None:
            payload["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            payload["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            payload["max_tokens"] = canonical.max_tokens
        if canonical.stop is not None:
            payload["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            payload["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            payload["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            payload["seed"] = canonical.seed
        if canonical.n != 1:
            payload["n"] = canonical.n
        if canonical.user:
            payload["user"] = canonical.user

        # Handle structured output
        if canonical.response_format:
            payload["response_format"] = VLLMOutTranslator._translate_response_format(
                canonical
            )

        return payload

    @staticmethod
    def translate_completion_request(
        canonical: CanonicalCompletionRequest,
    ) -> Dict[str, Any]:
        """Translate canonical completion request to vLLM/OpenAI format.

        Args:
            canonical: Canonical completion request

        Returns:
            OpenAI-compatible API request body
        """
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "prompt": canonical.prompt,
            "stream": canonical.stream,
        }

        if canonical.temperature is not None:
            payload["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            payload["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            payload["max_tokens"] = canonical.max_tokens
        if canonical.stop is not None:
            payload["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            payload["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            payload["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            payload["seed"] = canonical.seed
        if canonical.suffix is not None:
            payload["suffix"] = canonical.suffix
        if canonical.echo:
            payload["echo"] = canonical.echo
        if canonical.n != 1:
            payload["n"] = canonical.n

        return payload

    @staticmethod
    def translate_embedding_request(
        canonical: CanonicalEmbeddingRequest,
    ) -> Dict[str, Any]:
        """Translate canonical embedding request to vLLM/OpenAI format.

        Args:
            canonical: Canonical embedding request

        Returns:
            OpenAI-compatible API request body
        """
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "input": canonical.input,
        }

        if canonical.encoding_format != "float":
            payload["encoding_format"] = canonical.encoding_format
        if canonical.dimensions is not None:
            payload["dimensions"] = canonical.dimensions

        return payload

    @staticmethod
    def translate_chat_response(
        openai_response: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> CanonicalChatResponse:
        """Translate vLLM/OpenAI chat response to canonical format.

        Args:
            openai_response: Raw OpenAI-format response
            request_id: Override request ID

        Returns:
            CanonicalChatResponse
        """
        choices = []
        for choice_data in openai_response.get("choices", []):
            message_data = choice_data.get("message", {})
            message = CanonicalMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
            )
            choices.append(
                CanonicalChoice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        usage_data = openai_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CanonicalChatResponse(
            id=request_id or openai_response.get("id", ""),
            created=openai_response.get("created", int(time.time())),
            model=openai_response.get("model", ""),
            choices=choices,
            usage=usage,
        )

    @staticmethod
    async def translate_chat_stream(
        openai_stream: AsyncIterator[bytes],
        request_id: str,
        model: str,
    ) -> AsyncIterator[CanonicalStreamChunk]:
        """Translate vLLM/OpenAI streaming response to canonical stream chunks.

        OpenAI streams Server-Sent Events:
        data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}
        data: [DONE]

        Args:
            openai_stream: Async iterator of response bytes
            request_id: Request ID
            model: Model name

        Yields:
            CanonicalStreamChunk objects
        """
        buffer = ""

        async for chunk_bytes in openai_stream:
            buffer += chunk_bytes.decode("utf-8")

            # Process complete SSE messages
            while "\n\n" in buffer or "\r\n\r\n" in buffer:
                # Handle both Unix and Windows line endings
                if "\r\n\r\n" in buffer:
                    message, buffer = buffer.split("\r\n\r\n", 1)
                else:
                    message, buffer = buffer.split("\n\n", 1)

                for line in message.split("\n"):
                    line = line.strip()

                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()

                        if data_str == "[DONE]":
                            return

                        try:
                            data = json.loads(data_str)

                            choices = []
                            for choice_data in data.get("choices", []):
                                delta_data = choice_data.get("delta", {})
                                delta = CanonicalStreamDelta(
                                    role=(
                                        MessageRole(delta_data["role"])
                                        if "role" in delta_data
                                        else None
                                    ),
                                    content=delta_data.get("content"),
                                )
                                choices.append(
                                    CanonicalStreamChoice(
                                        index=choice_data.get("index", 0),
                                        delta=delta,
                                        finish_reason=choice_data.get("finish_reason"),
                                    )
                                )

                            # Check for usage in final chunks
                            usage = None
                            if "usage" in data and data["usage"]:
                                usage_data = data["usage"]
                                usage = UsageInfo(
                                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                                    completion_tokens=usage_data.get(
                                        "completion_tokens", 0
                                    ),
                                    total_tokens=usage_data.get("total_tokens", 0),
                                )

                            yield CanonicalStreamChunk(
                                id=data.get("id", request_id),
                                created=data.get("created", int(time.time())),
                                model=data.get("model", model),
                                choices=choices,
                                usage=usage,
                            )

                        except json.JSONDecodeError:
                            continue

    @staticmethod
    def translate_embedding_response(
        openai_response: Dict[str, Any],
    ) -> CanonicalEmbeddingResponse:
        """Translate vLLM/OpenAI embedding response to canonical format.

        Args:
            openai_response: Raw OpenAI-format response

        Returns:
            CanonicalEmbeddingResponse
        """
        usage_data = openai_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CanonicalEmbeddingResponse(
            data=openai_response.get("data", []),
            model=openai_response.get("model", ""),
            usage=usage,
        )

    @staticmethod
    def _translate_message(msg: CanonicalMessage) -> Dict[str, Any]:
        """Translate canonical message to OpenAI format."""
        result: Dict[str, Any] = {
            "role": msg.role.value,
        }

        # Handle multimodal content
        if isinstance(msg.content, list):
            content_blocks = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    content_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageUrlContent):
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": block.image_url,
                        }
                    )
                elif isinstance(block, ImageBase64Content):
                    # Convert to data URL format
                    data_url = f"data:{block.media_type};base64,{block.data}"
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                elif isinstance(block, dict):
                    # Already in dict format
                    content_blocks.append(block)

            result["content"] = content_blocks
        else:
            result["content"] = msg.content

        if msg.name:
            result["name"] = msg.name

        return result

    @staticmethod
    def _translate_response_format(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Translate response format to OpenAI format."""
        if not canonical.response_format:
            return {"type": "text"}

        if canonical.response_format.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}

        if canonical.response_format.type == ResponseFormatType.JSON_SCHEMA:
            result: Dict[str, Any] = {"type": "json_schema"}
            if canonical.response_format.json_schema:
                result["json_schema"] = canonical.response_format.json_schema
            return result

        return {"type": "text"}
