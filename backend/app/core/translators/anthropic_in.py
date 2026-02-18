############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# anthropic_in.py: Anthropic Messages API format to canonical schema translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Anthropic Messages API format to canonical schema translator."""

import json
from typing import Any, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    ContentBlock,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
    TextContent,
)


class AnthropicInTranslator:
    """Translate Anthropic Messages API requests to canonical format."""

    @staticmethod
    def translate_messages_request(data: Dict[str, Any]) -> CanonicalChatRequest:
        """Translate Anthropic Messages API request to canonical format.

        Args:
            data: Raw request body from Anthropic-compatible endpoint

        Returns:
            CanonicalChatRequest
        """
        messages: List[CanonicalMessage] = []

        # Handle system prompt (top-level in Anthropic API)
        system = data.get("system")
        if system is not None:
            system_msg = AnthropicInTranslator._translate_system(system)
            messages.append(system_msg)

        # Translate conversation messages
        for msg in data.get("messages", []):
            messages.append(AnthropicInTranslator._translate_message(msg))

        # Handle structured output via output_config
        response_format = None
        output_config = data.get("output_config")
        if output_config:
            fmt = output_config.get("format")
            if fmt and fmt.get("type") == "json_schema":
                response_format = ResponseFormat(
                    type=ResponseFormatType.JSON_SCHEMA,
                    json_schema=fmt.get("json_schema"),
                )

        # Handle thinking mode
        think = None
        thinking = data.get("thinking")
        if thinking:
            thinking_type = thinking.get("type")
            if thinking_type in ("enabled", "adaptive"):
                think = True
            elif thinking_type == "disabled":
                think = False

        # Map stop_sequences to stop
        stop = data.get("stop_sequences")

        # Map metadata.user_id to user
        user = None
        metadata = data.get("metadata")
        if metadata:
            user = metadata.get("user_id")

        return CanonicalChatRequest(
            model=data["model"],
            messages=messages,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            stream=data.get("stream", False),
            stop=stop,
            think=think,
            response_format=response_format,
            user=user,
        )

    @staticmethod
    def format_response(response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert OpenAI-format canonical response dict to Anthropic Messages response.

        Args:
            response: OpenAI-format response dict from inference service
            model: Model name to include in response

        Returns:
            Anthropic Messages API formatted response dict
        """
        # Extract content from OpenAI format
        choices = response.get("choices", [])
        content_text = ""
        finish_reason = "end_turn"
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content_text = message.get("content", "")
            finish_reason = AnthropicInTranslator._map_finish_reason(
                choice.get("finish_reason", "stop")
            )

        # Map usage fields
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return {
            "id": response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": content_text}],
            "stop_reason": finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    @staticmethod
    def format_stream_event(event_type: str, data: Dict[str, Any]) -> str:
        """Format a single Anthropic SSE event.

        Args:
            event_type: The event type (e.g. message_start, content_block_delta)
            data: The event data payload

        Returns:
            Formatted SSE string
        """
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def _translate_system(system: Any) -> CanonicalMessage:
        """Translate Anthropic system prompt to canonical system message.

        Anthropic system can be a string or an array of content blocks.
        """
        if isinstance(system, str):
            return CanonicalMessage(role=MessageRole.SYSTEM, content=system)

        # Array of content blocks (e.g. [{"type":"text","text":"..."}])
        if isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return CanonicalMessage(
                role=MessageRole.SYSTEM,
                content=" ".join(text_parts) if text_parts else "",
            )

        return CanonicalMessage(role=MessageRole.SYSTEM, content=str(system))

    @staticmethod
    def _translate_message(msg: Dict[str, Any]) -> CanonicalMessage:
        """Translate a single Anthropic message to canonical format."""
        role = MessageRole(msg["role"])
        content = msg.get("content")

        # Simple string content
        if isinstance(content, str):
            return CanonicalMessage(role=role, content=content)

        # Array of content blocks
        if isinstance(content, list):
            content_blocks: List[ContentBlock] = []
            for item in content:
                if isinstance(item, str):
                    content_blocks.append(TextContent(text=item))
                elif isinstance(item, dict):
                    block = AnthropicInTranslator._translate_content_block(item)
                    if block:
                        content_blocks.append(block)
            return CanonicalMessage(role=role, content=content_blocks)

        return CanonicalMessage(role=role, content=content or "")

    @staticmethod
    def _translate_content_block(item: Dict[str, Any]) -> Optional[ContentBlock]:
        """Translate an Anthropic content block to canonical format."""
        item_type = item.get("type")

        if item_type == "text":
            return TextContent(text=item.get("text", ""))

        elif item_type == "image":
            source = item.get("source", {})
            source_type = source.get("type")

            if source_type == "base64":
                return ImageBase64Content(
                    data=source.get("data", ""),
                    media_type=source.get("media_type", "image/png"),
                )
            elif source_type == "url":
                return ImageUrlContent(
                    image_url={
                        "url": source.get("url", ""),
                        "detail": "auto",
                    }
                )

        elif item_type in ("tool_use", "tool_result"):
            # Lossy conversion: represent as text for v1
            text = item.get("text", "") or json.dumps(item.get("input", item.get("content", "")))
            return TextContent(text=f"[{item_type}] {text}")

        return None

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        """Map OpenAI finish_reason to Anthropic stop_reason."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return mapping.get(reason, "end_turn")
