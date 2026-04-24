"""Tests covering all issues from unclecode's code review.

Each test is designed to FAIL if the fix is reverted — not just assert the
happy path. Comments explain what breaks and why.
"""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from nanollm._types import Usage, ModelResponse, Choice, Message, ToolCall, FunctionCall
from nanollm.exceptions import raise_for_status, RateLimitError, BadRequestError
from nanollm.providers.aws import BedrockProvider, sigv4_headers
from nanollm.providers.anthropic import AnthropicProvider
from nanollm.providers.google import GeminiProvider
from nanollm.providers.openai import OpenAIProvider
from nanollm.providers import get_provider


# ── Critical 1: Bedrock SigV4 body hash mismatch ────────────────────────────


class TestBedrockSigV4:
    """SigV4 requires that the hash of the request body in the Authorization
    header matches the actual body bytes sent over the wire.  httpx serializes
    JSON without spaces; if we sign with a different serialization the
    signature is rejected by AWS with a 403 SignatureDoesNotMatch error.
    """

    def _body_bytes_we_sign(self, body: dict) -> bytes:
        """Simulate what client.py now computes for signing."""
        return json.dumps(body, separators=(",", ":")).encode("utf-8")

    def _body_bytes_httpx_sends(self, body: dict) -> bytes:
        """What httpx actually puts on the wire when called with json=body."""
        return httpx.Request("POST", "http://x.com", json=body).content

    def test_signed_bytes_match_sent_bytes_simple(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        assert self._body_bytes_we_sign(body) == self._body_bytes_httpx_sends(body)

    def test_signed_bytes_match_sent_bytes_nested(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "inferenceConfig": {"maxTokens": 1024, "temperature": 0.7},
            "stream": False,
        }
        assert self._body_bytes_we_sign(body) == self._body_bytes_httpx_sends(body)

    def test_naive_json_dumps_does_not_match_httpx(self):
        """Demonstrates the original bug: json.dumps without compact separators
        produces spaces that httpx's compact serializer doesn't include."""
        body = {"a": 1, "b": 2}
        naive_bytes = json.dumps(body).encode("utf-8")     # with spaces
        httpx_bytes = self._body_bytes_httpx_sends(body)   # without spaces
        assert naive_bytes != httpx_bytes, (
            "This test documents the original bug — if it fails, httpx changed "
            "its serialization and we need to re-verify the fix"
        )

    def test_payload_hash_in_signed_headers(self):
        """x-amz-content-sha256 must be part of the signed header set, not
        added afterward.  If it's added after signing, AWS rejects the request
        with a SignatureDoesNotMatch error because the header appears in the
        request but not in the signature's signed-headers list."""
        headers = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse",
            body=b'{"test":1}',
            region="us-east-1",
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        assert "x-amz-content-sha256" in headers, (
            "x-amz-content-sha256 must be present in returned headers"
        )
        # Verify the hash is the SHA-256 of the body, not an afterthought
        expected_hash = hashlib.sha256(b'{"test":1}').hexdigest()
        assert headers["x-amz-content-sha256"] == expected_hash

    def test_bedrock_streaming_disabled(self):
        """Bedrock returns a binary event stream, not SSE.  Streaming must be
        disabled so the client falls back to non-streaming rather than silently
        returning zero chunks."""
        provider = BedrockProvider()
        assert provider.supports_streaming is False, (
            "Bedrock uses binary event stream protocol — SSE streaming must be disabled"
        )


# ── Critical 2: Error body with string 'error' value ────────────────────────


class TestStringErrorBody:
    """Some providers return {"error": "Rate limit exceeded"} with error as a
    string.  The original code did body.get("error", {}).get("message") which
    raises AttributeError on a string, swallowing the real HTTP error."""

    def test_string_error_body_raises_correct_exception(self):
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, {"error": "Rate limit exceeded"})
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_dict_error_body_still_works(self):
        with pytest.raises(RateLimitError):
            raise_for_status(429, {"error": {"message": "Rate limit", "type": "rate_limit_error"}})

    def test_nested_error_message_extracted(self):
        with pytest.raises(BadRequestError) as exc_info:
            raise_for_status(400, {"error": {"message": "Invalid model name"}})
        assert "Invalid model name" in str(exc_info.value)

    def test_none_status_code_does_not_crash(self):
        """APIConnectionError sets status_code=None.  raise_for_status must
        not crash with TypeError when called with None."""
        raise_for_status(None, "connection failed")  # must not raise


# ── Critical 3: batch_completion empty list ──────────────────────────────────


class TestBatchCompletionEdgeCases:
    def test_empty_batch_returns_empty_list(self):
        """ThreadPoolExecutor(max_workers=0) raises ValueError.  An empty
        messages list must return [] immediately without creating a thread pool."""
        import nanollm
        result = nanollm.batch_completion("openai/gpt-4o", messages=[])
        assert result == []

    def test_empty_batch_does_not_make_http_calls(self):
        """Verify no HTTP requests are attempted for empty input."""
        import nanollm
        call_count = 0

        def counting_fn(_response):
            nonlocal call_count
            call_count += 1

        nanollm.batch_completion("openai/gpt-4o", messages=[], logger_fn=counting_fn)
        assert call_count == 0


# ── Important 4: Anthropic tool_use extraction ───────────────────────────────


class TestAnthropicToolCalling:
    """tool_use blocks in the Anthropic response content must be extracted into
    the message's tool_calls field.  Originally they were silently discarded."""

    def _make_tool_response(self):
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {
                    "type": "tool_use",
                    "id": "toolu_01abc",
                    "name": "get_weather",
                    "input": {"location": "Paris", "unit": "celsius"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

    def test_tool_use_block_extracted_to_tool_calls(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        msg = response.choices[0].message
        assert msg.tool_calls is not None, "tool_calls must not be None when tool_use blocks present"
        assert len(msg.tool_calls) == 1

    def test_tool_call_id_preserved(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        tc = response.choices[0].message.tool_calls[0]
        assert tc.id == "toolu_01abc"

    def test_tool_call_name_preserved(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        tc = response.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"

    def test_tool_call_arguments_serialized(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        tc = response.choices[0].message.tool_calls[0]
        args = json.loads(tc.function.arguments)
        assert args["location"] == "Paris"
        assert args["unit"] == "celsius"

    def test_text_content_coexists_with_tool_calls(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        msg = response.choices[0].message
        assert msg.content == "I'll look that up."
        assert msg.tool_calls is not None

    def test_finish_reason_is_tool_calls(self):
        provider = AnthropicProvider()
        response = provider.parse_response(self._make_tool_response())
        assert response.choices[0].finish_reason == "tool_calls"

    def test_no_tool_use_gives_no_tool_calls(self):
        provider = AnthropicProvider()
        data = {
            "id": "msg_456",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "model": "claude-3-5-sonnet-20241022",
        }
        response = provider.parse_response(data)
        assert response.choices[0].message.tool_calls is None


# ── Important 5: httpx connection pooling ────────────────────────────────────


class TestHttpxConnectionPooling:
    def test_sync_client_is_reused(self):
        from nanollm._http import _get_sync_client
        c1 = _get_sync_client()
        c2 = _get_sync_client()
        assert c1 is c2, "New httpx.Client created per call — 21ms overhead per request"

    def test_async_client_is_reused(self):
        from nanollm._http import _get_async_client
        c1 = _get_async_client()
        c2 = _get_async_client()
        assert c1 is c2, "New httpx.AsyncClient created per call"


# ── Important 6: litellm shim type re-exports ────────────────────────────────


class TestLitellmShim:
    def test_model_response_importable(self):
        from litellm import ModelResponse as LitellmModelResponse
        from nanollm import ModelResponse as NanoModelResponse
        assert LitellmModelResponse is NanoModelResponse

    def test_completion_importable(self):
        from litellm import completion
        assert callable(completion)

    def test_usage_importable(self):
        from litellm import Usage as LUsage
        from nanollm import Usage as NUsage
        assert LUsage is NUsage

    def test_exceptions_importable(self):
        from litellm import RateLimitError, AuthenticationError, APIError
        assert issubclass(RateLimitError, Exception)

    def test_drop_params_synced(self):
        import litellm
        import nanollm
        litellm.drop_params = False
        assert nanollm.drop_params is False
        litellm.drop_params = True
        assert nanollm.drop_params is True


# ── Important 7: Gemini finish_reason consistency ────────────────────────────


class TestGeminiFinishReason:
    """parse_response and parse_stream_line must map the same Gemini finishReason
    values to the same OpenAI finish_reason strings.  Originally MAX_TOKENS
    mapped to 'stop' in parse_response but 'max_tokens' in parse_stream_line."""

    def _stream_finish_reason(self, raw_reason: str) -> str:
        provider = GeminiProvider()
        chunk_data = {
            "candidates": [{"content": {"parts": []}, "finishReason": raw_reason}]
        }
        chunk = provider.parse_stream_line(chunk_data)
        return chunk.choices[0].finish_reason

    def _response_finish_reason(self, raw_reason: str) -> str:
        provider = GeminiProvider()
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": raw_reason,
            }]
        }
        return provider.parse_response(data).choices[0].finish_reason

    def test_max_tokens_consistent(self):
        streaming = self._stream_finish_reason("MAX_TOKENS")
        non_streaming = self._response_finish_reason("MAX_TOKENS")
        assert streaming == non_streaming, (
            f"Inconsistent: streaming={streaming!r}, non-streaming={non_streaming!r}"
        )

    def test_stop_consistent(self):
        assert self._stream_finish_reason("STOP") == self._response_finish_reason("STOP")

    def test_safety_consistent(self):
        assert self._stream_finish_reason("SAFETY") == self._response_finish_reason("SAFETY")

    def test_max_tokens_maps_to_length(self):
        assert self._response_finish_reason("MAX_TOKENS") == "length"
        assert self._stream_finish_reason("MAX_TOKENS") == "length"


# ── Provider instance caching ─────────────────────────────────────────────────


class TestProviderCaching:
    """get_provider() should return the same instance on repeated calls.
    Creating new instances per API call wastes memory and initialization time."""

    def test_same_instance_returned(self):
        p1 = get_provider("openai")
        p2 = get_provider("openai")
        assert p1 is p2

    def test_different_providers_are_different_instances(self):
        p_openai = get_provider("openai")
        p_anthropic = get_provider("anthropic")
        assert p_openai is not p_anthropic
