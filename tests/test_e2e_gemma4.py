"""E2E tests: Validate vLLM/Gemma 4 endpoint connectivity and basic capabilities.

These tests require a running vLLM instance accessible via SSH tunnel or directly.
Set VLLM_BASE_URL env var to point at the endpoint (default: http://localhost:8002/v1).

Skip with: pytest -m "not e2e"
"""

import json
import os

import httpx
import pytest

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8002/v1")
MODEL_ID = "google/gemma-4-31B-it"

pytestmark = pytest.mark.e2e


def _have_vllm() -> bool:
    """Check if vLLM endpoint is reachable."""
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


skip_no_vllm = pytest.mark.skipif(
    not _have_vllm(),
    reason=f"vLLM not reachable at {VLLM_BASE_URL}",
)


@skip_no_vllm
class TestGemma4Connectivity:
    """E2E-001: Validate vLLM/Gemma 4 endpoint connectivity."""

    def test_models_endpoint_returns_gemma4(self):
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=10)
        assert r.status_code == 200
        data = r.json()
        model_ids = [m["id"] for m in data["data"]]
        assert MODEL_ID in model_ids

    def test_simple_chat_completion(self):
        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
                "max_tokens": 32,
                "temperature": 0.1,
            },
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        assert data["usage"]["completion_tokens"] > 0

    def test_code_generation(self):
        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function `add(a, b)` that returns a+b. Only the function, no explanation.",
                    }
                ],
                "max_tokens": 64,
                "temperature": 0.0,
            },
            timeout=30,
        )
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert "def add" in content
        assert "return" in content

    def test_tool_calling_format(self):
        """Verify Gemma 4 produces tool calls (even if in content field)."""
        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "What is 2+2? Use the calculator tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "description": "Evaluate a math expression",
                            "parameters": {
                                "type": "object",
                                "properties": {"expression": {"type": "string", "description": "Math expression"}},
                                "required": ["expression"],
                            },
                        },
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=30,
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        # Gemma 4 may put tool calls in content or tool_calls field
        has_tool_call = (msg.get("tool_calls") and len(msg["tool_calls"]) > 0) or (
            "calculator" in (msg.get("content") or "")
        )
        assert has_tool_call, f"Expected tool call, got: {msg}"

    def test_streaming_completion(self):
        """Verify streaming works with SSE."""
        with httpx.stream(
            "POST",
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Count to 5."}],
                "max_tokens": 64,
                "temperature": 0.0,
                "stream": True,
            },
            timeout=30,
        ) as r:
            assert r.status_code == 200
            chunks = []
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        chunks.append(delta["content"])
            full = "".join(chunks)
            assert len(full) > 0

    def test_system_prompt(self):
        """Verify system prompt is respected."""
        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": "You are a pirate. Always respond with 'Arrr'."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=30,
        )
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"].lower()
        assert "arr" in content


@skip_no_vllm
class TestGemma4AgentCapabilities:
    """Test capabilities needed for Symphony agent workflows."""

    def test_issue_understanding_and_code_generation(self):
        """Simulate an agent receiving a GitHub issue and generating code."""
        prompt = """You are an autonomous software engineer. Implement the following issue:

## Issue: Add a `fibonacci(n)` function

Write a Python function that returns the nth Fibonacci number.
Include a simple test.

Respond with ONLY the Python code in a single code block."""

        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.2,
            },
            timeout=60,
        )
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert "fibonacci" in content.lower() or "fib" in content.lower()
        assert "def " in content

    def test_multi_turn_continuation(self):
        """Simulate a continuation prompt (turn 2+)."""
        r = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Write a Python function that reverses a string."},
                    {"role": "assistant", "content": "```python\ndef reverse(s):\n    return s[::-1]\n```"},
                    {"role": "user", "content": "Now add a test for that function."},
                ],
                "max_tokens": 256,
                "temperature": 0.2,
            },
            timeout=30,
        )
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert "test" in content.lower() or "assert" in content.lower()
