import os
from collections.abc import Generator
from pathlib import Path
from typing import cast

from openai import OpenAI


_local_env_loaded = False
_ALLOWED_ENV_KEYS = {"OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL"}


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _load_local_env_file(file_path: Path) -> None:
    if not file_path.exists():
        return

    for line in file_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key not in _ALLOWED_ENV_KEYS:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = _strip_wrapping_quotes(value.strip())


def _load_local_env() -> None:
    global _local_env_loaded
    if _local_env_loaded:
        return

    base = Path(__file__).resolve().parent
    for name in (".env.local", ".env"):
        _load_local_env_file(base / name)

    _local_env_loaded = True


_agent_config_cache: dict[str, dict[str, str]] | None = None


def _load_agent_config() -> dict[str, dict[str, str]]:
    """Load per-agent LLM overrides from llm_config.json (cached)."""
    global _agent_config_cache
    if _agent_config_cache is not None:
        return _agent_config_cache
    import json
    cfg_path = Path(__file__).resolve().parent / "llm_config.json"
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _agent_config_cache = {k: v for k, v in raw.items() if isinstance(v, dict)}
            else:
                _agent_config_cache = {}
        except Exception:
            _agent_config_cache = {}
    else:
        _agent_config_cache = {}
    return _agent_config_cache


def get_client(*, agent_name: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name).

    If *agent_name* matches a key in ``llm_config.json``, that config is used;
    otherwise fall back to environment variables.
    """
    _load_local_env()
    cfg = _load_agent_config().get(agent_name or "", {})
    base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8045/v1")
    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    model_name = cfg.get("model") or os.environ.get("LLM_MODEL", "gemini-3-pro-high")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it in your shell.")
    return OpenAI(base_url=base_url, api_key=api_key), model_name


def chat_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    agent_name: str | None = None,
    max_retries: int = 2,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Call the model and parse a JSON object response.

    Retries on transient errors (network, rate-limit, timeout).
    The system prompt must instruct the model to output JSON only.
    If *agent_name* is given, LLM config is loaded from ``llm_config.json``.
    """
    import time

    client, resolved_model = get_client(agent_name=agent_name)
    model_name = model or resolved_model

    last_err: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=timeout_seconds,
            )
            content = resp.choices[0].message.content or ""
            return _extract_json_object(content)
        except (ValueError, KeyError) as exc:
            raise  # JSON parse / schema errors are not retryable
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                import sys
                print(f"[llm_client] attempt {attempt+1} failed ({exc}), retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {1 + max_retries} attempts: {last_err}") from last_err


def chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    agent_name: str | None = None,
    max_retries: int = 2,
    timeout_seconds: float = 60.0,
) -> str:
    """Call the model and return plain text (no JSON parsing)."""
    import time

    client, resolved_model = get_client(agent_name=agent_name)
    model_name = model or resolved_model

    last_err: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=timeout_seconds,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                import sys
                print(f"[llm_client] attempt {attempt+1} failed ({exc}), retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {1 + max_retries} attempts: {last_err}") from last_err


def chat_text_stream(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    agent_name: str | None = None,
    timeout_seconds: float = 120.0,
    enable_thinking: bool = False,
) -> Generator[tuple[str, str], None, None]:
    """Yield ``(kind, text)`` tuples from a streaming LLM call.

    *kind* is ``"thinking"`` for reasoning tokens or ``"content"`` for the
    final answer.  When *enable_thinking* is ``False`` only ``"content"``
    tuples are emitted.

    No automatic retry — streaming is not idempotent.
    """
    client, resolved_model = get_client(agent_name=agent_name)
    model_name = model or resolved_model

    extra_body = {"enable_thinking": True} if enable_thinking else None
    temperature = 1.0 if enable_thinking else 0.0

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout_seconds,
        stream=True,
        extra_body=extra_body,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Thinking tokens arrive in reasoning_content (GLM-5 / DashScope)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("thinking", reasoning)
        if delta.content:
            yield ("content", delta.content)


def _extract_json_object(text: str) -> dict[str, object]:
    """Best-effort JSON object extraction.

    Handles:
    - Clean JSON
    - Markdown code fences (```json ... ```)
    - Leading/trailing non-JSON text
    - Trailing commas before } or ]
    """
    import json
    import re

    def _loads_dict(payload: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(payload)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _strip_trailing_commas(s: str) -> str:
        return re.sub(r',\s*([}\]])', r'\1', s)

    s = text.strip()

    # Try as-is
    parsed = _loads_dict(s)
    if parsed is not None:
        return parsed

    # Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        parsed = _loads_dict(inner)
        if parsed is not None:
            return parsed
        parsed = _loads_dict(_strip_trailing_commas(inner))
        if parsed is not None:
            return parsed

    # Slice first { to last }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return a JSON object. Raw output: {s[:200]}")
    sliced = s[start : end + 1]
    parsed = _loads_dict(sliced)
    if parsed is not None:
        return parsed

    # Try with trailing comma removal
    parsed = _loads_dict(_strip_trailing_commas(sliced))
    if parsed is not None:
        return parsed

    raise ValueError(f"Model JSON root is not a valid object. Raw output: {s[:200]}")
