import contextvars
import inspect
import json
import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

_attempt_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_attempt_var", default=1
)


def record_attempt(retry_state: Any) -> None:
    _attempt_var.set(retry_state.attempt_number)


def _serialize_payload(payload: Any) -> str | None:
    if payload is None:
        return None

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        return str(payload)


def _initialize_db() -> sqlite3.Connection:
    traces_dir = Path(__file__).resolve().parent / "traces"
    traces_dir.mkdir(exist_ok=True)

    db_path = traces_dir / "llmkit_traces.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            prompt TEXT,
            request TEXT,
            response TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            latency_ms REAL,
            retry_count INTEGER,
            retried INTEGER,
            status TEXT,
            error TEXT,
            created_at TEXT
        )
        """
    )
    connection.commit()
    return connection


def _insert_trace(
    *,
    model: str | None,
    prompt: str | None,
    request: str,
    response: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    latency_ms: float,
    retry_count: int,
    status: str,
    error: str | None,
) -> None:
    connection = _initialize_db()
    connection.execute(
        """
        INSERT INTO traces (
            model,
            prompt,
            request,
            response,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
            retry_count,
            retried,
            status,
            error,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model,
            prompt,
            request,
            response,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
            retry_count,
            int(retry_count > 1),
            status,
            error,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    connection.close()


def traced(func: F) -> F:
    func_signature = inspect.signature(func)
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        attempt_token = _attempt_var.set(1)

        bound_args = func_signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        resolved_args = bound_args.arguments

        model = resolved_args.get("model")
        prompt = resolved_args.get("prompt")

        serialized_args = {k: _serialize_payload(v) for k, v in resolved_args.items() if k != "self"}
        request_payload = json.dumps({"args": serialized_args})

        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            retry_count = _attempt_var.get()
            _attempt_var.reset(attempt_token)
            _insert_trace(
                model=model,
                prompt=prompt,
                request=request_payload,
                response=None,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_ms=latency_ms,
                retry_count=retry_count,
                status="error",
                error=f"{type(exc).__name__}: {str(exc)}",
            )
            return None

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        retry_count = _attempt_var.get()
        _attempt_var.reset(attempt_token)
        _insert_trace(
            model=model,
            prompt=prompt,
            request=request_payload,
            response=_serialize_payload(result),
            input_tokens=getattr(result, "input_tokens", None),
            output_tokens=getattr(result, "output_tokens", None),
            cost_usd=getattr(result, "cost_usd", None),
            latency_ms=latency_ms,
            retry_count=retry_count,
            status="success",
            error=None,
        )
        return result
        
    return cast(F, wrapper)
