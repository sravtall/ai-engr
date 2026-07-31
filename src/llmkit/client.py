from typing import Any

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import anthropic
import instructor
import openai


class ClientResponse(BaseModel):
    model: str = Field(description="Model Name/Used")
    output: str = Field(description="Output from model")
    input_tokens: int = Field(description="Input Tokens")
    output_tokens: int = Field(description="Output Token")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


class Client:
    _MODEL_PRICING: dict[str, dict[str, float]] = {
        "claude-3-5-haiku": {"input_per_1m": 0.8, "output_per_1m": 4.0},
        "claude-3-5-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0},
        "claude-3-7-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0},
        "claude-4-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0},
        "claude-4-opus": {"input_per_1m": 15.0, "output_per_1m": 75.0},
        "gpt-4.1": {"input_per_1m": 2.0, "output_per_1m": 8.0},
        "gpt-4o": {"input_per_1m": 2.5, "output_per_1m": 10.0},
        "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6},
        "gpt-3.5": {"input_per_1m": 0.5, "output_per_1m": 1.5},
    }

    _EXCEPTIONS: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        openai.APIConnectionError,
        openai.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
    )

    def _get_client(self, model: str) -> Any:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gpt"):
            provider = "openai"
        else:
            raise ValueError(f"Unsupported model: {model}")

        try:
            return instructor.from_provider(f"{provider}/{model}")
        except Exception as exc:
            raise ValueError(f"Unsupported model: {model}") from exc

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        for model_name, pricing in self._MODEL_PRICING.items():
            if model.startswith(model_name):
                input_cost = input_tokens * pricing["input_per_1m"] / 1_000_000
                output_cost = output_tokens * pricing["output_per_1m"] / 1_000_000
                return round(input_cost + output_cost, 6)

        return 0.0

    @retry(
        retry=retry_if_exception_type(_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
    )
    def create(self, model: str, prompt: str, max_output_tokens: int = 500) -> ClientResponse:
        client = self._get_client(model)

        response = client.create(
            model=model,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_model=ClientResponse,
        )

        response.cost_usd = self._estimate_cost(
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return response