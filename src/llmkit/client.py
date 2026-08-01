from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import anthropic
import openai

from .tracing import record_attempt, traced

load_dotenv()


CLIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        ValidationError,
        openai.APIConnectionError,
        openai.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
    )

PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-fable-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gpt-5.6-sol": (2.50, 10.00),
    "gpt-5.6-luna": (0.15, 0.60),
}
DEFAULT_PRICING: tuple[float, float] = (1.00, 3.00)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = PRICING.get(model, DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
    return round(cost, 6)


class ClientResponse(BaseModel):
    model: str = Field(default="",description="Model Name/Used")
    output: str = Field(default="", description="Output from model")
    input_tokens: int = Field(default=0,description="Input Tokens")
    output_tokens: int = Field(default=0, description="Output Token")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


class Client:    
    @traced
    @retry(
        retry=retry_if_exception_type(CLIENT_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        reraise=True,
        before=record_attempt,
    )
    def create(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int = 512,
    ) -> ClientResponse:
        if model.startswith("claude"):
            client = anthropic.Anthropic()

            response = client.messages.parse(
                model=model,
                max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_format=ClientResponse,
            )

            output = response.parsed_output

        elif model.startswith("gpt"):
            client = openai.OpenAI()

            response = client.responses.parse(
                model=model,
                max_output_tokens=max_output_tokens,
                input=[{"role": "user", "content": prompt}],
                text_format=ClientResponse,
            )

            output = response.output_parsed

        else:
            raise ValueError(f"Unsupported model: {model}")

        output.model = response.model
        output.input_tokens = response.usage.input_tokens
        output.output_tokens = response.usage.output_tokens
        output.cost_usd = _estimate_cost_usd(
            response.model, output.input_tokens, output.output_tokens
        )

        return output