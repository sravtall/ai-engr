from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ClientResponse:
    model: str
    output: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any]

@dataclass
class Client:
    anthropic_client: anthropic.Anthropic = anthropic.Anthropic()
    openai_client: openai.OpenAI = openai.OpenAI()

    def create(self, model: str, input: str, max_output_tokens: int = 500):
        if model.startswith("claude"):
            res = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": input}],
            )

            return ClientResponse(
                model=model,
                output=res.content[0].text,
                input_tokens=res.usage.input_tokens,
                output_tokens=res.usage.output_tokens,
                raw_response=res.to_dict(),
            )
        
        elif model.startswith("gpt"):
            res = self.openai_client.responses.create(
                model=model,
                input=input,
                max_output_tokens=max_output_tokens,
            )

            return ClientResponse(
                model=model,
                output=res.output_text,
                input_tokens=res.usage.input_tokens,
                output_tokens=res.usage.output_tokens,
                raw_response=res.to_dict(),
            )
        else:
            raise ValueError(f"Unsupported model: {model}") 