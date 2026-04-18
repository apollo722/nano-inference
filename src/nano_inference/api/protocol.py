from typing import List, Optional

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """OpenAI-compatible completions request model (text-only subset)."""

    model: str = Field(description="Name of the model to use.")
    prompt: str = Field(description="The prompt(s) to generate completions for.")
    max_tokens: Optional[int] = Field(
        256, description="The maximum number of tokens to generate."
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature. 0 = greedy."
    )
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling threshold.")
    top_k: Optional[int] = Field(-1, description="Top-k sampling. -1 = disable.")
    n: Optional[int] = Field(
        1, description="Number of completions to generate (ignored in Phase 1)."
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream output (ignored in Phase 1)."
    )


class CompletionChoice(BaseModel):
    """A single completion choice."""

    text: str = Field(description="The generated text.")
    index: int = Field(description="The index of this choice.")
    finish_reason: Optional[str] = Field(
        None, description="The reason the generation stopped."
    )


class CompletionResponse(BaseModel):
    """OpenAI-compatible completions response model."""

    id: str = Field(description="Unique identifier for this completion.")
    object: str = Field("text_completion", description="Object type.")
    created: int = Field(description="Unix timestamp when the completion was created.")
    model: str = Field(description="Name of the model used.")
    choices: List[CompletionChoice] = Field(description="List of completion choices.")
    usage: Optional[dict] = Field(
        None, description="Token usage statistics (Phase 2+)."
    )
