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
    stream: Optional[bool] = Field(False, description="Whether to stream output.")


class CompletionChoice(BaseModel):
    """A single completion choice."""

    text: str = Field(description="The generated text.")
    index: int = Field(description="The index of this choice.")
    finish_reason: Optional[str] = Field(
        None, description="The reason the generation stopped."
    )


class ChatCompletionMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(
        description="The role of the message author (system, user, assistant)."
    )
    content: str = Field(description="The content of the message.")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completions request model."""

    model: str = Field(description="Name of the model to use.")
    messages: List[ChatCompletionMessage] = Field(
        description="A list of messages comprising the conversation so far."
    )
    max_tokens: Optional[int] = Field(
        256, description="The maximum number of tokens to generate."
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature. 0 = greedy."
    )
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling threshold.")
    top_k: Optional[int] = Field(-1, description="Top-k sampling. -1 = disable.")
    stream: Optional[bool] = Field(False, description="Whether to stream output.")


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    message: Optional[ChatCompletionMessage] = Field(
        None, description="The assistant message."
    )
    delta: Optional[ChatCompletionMessage] = Field(
        None, description="The incremental delta for streaming."
    )
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
    usage: Optional[dict] = Field(None, description="Token usage statistics.")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completions response model."""

    id: str = Field(description="Unique identifier for this completion.")
    object: str = Field(
        "chat.completion",
        description="Object type (chat.completion or chat.completion.chunk).",
    )
    created: int = Field(description="Unix timestamp when the completion was created.")
    model: str = Field(description="Name of the model used.")
    choices: List[ChatCompletionChoice] = Field(
        description="List of completion choices."
    )
    usage: Optional[dict] = Field(None, description="Token usage statistics.")
