from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .sampling import SamplingParams


@dataclass
class GenerationInputs:
    prompt_token_ids: List[int]
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    images: Optional[List[Any]] = None
    videos: Optional[List[Any]] = None


@dataclass
class Request:
    request_id: str
    generation_inputs: GenerationInputs
    sampling_params: SamplingParams
    eos_token_id: Union[int, List[int]]
    arrival_time: float


class GenerationStage(Enum):
    PREFILL = "prefill"
    RECOMPUTE = "recompute"
    DECODE = "decode"
    FINISHED = "finished"


class FinishedReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


class GenerateQuery:
    def __init__(
        self,
        request_id: str,
        generation_inputs: GenerationInputs,
        sampling_params: SamplingParams,
        eos_token_id: Union[int, List[int]],
        arrival_time: float,
    ):
        from nano_inference.kv_cache.block import KVCacheBlock

        self.request_id = request_id
        self.generation_inputs = generation_inputs
        self.sampling_params = sampling_params
        self.output_token_ids: List[int] = []
        self.eos_token_id = eos_token_id
        self.stage = GenerationStage.PREFILL
        self.computed_length = 0
        self.arrival_time = arrival_time
        self.first_token_time: Optional[float] = None
        self.previous_tokens_len = 0  # For incremental detokenization
        self.kv_cache_block: Optional[KVCacheBlock] = None

    @classmethod
    def from_request(cls, request: Request) -> "GenerateQuery":
        return cls(
            request_id=request.request_id,
            generation_inputs=request.generation_inputs,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
        )


@dataclass
class GenerateOutput:
    output_token_ids: List[int]
    finished: bool
    finished_reason: Optional[FinishedReason] = None
    delta_text: str = ""
    full_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
