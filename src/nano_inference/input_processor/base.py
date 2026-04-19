import abc
from typing import Any, Dict, List, Optional, Type, TypeVar

from nano_inference.core.request import GenerationInputs
from transformers import PreTrainedTokenizerBase

TInputProcessor = TypeVar("TInputProcessor", bound="BaseInputProcessor")

CLS_TO_INPUT_PROCESSOR: Dict[str, Type["BaseInputProcessor"]] = {}


def register_input_processor(cls: Type[TInputProcessor]) -> Type[TInputProcessor]:
    CLS_TO_INPUT_PROCESSOR[cls.name] = cls
    return cls


def get_input_processor(name: str) -> Type["BaseInputProcessor"]:
    return CLS_TO_INPUT_PROCESSOR[name]


class BaseInputProcessor(abc.ABC):
    name: str

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len is not None else (1 << 64) - 1

    @abc.abstractmethod
    def encode(
        self,
        messages: List[Dict[str, str]],
        max_prompt_tokens: Optional[int] = None,
        **kwargs,
    ) -> GenerationInputs:
        raise NotImplementedError

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
