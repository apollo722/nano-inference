from abc import ABC, abstractmethod
from typing import List

from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerateOutput, Request


class InferencerBase(ABC):
    name: str

    @abstractmethod
    def load_model(self, model_config: ModelConfig) -> None: ...

    @abstractmethod
    def generate(self, request: Request) -> GenerateOutput: ...

    def generate_batch(self, requests: List[Request]) -> List[GenerateOutput]:
        return [self.generate(request) for request in requests]
