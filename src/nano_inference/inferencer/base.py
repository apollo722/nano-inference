from abc import ABC, abstractmethod

from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerateOutput, Request


class InferencerBase(ABC):
    name: str

    @abstractmethod
    def load_model(self, model_config: ModelConfig) -> None: ...

    @abstractmethod
    def generate(self, request: Request) -> GenerateOutput: ...