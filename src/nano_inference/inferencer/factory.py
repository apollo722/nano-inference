from typing import Dict, Type, TypeVar

from nano_inference.core.config import ModelConfig
from nano_inference.inferencer.base import InferencerBase

TInferencer = TypeVar("TInferencer", bound="InferencerBase")

INFERENCER_REGISTRY: Dict[str, Type["InferencerBase"]] = {}


def register_inferencer(cls: Type[TInferencer]) -> Type[TInferencer]:
    INFERENCER_REGISTRY[cls.name] = cls
    return cls


def get_inferencer(name: str) -> Type["InferencerBase"]:
    return INFERENCER_REGISTRY[name]


class InferencerFactory:
    @staticmethod
    def create(inferencer_type: str, model_config: ModelConfig) -> InferencerBase:
        if inferencer_type not in INFERENCER_REGISTRY:
            raise ValueError(f"Unknown inferencer type: {inferencer_type}")

        inferencer_cls = INFERENCER_REGISTRY[inferencer_type]
        inferencer = inferencer_cls()
        inferencer.load_model(model_config)
        return inferencer
