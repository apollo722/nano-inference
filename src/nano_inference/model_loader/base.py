from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from nano_inference.core.config import ModelConfig


@dataclass
class LoadReport:
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]


class BaseModelLoader(ABC):
    @abstractmethod
    def can_load(self, hf_config) -> bool: ...

    @abstractmethod
    def to_decoder_config(
        self,
        hf_config,
        runtime_config: ModelConfig | None = None,
    ) -> Any: ...

    @abstractmethod
    def remap_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        decoder_config: Any,
    ) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def build_model(self, decoder_config: Any) -> nn.Module: ...
