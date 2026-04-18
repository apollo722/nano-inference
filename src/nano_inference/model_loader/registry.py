from __future__ import annotations

from nano_inference.model_loader.base import BaseModelLoader
from nano_inference.model_loader.qwen3 import Qwen3Loader
from transformers import AutoConfig


def select_loader(hf_config) -> BaseModelLoader:
    candidates: list[BaseModelLoader] = [
        Qwen3Loader(),
    ]
    for loader in candidates:
        if loader.can_load(hf_config):
            return loader
    raise ValueError(
        f"No compatible loader found for model_type={getattr(hf_config, 'model_type', None)} "
        f"architectures={getattr(hf_config, 'architectures', None)}"
    )


def load_hf_config(model_name_or_path: str):
    return AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
