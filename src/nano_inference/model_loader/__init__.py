from nano_inference.model_loader.base import BaseModelLoader, LoadReport
from nano_inference.model_loader.qwen3 import Qwen3Loader
from nano_inference.model_loader.registry import load_hf_config, select_loader


def decoder_config_from_hf_config(hf_config, runtime_config=None):
    loader = select_loader(hf_config)
    return loader.to_decoder_config(hf_config, runtime_config)


__all__ = [
    "BaseModelLoader",
    "LoadReport",
    "Qwen3Loader",
    "load_hf_config",
    "select_loader",
    "decoder_config_from_hf_config",
]