from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_dir: str
    max_length: int = 32768
    dtype: str = "bfloat16"
    device: str = "cuda"
