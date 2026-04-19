from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a config object from a dictionary, ignoring extra fields."""
        valid_fields = {f.name for f in field_info(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def field_info(cls):
    from dataclasses import fields

    return fields(cls)


@dataclass
class ModelConfig(BaseConfig):
    """Architectural configuration for the model itself."""

    model_dir: str
    max_length: int = 32768
    dtype: str = "bfloat16"
    device: str = "cuda"
    tokenizer_path: Optional[str] = None


@dataclass
class SchedulerConfig(BaseConfig):
    """Configuration for request scheduling and admission control."""

    max_batch_size: int = 32
    max_prefill_batch_size: int = 1
    max_num_requests: int = 100
    request_timeout: float = 300.0


@dataclass
class KVCacheConfig(BaseConfig):
    """Configuration for Paged Attention (Phase 3+)."""

    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB


@dataclass
class ParallelConfig(BaseConfig):
    """Configuration for distributed inference (Phase 7+)."""

    tp_size: int = 1
    pp_size: int = 1


@dataclass
class RuntimeConfig(BaseConfig):
    """The Big Config: Global configuration holding all sub-configs."""

    model: ModelConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def load(
        cls,
        yaml_path: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeConfig":
        """Unified loader: YAML -> Overrides -> Defaults."""
        data = {}
        if yaml_path:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}

        # 1. Extract sections from data
        model_data = data.get("model", {})
        scheduler_data = data.get("scheduler", {})
        kv_data = data.get("kv_cache", {})
        parallel_data = data.get("parallel", {})

        # 2. Apply CLI overrides to specific sections if provided
        # (This logic can be expanded to be more granular if needed)
        if cli_overrides:
            for k, v in cli_overrides.items():
                if v is not None:
                    if k in ["model_dir", "device", "dtype"]:
                        model_data[k] = v
                    elif k in ["max_batch_size", "request_timeout"]:
                        scheduler_data[k] = v

        # 3. Assemble objects
        # Note: model_dir is required, so we check it here
        if not model_data.get("model_dir"):
            raise ValueError("model_dir must be specified in config or CLI.")

        return cls(
            model=ModelConfig.from_dict(model_data),
            scheduler=SchedulerConfig.from_dict(scheduler_data),
            kv_cache=KVCacheConfig.from_dict(kv_data),
            parallel=ParallelConfig.from_dict(parallel_data),
        )
