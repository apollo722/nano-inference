import os
import pickle
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import torch
from nano_inference.envs.utils import ENV_UTILS_PICKLE_DUMP_ENABLED

T = TypeVar("T")
TEST_FILES_DIR = Path(__file__).resolve().parents[3] / "test-files"


def dump_output_pickle(
    name: Optional[str] = None,
    subdir: str = "tensors",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            if (
                os.environ.get(ENV_UTILS_PICKLE_DUMP_ENABLED, "").strip().lower()
                == "true"
            ):
                # Determine the dump name
                dump_name = name
                if dump_name is None:
                    # Try to get class name and method name
                    if args and hasattr(args[0], "__class__"):
                        class_name = args[0].__class__.__name__
                        method_name = func.__name__
                        dump_name = f"{class_name}.{method_name}"
                    else:
                        dump_name = func.__name__

                dump_pickle(dump_name, result, subdir=subdir)

            return result

        return cast(Callable[..., T], wrapper)

    return decorator


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_serializable(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    return value


def dump_pickle(
    name: str,
    value: Any,
    subdir: str = "tensors",
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_dir = TEST_FILES_DIR / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = name.replace("/", "--")
    file_name = f"{timestamp}-{safe_name}.pkl"
    output_path = output_dir / file_name

    with output_path.open("wb") as f:
        pickle.dump(_to_serializable(value), f)

    return str(output_path)


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def compare_pickles(left_path: str, right_path: str) -> Dict[str, Any]:
    left = load_pickle(left_path)
    right = load_pickle(right_path)

    result: Dict[str, Any] = {
        "left_path": left_path,
        "right_path": right_path,
        "left_type": type(left).__name__,
        "right_type": type(right).__name__,
    }

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        result["left_shape"] = tuple(left.shape)
        result["right_shape"] = tuple(right.shape)
        result["cosine"] = (
            cosine_similarity(left, right) if left.numel() == right.numel() else None
        )
        return result

    if isinstance(left, list) and isinstance(right, list):
        result["left_len"] = len(left)
        result["right_len"] = len(right)
        comparisons = []
        for a, b in zip(left, right):
            item = {
                "left_type": type(a).__name__,
                "right_type": type(b).__name__,
            }
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                item["left_shape"] = tuple(a.shape)
                item["right_shape"] = tuple(b.shape)
                item["cosine"] = (
                    cosine_similarity(a, b) if a.numel() == b.numel() else None
                )
            else:
                item["equal"] = a == b
            comparisons.append(item)
        result["items"] = comparisons
        return result

    result["equal"] = left == right
    return result
