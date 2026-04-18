import os
from pathlib import Path

from huggingface_hub import snapshot_download

TEST_MODELS_DIR = Path(__file__).resolve().parents[2] / "test-models"


def ensure_test_model_downloaded(model_id: str) -> str:
    sanitized_name = model_id.replace("/", "--")
    local_model_dir = TEST_MODELS_DIR / sanitized_name

    if (local_model_dir / "config.json").exists():
        return str(local_model_dir)

    local_model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_model_dir),
    )
    return str(local_model_dir)
