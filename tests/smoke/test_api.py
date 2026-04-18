import pytest
from fastapi.testclient import TestClient
from nano_inference.api.server import create_app
from nano_inference.core.config import ModelConfig

from tests.utils import ensure_test_model_downloaded


@pytest.mark.smoke
@pytest.mark.skipif(
    not ensure_test_model_downloaded("Qwen/Qwen3-0.6B"),
    reason="Test model not downloaded",
)
def test_api_v1_completions():
    """Test that /v1/completions endpoint works end-to-end through the full stack."""
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    app = create_app(config, inferencer_type="torch")

    # Use TestClient within the app context to ensure lifespan is executed
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "qwen3-0.6b",
                "prompt": "Count: one, two, three,",
                "max_tokens": 4,
                "temperature": 0.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "qwen3-0.6b"
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert "text" in choice
        assert len(choice["text"]) > 0
