import pytest
from fastapi.testclient import TestClient
from nano_inference.api.server import create_app
from nano_inference.core.config import ModelConfig, RuntimeConfig

from tests.utils import ensure_test_model_downloaded


@pytest.mark.smoke
@pytest.mark.skipif(
    not ensure_test_model_downloaded("Qwen/Qwen3-0.6B"),
    reason="Test model not downloaded",
)
def test_api_v1_completions():
    """Test that /v1/completions endpoint works end-to-end through the full stack."""
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    model_config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    config = RuntimeConfig(model=model_config)
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


@pytest.mark.smoke
@pytest.mark.skipif(
    not ensure_test_model_downloaded("Qwen/Qwen3-0.6B"),
    reason="Test model not downloaded",
)
def test_api_v1_completions_streaming():
    """Test that /v1/completions endpoint works with streaming."""
    import json

    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    model_config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    config = RuntimeConfig(model=model_config)
    app = create_app(config, inferencer_type="torch")

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "qwen3-0.6b",
                "prompt": "Count: one, two, three,",
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200

            chunks = []
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    chunks.append(data)

            assert len(chunks) > 0
            assert chunks[0]["object"] == "text_completion"
            # Ensure we got some text
            text = "".join(c["choices"][0]["text"] for c in chunks)
            assert len(text) > 0


@pytest.mark.smoke
@pytest.mark.skipif(
    not ensure_test_model_downloaded("Qwen/Qwen3-0.6B"),
    reason="Test model not downloaded",
)
def test_api_v1_chat_completions():
    """Test that /v1/chat/completions endpoint works end-to-end."""
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    model_config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    config = RuntimeConfig(model=model_config)
    app = create_app(config, inferencer_type="torch")

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 4,
                "temperature": 0.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.smoke
@pytest.mark.skipif(
    not ensure_test_model_downloaded("Qwen/Qwen3-0.6B"),
    reason="Test model not downloaded",
)
def test_api_v1_chat_completions_streaming():
    """Test that /v1/chat/completions endpoint works with streaming."""
    import json

    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    model_config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    config = RuntimeConfig(model=model_config)
    app = create_app(config, inferencer_type="torch")

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200

            chunks = []
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    chunks.append(data)

            assert len(chunks) > 0
            assert chunks[0]["object"] == "chat.completion.chunk"
            # Ensure we got some text from deltas
            text = "".join(
                c["choices"][0]["delta"]["content"]
                for c in chunks
                if c["choices"][0]["delta"].get("content")
            )
            assert len(text) > 0
