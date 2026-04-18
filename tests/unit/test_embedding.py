import torch
from nano_inference.layers.embedding import NaiveTokenEmbedding


def test_token_embedding_preserves_token_shape_and_hidden_dim():
    emb = NaiveTokenEmbedding(vocab_size=32, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    out = emb(input_ids)

    assert out.shape == (2, 3, 16)


def test_token_embedding_same_token_maps_to_same_vector():
    emb = NaiveTokenEmbedding(vocab_size=32, hidden_size=16)
    input_ids = torch.tensor([[7, 7]], dtype=torch.long)

    out = emb(input_ids)

    torch.testing.assert_close(out[:, 0, :], out[:, 1, :])
