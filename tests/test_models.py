import torch

from sfraud.models.autoencoder import TransactionAutoencoder
from sfraud.models.lstm import LSTMFraudModel
from sfraud.models.transformer import TransformerFraudModel

VOCAB_SIZES = [4, 5]
N_NUMERIC = 3
B, T = 2, 6


def _random_batch():
    x_num = torch.randn(B, T, N_NUMERIC)
    x_cat = torch.stack(
        [torch.randint(0, v, (B, T)) for v in VOCAB_SIZES], dim=-1
    )
    return x_num, x_cat


def test_lstm_output_shape():
    cfg = {"embedding_dim": 4, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "bidirectional": False}
    model = LSTMFraudModel(N_NUMERIC, VOCAB_SIZES, cfg)
    x_num, x_cat = _random_batch()
    logits = model(x_num, x_cat)
    assert logits.shape == (B, T)


def test_transformer_output_shape():
    cfg = {
        "embedding_dim": 4, "d_model": 8, "n_heads": 2, "n_layers": 2,
        "dim_feedforward": 16, "dropout": 0.0,
    }
    model = TransformerFraudModel(N_NUMERIC, VOCAB_SIZES, cfg, max_seq_len=T)
    x_num, x_cat = _random_batch()
    logits, attn = model(x_num, x_cat, return_attn=True)
    assert logits.shape == (B, T)
    assert len(attn) == cfg["n_layers"]
    assert attn[0].shape == (B, T, T)


def test_transformer_is_causal():
    """Changing a future timestep's input must not change an earlier
    timestep's output -- this is the whole point of the causal mask, and a
    bug here would silently leak future information into every prediction."""
    cfg = {
        "embedding_dim": 4, "d_model": 8, "n_heads": 2, "n_layers": 2,
        "dim_feedforward": 16, "dropout": 0.0,
    }
    model = TransformerFraudModel(N_NUMERIC, VOCAB_SIZES, cfg, max_seq_len=T)
    model.eval()

    x_num, x_cat = _random_batch()
    with torch.no_grad():
        logits_before = model(x_num, x_cat)

    x_num_perturbed = x_num.clone()
    x_num_perturbed[:, -1, :] += 100.0  # blow up the LAST timestep only
    with torch.no_grad():
        logits_after = model(x_num_perturbed, x_cat)

    # every position except the perturbed last one must be unaffected
    assert torch.allclose(logits_before[:, :-1], logits_after[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_before[:, -1], logits_after[:, -1])


def test_transformer_attention_rows_sum_to_one_in_eval_mode():
    """Regression test: attention weights are a softmax over valid keys, so
    each row must sum to ~1 -- this only holds with dropout disabled
    (model.eval()). We shipped a bug where a model loaded for inference was
    left in train() mode, which silently corrupted the attention-heatmap
    plot (row sums ranged ~0.7-1.2 instead of 1.0). Fixed in
    evaluation/predict.py::load_trained_model; this locks in the invariant
    that made the bug visible in the first place."""
    cfg = {
        "embedding_dim": 4, "d_model": 8, "n_heads": 2, "n_layers": 2,
        "dim_feedforward": 16, "dropout": 0.3,  # nonzero dropout -- would fail this test if still in train() mode
    }
    model = TransformerFraudModel(N_NUMERIC, VOCAB_SIZES, cfg, max_seq_len=T)
    model.eval()
    x_num, x_cat = _random_batch()
    with torch.no_grad():
        _, attn_weights_all = model(x_num, x_cat, return_attn=True)
    for layer_attn in attn_weights_all:
        row_sums = layer_attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_autoencoder_per_row_loss_shape():
    # TransactionAutoencoder is per-transaction (flat), not per-sequence --
    # batches are (B, features), with no time dimension. See its docstring.
    cfg = {"embedding_dim": 4, "hidden_size": 8, "latent_size": 4}
    model = TransactionAutoencoder(N_NUMERIC, VOCAB_SIZES, cfg)
    x_num = torch.randn(B, N_NUMERIC)
    x_cat = torch.stack([torch.randint(0, v, (B,)) for v in VOCAB_SIZES], dim=-1)
    loss = model.per_row_loss(x_num, x_cat)
    assert loss.shape == (B,)
    assert (loss >= 0).all()
