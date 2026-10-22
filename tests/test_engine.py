import pytest
import torch

from sfraud.training.engine import (
    build_loss_fn,
    masked_bce_with_logits,
    masked_focal_loss_with_logits,
)

B, T = 4, 6


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, generator=g)
    y = (torch.rand(B, T, generator=g) < 0.3).float()
    mask = torch.ones(B, T)
    mask[:, 4:] = 0.0  # last two positions are padding
    return logits, y, mask


def test_focal_loss_is_finite_and_non_negative():
    logits, y, mask = _batch()
    loss = masked_focal_loss_with_logits(logits, y, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_loss_ignores_padded_positions():
    logits, y, mask = _batch()
    loss_before = masked_focal_loss_with_logits(logits, y, mask)

    logits_perturbed, y_perturbed = logits.clone(), y.clone()
    logits_perturbed[mask == 0] = 50.0  # wildly wrong predictions on padding only
    y_perturbed[mask == 0] = 1.0
    loss_after = masked_focal_loss_with_logits(logits_perturbed, y_perturbed, mask)

    assert torch.allclose(loss_before, loss_after)


def test_focal_loss_gamma_zero_reduces_to_alpha_weighted_bce():
    logits, y, mask = _batch()
    focal = masked_focal_loss_with_logits(logits, y, mask, alpha=0.5, gamma=0.0)
    bce = masked_bce_with_logits(logits, y, mask, pos_weight=torch.tensor(1.0))
    assert torch.allclose(focal, 0.5 * bce, atol=1e-6)


def test_build_loss_fn_dispatches_to_bce():
    logits, y, mask = _batch()
    pos_weight = torch.tensor(5.0)
    loss_fn = build_loss_fn({"loss": "bce"}, pos_weight)
    assert torch.allclose(loss_fn(logits, y, mask), masked_bce_with_logits(logits, y, mask, pos_weight))


def test_build_loss_fn_defaults_to_bce_when_unset():
    logits, y, mask = _batch()
    pos_weight = torch.tensor(5.0)
    loss_fn = build_loss_fn({}, pos_weight)
    assert torch.allclose(loss_fn(logits, y, mask), masked_bce_with_logits(logits, y, mask, pos_weight))


def test_build_loss_fn_dispatches_to_focal_with_config_params():
    logits, y, mask = _batch()
    cfg = {"loss": "focal", "focal_alpha": 0.4, "focal_gamma": 1.5}
    loss_fn = build_loss_fn(cfg, torch.tensor(5.0))
    expected = masked_focal_loss_with_logits(logits, y, mask, alpha=0.4, gamma=1.5)
    assert torch.allclose(loss_fn(logits, y, mask), expected)


def test_build_loss_fn_rejects_unknown_loss():
    with pytest.raises(ValueError):
        build_loss_fn({"loss": "hinge"}, torch.tensor(1.0))
