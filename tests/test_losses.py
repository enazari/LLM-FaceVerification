"""Tests for loss functions."""

import torch
import torch.nn.functional as F

from src.losses.infonce import InfoNCELoss
from src.losses.factory import build_loss


def test_infonce_output_shape():
    """InfoNCE returns a scalar loss."""
    loss_fn = InfoNCELoss(temperature=0.07)
    emb = F.normalize(torch.randn(8, 128), dim=1)
    labels = torch.arange(8)  # not used internally but passed by convention
    loss = loss_fn(emb, labels)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_infonce_perfect_pairs():
    """Loss should be low when paired embeddings are identical."""
    loss_fn = InfoNCELoss(temperature=0.07)
    # 4 pairs: (0,1), (2,3), (4,5), (6,7)
    base = F.normalize(torch.randn(4, 128), dim=1)
    emb = base.repeat_interleave(2, dim=0)  # duplicate each row
    labels = torch.arange(8)
    loss = loss_fn(emb, labels)
    assert loss.item() < 1.0  # should be quite low


def test_infonce_gradient_flows():
    """Gradients flow through InfoNCE."""
    loss_fn = InfoNCELoss(temperature=0.07)
    emb = torch.randn(8, 128, requires_grad=True)
    emb_norm = F.normalize(emb, dim=1)
    labels = torch.arange(8)
    loss = loss_fn(emb_norm, labels)
    loss.backward()
    assert emb.grad is not None
    assert emb.grad.abs().sum() > 0


def test_build_loss_infonce():
    """build_loss returns InfoNCELoss for 'infonce'."""
    cfg = {"loss": {"name": "infonce", "temperature": 0.1}}
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, InfoNCELoss)


def test_build_loss_unknown():
    """build_loss raises on unknown loss name."""
    cfg = {"loss": {"name": "unknown"}}
    try:
        build_loss(cfg)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
