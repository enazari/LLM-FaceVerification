"""Tests for backbone factory."""

import pytest

from src.backbones.factory import build_backbone


def test_unknown_backbone_raises():
    """Unknown backbone name raises ValueError."""
    cfg = {"backbone": {"name": "nonexistent", "embedding_dim": 512, "dropout": 0.0}}
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone(cfg)
