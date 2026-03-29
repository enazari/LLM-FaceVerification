"""Tests for shared training utilities."""

import math
import os
import tempfile

import torch
import torch.nn as nn

from src.utils import cosine_lr, save_checkpoint


def test_cosine_lr_warmup():
    """LR ramps linearly during warmup."""
    model = nn.Linear(10, 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    cosine_lr(opt, epoch=0, total_epochs=10, warmup_epochs=5, base_lr=0.1)
    assert abs(opt.param_groups[0]['lr'] - 0.02) < 1e-6  # 0.1 * 1/5

    cosine_lr(opt, epoch=2, total_epochs=10, warmup_epochs=5, base_lr=0.1)
    assert abs(opt.param_groups[0]['lr'] - 0.06) < 1e-6  # 0.1 * 3/5


def test_cosine_lr_post_warmup():
    """LR follows cosine schedule after warmup."""
    model = nn.Linear(10, 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # At warmup boundary: full LR
    cosine_lr(opt, epoch=5, total_epochs=10, warmup_epochs=5, base_lr=0.1)
    expected = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 0.0))  # = 0.1
    assert abs(opt.param_groups[0]['lr'] - expected) < 1e-6

    # At end: near zero
    cosine_lr(opt, epoch=9, total_epochs=10, warmup_epochs=5, base_lr=0.1)
    progress = 4 / 5
    expected = 0.1 * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert abs(opt.param_groups[0]['lr'] - expected) < 1e-6


def test_cosine_lr_zero_epochs():
    """Handles edge case of 0 total epochs."""
    model = nn.Linear(10, 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cosine_lr(opt, epoch=0, total_epochs=0, warmup_epochs=0, base_lr=0.1)
    # Should not crash; LR = base_lr * 0.5 * (1 + cos(0)) = base_lr
    assert opt.param_groups[0]['lr'] > 0


def test_save_checkpoint_creates_file():
    """save_checkpoint writes a .pth file."""
    model = nn.Linear(10, 10)

    class FakeAccelerator:
        is_main_process = True
        def unwrap_model(self, m):
            return m

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(FakeAccelerator(), model, None, 1, tmpdir, "test")
        path = os.path.join(tmpdir, "test.pth")
        assert os.path.exists(path)
        ckpt = torch.load(path, weights_only=True)
        assert "backbone" in ckpt
        assert ckpt["epoch"] == 1


def test_save_checkpoint_with_optimizer():
    """save_checkpoint includes optimizer state when provided."""
    model = nn.Linear(10, 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    class FakeAccelerator:
        is_main_process = True
        def unwrap_model(self, m):
            return m

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(FakeAccelerator(), model, opt, 5, tmpdir, "best")
        ckpt = torch.load(os.path.join(tmpdir, "best.pth"), weights_only=True)
        assert "optimizer" in ckpt
        assert ckpt["epoch"] == 5
