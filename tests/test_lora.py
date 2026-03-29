"""Tests for LoRA injection."""

import torch
import torch.nn as nn

from src.backbones.lora import LoRALinear, inject_lora


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.other = nn.Linear(64, 32)

    def forward(self, x):
        return self.other(self.fc2(torch.relu(self.fc1(x))))


def test_lora_linear_shape():
    """LoRALinear preserves input/output dimensions."""
    orig = nn.Linear(64, 128)
    lora = LoRALinear(orig, r=4, alpha=4)
    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 128)


def test_lora_linear_frozen_original():
    """Original weight is frozen, A and B are trainable."""
    orig = nn.Linear(64, 128)
    lora = LoRALinear(orig, r=4, alpha=4)
    assert not lora.original.weight.requires_grad
    assert lora.A.weight.requires_grad
    assert lora.B.weight.requires_grad


def test_lora_linear_init_zero():
    """B is initialized to zero, so LoRA output starts as zero."""
    orig = nn.Linear(64, 128)
    lora = LoRALinear(orig, r=4, alpha=4)
    x = torch.randn(2, 64)
    # At init, LoRA should not change the output (B=0)
    with torch.no_grad():
        out_orig = orig(x)
        out_lora = lora(x)
    torch.testing.assert_close(out_orig, out_lora)


def test_inject_lora_replaces_targets():
    """inject_lora replaces matching Linear layers with LoRALinear."""
    model = DummyMLP()
    n = inject_lora(model, target_names=["fc1", "fc2"], r=4, alpha=4)
    assert n == 2
    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, LoRALinear)
    # 'other' should not be replaced
    assert isinstance(model.other, nn.Linear)


def test_inject_lora_no_match():
    """inject_lora returns 0 when no layers match."""
    model = DummyMLP()
    n = inject_lora(model, target_names=["nonexistent"], r=4, alpha=4)
    assert n == 0


def test_inject_lora_forward():
    """Model still works after LoRA injection."""
    model = DummyMLP()
    inject_lora(model, target_names=["fc1", "fc2"], r=4, alpha=4)
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 32)


def test_inject_lora_trainable_params():
    """Only LoRA params are trainable after injection."""
    model = DummyMLP()
    model.requires_grad_(False)
    inject_lora(model, target_names=["fc1"], r=4, alpha=4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 2  # A and B
    assert all("fc1" in n for n in trainable)
