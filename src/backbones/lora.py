"""LoRA (Low-Rank Adaptation) injection for nn.Linear layers.

Replaces target nn.Linear modules with LoRALinear wrappers that add a
trainable low-rank path while keeping the original weights frozen.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a parallel low-rank adapter."""

    def __init__(self, original: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        # LoRA adapters in float32 for stable optimizer updates (base stays fp16)
        self.A = nn.Linear(original.in_features, r, bias=False, dtype=torch.float32)
        self.B = nn.Linear(r, original.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        base = self.original(x)
        lora = self.B(self.A(x.float())) * self.scale
        return base + lora.to(base.dtype)


def inject_lora(model: nn.Module, target_names: list[str], r: int, alpha: float):
    """Replace matching nn.Linear layers with LoRALinear wrappers.

    Args:
        model: The model to inject LoRA into.
        target_names: List of substrings to match against module names
                      (e.g. ["out_proj", "c_fc", "c_proj"]).
        r: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_names):
            continue
        # Navigate to the parent module and replace the attribute
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha))
        replaced += 1
    return replaced


def inject_mllm_lora(backbone, lora_cfg: dict) -> None:
    """Inject LoRA into an InternVL2 model's ViT and/or LLM components.

    Args:
        backbone: Module with a `.model` attribute containing vision_model,
                  language_model, and mlp1.
        lora_cfg: Dict with keys: rank, alpha, component (vit|llm|both),
                  vit_targets, llm_targets, train_projector.
    """
    component = lora_cfg.get("component", "both")

    if component in ("vit", "both"):
        vit_targets = lora_cfg.get("vit_targets", ["fc1", "fc2"])
        n = inject_lora(backbone.model.vision_model, vit_targets,
                        r=lora_cfg["rank"], alpha=lora_cfg["alpha"])
        print(f"  LoRA (ViT): injected {n} adapters")

    if component in ("llm", "both"):
        llm_targets = lora_cfg.get("llm_targets", ["w1", "w2", "w3"])
        n = inject_lora(backbone.model.language_model, llm_targets,
                        r=lora_cfg["rank"], alpha=lora_cfg["alpha"])
        print(f"  LoRA (LLM): injected {n} adapters")

    if lora_cfg.get("train_projector", False):
        backbone.model.mlp1.requires_grad_(True)
        print("  MLP projector unfrozen")
