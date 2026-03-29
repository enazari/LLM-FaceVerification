"""InternViT-300M wrapper for face verification (Track 1).

Extracts the vision encoder from InternVL2-2B. Accepts [B, 3, 112, 112]
in [-1, 1], resizes to 448x448 with ImageNet normalization.
Returns [B, 1024] via mean pooling over patch tokens.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_CACHE_DIR = os.path.join(
    os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    "checkpoints",
)

# ImageNet normalization
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _patch_grad_ckpt(encoder):
    """Patch InternViT encoder to use use_reentrant=False in gradient checkpointing.

    The upstream InternVL2 code uses use_reentrant=True (default), which requires
    all inputs to have requires_grad=True. With LoRA (frozen base + trainable adapters),
    the input hidden_states tensor doesn't require grad, so gradients silently vanish.
    """
    import functools
    original_forward = encoder.forward

    @functools.wraps(original_forward)
    def patched_forward(self_enc, *args, **kwargs):
        import torch.utils.checkpoint as ckpt
        orig_ckpt = ckpt.checkpoint

        def ckpt_no_reentrant(fn, *a, **kw):
            kw.setdefault("use_reentrant", False)
            return orig_ckpt(fn, *a, **kw)

        ckpt.checkpoint = ckpt_no_reentrant
        try:
            return original_forward(*args, **kwargs)
        finally:
            ckpt.checkpoint = orig_ckpt

    import types
    encoder.forward = types.MethodType(patched_forward, encoder)


class InternViTBackbone(nn.Module):
    """InternViT-300M-448px from InternVL2-2B.

    Accepts [B, 3, 112, 112] in [-1, 1] (pipeline convention).
    Internally resizes to 448x448 and applies ImageNet normalization.
    Returns [B, 1024] (not L2-normalized).
    """

    def __init__(self, model_path="OpenGVLab/InternVL2-2B", pool="mean"):
        super().__init__()
        from transformers import AutoModel

        # Use local cache if available
        local_path = os.path.join(_CACHE_DIR, "InternVL2-2B")
        load_path = local_path if os.path.isdir(local_path) else model_path

        full_model = AutoModel.from_pretrained(
            load_path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        self.vit = full_model.vision_model
        del full_model  # free LLM + projector memory

        self.vit.requires_grad_(False)
        if hasattr(self.vit, "encoder"):
            _patch_grad_ckpt(self.vit.encoder)
        self.pool = pool
        self.register_buffer("img_mean", _IN_MEAN)
        self.register_buffer("img_std", _IN_STD)

    def forward(self, x):
        # x: [B, 3, 112, 112] in [-1, 1]
        x = x * 0.5 + 0.5  # -> [0, 1]
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        x = x.to(next(self.vit.parameters()).dtype)

        out = self.vit(pixel_values=x, output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state  # [B, 1025, 1024] (CLS + 1024 patches)

        if self.pool == "cls":
            return hidden[:, 0]               # [B, 1024]
        else:
            return hidden[:, 1:].mean(dim=1)  # [B, 1024] mean over patches


def build_internvit(cfg):
    pool = cfg["backbone"].get("pool", "mean")
    model_path = cfg["backbone"].get("model_path", "OpenGVLab/InternVL2-2B")
    backbone = InternViTBackbone(model_path=model_path, pool=pool)

    lora_cfg = cfg["backbone"].get("lora")
    if lora_cfg:
        from src.backbones.lora import inject_lora
        n = inject_lora(
            backbone.vit, lora_cfg["targets"],
            r=lora_cfg["rank"], alpha=lora_cfg["alpha"],
        )
        print(f"  LoRA: injected {n} adapters (rank={lora_cfg['rank']})")

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  InternViT: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    return backbone
