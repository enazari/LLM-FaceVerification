"""Full InternVL2-2B wrapper for face verification (Track 3).

Siamese usage: each face image goes through the entire MLLM pipeline
independently (ViT + pixel shuffle + MLP projector + LLM).
Returns LLM hidden states as face embeddings.

Accepts [B, 3, 112, 112] in [-1, 1].
Returns [B, 2048] via pooling over LLM hidden states.
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


class InternVLBackbone(nn.Module):
    """Full InternVL2-2B: ViT + pixel shuffle + MLP projector + LLM.

    Accepts [B, 3, 112, 112] in [-1, 1] (pipeline convention).
    Returns [B, 2048] via pooling over LLM hidden states.
    """

    def __init__(self, model_path="OpenGVLab/InternVL2-2B",
                 pool="visual_mean",
                 prompt="Describe this person's facial features for identification."):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        local_path = os.path.join(_CACHE_DIR, "InternVL2-2B")
        load_path = local_path if os.path.isdir(local_path) else model_path

        self.model = AutoModel.from_pretrained(
            load_path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        self.model.requires_grad_(False)
        self.pool = pool

        # Pre-tokenize prompt template
        tokenizer = AutoTokenizer.from_pretrained(
            load_path, trust_remote_code=True, use_fast=False,
        )
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        num_image_tokens = 256  # (448/14)^2 * downsample_ratio^2

        text_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        bos = tokenizer.bos_token_id
        input_ids = [bos] + [img_context_token_id] * num_image_tokens + text_tokens

        self.register_buffer("input_ids_template", torch.tensor([input_ids], dtype=torch.long))
        self.register_buffer("img_mean", _IN_MEAN)
        self.register_buffer("img_std", _IN_STD)
        self.num_visual_tokens = num_image_tokens

        # Memory optimization — use_reentrant=False required for LoRA
        self.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(self, x):
        """x: [B, 3, 112, 112] in [-1, 1]. Returns [B, 2048]."""
        B = x.size(0)

        # Preprocess
        x = x * 0.5 + 0.5
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        x = x.to(next(self.model.vision_model.parameters()).dtype)

        # Visual features through ViT + pixel shuffle + MLP
        vit_embeds = self.model.extract_feature(x)  # [B, 256, 2048]

        # Build input embeddings: [BOS] + visual tokens + text tokens
        input_ids = self.input_ids_template.expand(B, -1)
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        n = self.num_visual_tokens
        # Use cat instead of in-place assignment to preserve gradient flow
        input_embeds = torch.cat([
            text_embeds[:, :1, :],        # BOS
            vit_embeds,                     # visual tokens (has grad from LoRA)
            text_embeds[:, 1 + n:, :],     # text tokens
        ], dim=1)

        # Forward through LLM
        attention_mask = torch.ones(B, input_ids.size(1), device=x.device, dtype=torch.long)
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]  # [B, seq_len, 2048]

        # Pool to get single embedding
        if self.pool == "visual_mean":
            emb = hidden[:, 1:1 + n, :].mean(dim=1)
        elif self.pool == "all_mean":
            emb = hidden.mean(dim=1)
        elif self.pool == "last_token":
            emb = hidden[:, -1, :]
        else:
            emb = hidden[:, 1:1 + n, :].mean(dim=1)

        return emb  # [B, 2048]


def build_internvl(cfg):
    model_path = cfg["backbone"].get("model_path", "OpenGVLab/InternVL2-2B")
    pool = cfg["backbone"].get("pool", "visual_mean")
    prompt = cfg["backbone"].get(
        "prompt", "Describe this person's facial features for identification.")
    backbone = InternVLBackbone(model_path=model_path, pool=pool, prompt=prompt)

    lora_cfg = cfg["backbone"].get("lora")
    if lora_cfg:
        from src.backbones.lora import inject_lora
        component = lora_cfg.get("component", "both")
        total_replaced = 0

        if component in ("vit", "both"):
            vit_targets = lora_cfg.get("vit_targets", ["fc1", "fc2"])
            n = inject_lora(backbone.model.vision_model, vit_targets,
                            r=lora_cfg["rank"], alpha=lora_cfg["alpha"])
            total_replaced += n
            print(f"  LoRA (ViT): injected {n} adapters")

        if component in ("llm", "both"):
            llm_targets = lora_cfg.get("llm_targets", ["w1", "w2", "w3"])
            n = inject_lora(backbone.model.language_model, llm_targets,
                            r=lora_cfg["rank"], alpha=lora_cfg["alpha"])
            total_replaced += n
            print(f"  LoRA (LLM): injected {n} adapters")

        if lora_cfg.get("train_projector", False):
            backbone.model.mlp1.requires_grad_(True)
            print("  MLP projector unfrozen")

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  InternVL: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    return backbone
