"""Full InternVL2-2B pairwise classifier for face verification (Track 2).

Feeds two face images and a text prompt through the MLLM.
Returns logits at the answer position for "Yes"/"No" classification.
P("Yes") serves as the similarity score at eval time.
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


class InternVLPairBackbone(nn.Module):
    """Full InternVL2-2B for pairwise face verification.

    Accepts two images [B, 3, 112, 112] in [-1, 1].
    Returns logits [B, vocab_size] at the answer position.
    """

    def __init__(self, model_path="OpenGVLab/InternVL2-2B",
                 prompt="Do these two images show the same person? Answer Yes or No."):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        local_path = os.path.join(_CACHE_DIR, "InternVL2-2B")
        load_path = local_path if os.path.isdir(local_path) else model_path

        self.model = AutoModel.from_pretrained(
            load_path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        self.model.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(
            load_path, trust_remote_code=True, use_fast=False,
        )

        # Pre-tokenize prompt template
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.num_image_tokens = 256  # per image after pixel shuffle

        text_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        bos = tokenizer.bos_token_id

        # Template: [BOS] <img1 tokens×256> <img2 tokens×256> <prompt text>
        input_ids = (
            [bos]
            + [img_context_token_id] * self.num_image_tokens   # image 1
            + [img_context_token_id] * self.num_image_tokens   # image 2
            + text_tokens
        )

        self.register_buffer("input_ids_template", torch.tensor([input_ids], dtype=torch.long))
        self.register_buffer("img_mean", _IN_MEAN)
        self.register_buffer("img_std", _IN_STD)

        # Store Yes/No token IDs
        self.yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

        # Memory optimization
        self.model.language_model.gradient_checkpointing_enable()

    def forward(self, img_a, img_b):
        """
        img_a, img_b: [B, 3, 112, 112] in [-1, 1]
        Returns: logits [B, vocab_size] at the answer position
        """
        B = img_a.size(0)

        # Preprocess both images
        imgs = torch.cat([img_a, img_b], dim=0)  # [2B, 3, 112, 112]
        imgs = imgs * 0.5 + 0.5
        imgs = F.interpolate(imgs, size=(448, 448), mode="bilinear", align_corners=False)
        imgs = (imgs - self.img_mean) / self.img_std
        imgs = imgs.to(next(self.model.vision_model.parameters()).dtype)

        # Extract visual features for all images
        vit_embeds = self.model.extract_feature(imgs)  # [2B, 256, 2048]
        vit_a = vit_embeds[:B]   # [B, 256, 2048]
        vit_b = vit_embeds[B:]   # [B, 256, 2048]

        # Build input embeddings
        input_ids = self.input_ids_template.expand(B, -1)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        input_embeds = input_embeds.clone()

        # Replace image placeholder positions
        n = self.num_image_tokens
        input_embeds[:, 1:1 + n, :] = vit_a          # positions 1..256: image 1
        input_embeds[:, 1 + n:1 + 2 * n, :] = vit_b  # positions 257..512: image 2

        # Forward through LLM
        attention_mask = torch.ones(B, input_ids.size(1), device=img_a.device, dtype=torch.long)
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Logits at the last position (answer position)
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        return logits

    def get_yes_no_scores(self, logits):
        """Extract P(Yes) from logits for use as similarity score."""
        yes_no_logits = logits[:, [self.yes_token_id, self.no_token_id]]  # [B, 2]
        probs = F.softmax(yes_no_logits, dim=1)
        return probs[:, 0]  # P(Yes)


def build_internvl_pair(cfg):
    model_path = cfg["backbone"].get("model_path", "OpenGVLab/InternVL2-2B")
    prompt = cfg["backbone"].get(
        "prompt", "Do these two images show the same person? Answer Yes or No.")
    backbone = InternVLPairBackbone(model_path=model_path, prompt=prompt)

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
    print(f"  InternVL-Pair: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    return backbone
