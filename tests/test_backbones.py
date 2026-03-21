"""Smoke tests for InternVL backbones using mocked model loading.

These tests verify forward pass shapes and interface correctness
without requiring actual InternVL2-2B weights.
"""

from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock objects that mimic InternVL2-2B's interface
# ---------------------------------------------------------------------------

class MockVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Linear(3, 1024)

    def forward(self, pixel_values, **kwargs):
        B = pixel_values.shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.randn(B, 1025, 1024)
        return out


class MockEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(92547, 2048))

    def forward(self, input_ids):
        return self.weight[input_ids]


class MockLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._embedding = MockEmbedding()

    def get_input_embeddings(self):
        return self._embedding

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, return_dict=True):
        B, S, D = inputs_embeds.shape
        out = MagicMock()
        h = torch.randn(B, S, 2048)
        out.hidden_states = (h,)  # tuple of hidden states, [-1] = last layer
        out.logits = torch.randn(B, S, 92547)
        return out


class MockInternVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = MockVisionModel()
        self.language_model = MockLanguageModel()
        self.mlp1 = nn.Linear(4096, 2048)
        self.img_context_token_id = 92546

    def extract_feature(self, x):
        B = x.shape[0]
        return torch.randn(B, 256, 2048)


class MockTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [100, 200, 300, 400]

    def convert_tokens_to_ids(self, token):
        return 92546  # mock IMG_CONTEXT token id


def _mock_auto_model_from_pretrained(*args, **kwargs):
    return MockInternVL()


def _mock_auto_tokenizer_from_pretrained(*args, **kwargs):
    return MockTokenizer()


# Patch at the transformers level (imports are local to __init__)
_MODEL_PATCH = "transformers.AutoModel.from_pretrained"
_TOKENIZER_PATCH = "transformers.AutoTokenizer.from_pretrained"


# ---------------------------------------------------------------------------
# Track 1: InternViT
# ---------------------------------------------------------------------------

@patch(_MODEL_PATCH, _mock_auto_model_from_pretrained)
def test_internvit_forward_shape():
    """InternViTBackbone produces [B, 1024] output."""
    from src.backbones.internvit import InternViTBackbone

    backbone = InternViTBackbone(model_path="mock")
    x = torch.randn(2, 3, 112, 112)
    out = backbone(x)
    assert out.shape == (2, 1024)


@patch(_MODEL_PATCH, _mock_auto_model_from_pretrained)
def test_internvit_cls_pool():
    """InternViTBackbone with CLS pooling returns [B, 1024]."""
    from src.backbones.internvit import InternViTBackbone

    backbone = InternViTBackbone(model_path="mock", pool="cls")
    x = torch.randn(2, 3, 112, 112)
    out = backbone(x)
    assert out.shape == (2, 1024)


# ---------------------------------------------------------------------------
# Track 3: InternVL siamese
# ---------------------------------------------------------------------------

@patch(_TOKENIZER_PATCH, _mock_auto_tokenizer_from_pretrained)
@patch(_MODEL_PATCH, _mock_auto_model_from_pretrained)
def test_internvl_forward_shape():
    """InternVLBackbone produces [B, 2048] output."""
    from src.backbones.internvl import InternVLBackbone

    backbone = InternVLBackbone(model_path="mock")
    x = torch.randn(2, 3, 112, 112)
    out = backbone(x)
    assert out.shape == (2, 2048)


# ---------------------------------------------------------------------------
# Track 2: InternVL pairwise
# ---------------------------------------------------------------------------

@patch(_TOKENIZER_PATCH, _mock_auto_tokenizer_from_pretrained)
@patch(_MODEL_PATCH, _mock_auto_model_from_pretrained)
def test_internvl_pair_forward_shape():
    """InternVLPairBackbone produces [B, vocab_size] logits."""
    from src.backbones.internvl_pair import InternVLPairBackbone

    backbone = InternVLPairBackbone(model_path="mock")
    img_a = torch.randn(2, 3, 112, 112)
    img_b = torch.randn(2, 3, 112, 112)
    logits = backbone(img_a, img_b)
    assert logits.shape[0] == 2
    assert logits.dim() == 2


@patch(_TOKENIZER_PATCH, _mock_auto_tokenizer_from_pretrained)
@patch(_MODEL_PATCH, _mock_auto_model_from_pretrained)
def test_internvl_pair_yes_no_scores():
    """get_yes_no_scores returns P(Yes) in [0, 1]."""
    from src.backbones.internvl_pair import InternVLPairBackbone

    backbone = InternVLPairBackbone(model_path="mock")
    logits = torch.randn(4, 92547)
    scores = backbone.get_yes_no_scores(logits)
    assert scores.shape == (4,)
    assert (scores >= 0).all() and (scores <= 1).all()
