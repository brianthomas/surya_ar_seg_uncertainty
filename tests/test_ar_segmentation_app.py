"""Tests for the ar_segmentation app's LoRA setup.

This app predates the ``head_`` LoRA fix and keeps its own model and
apply_peft_lora(), so the shared tests in test_lora_setup.py do not cover it.
These pin the part that actually broke: the fine-tuning head must stay
trainable once PEFT wraps the model, and the released pre-fix checkpoint must
still load.
"""

import pytest
import torch

from conftest import IMG_SIZE, IN_CHANS, PATCH_SIZE, make_batch
from finetune import HEAD_PREFIX, apply_peft_lora, discover_head_modules
from infer import remap_legacy_head_keys
from segmentation_models import HelioSpectformer2D

LORA_CONFIG = {
    "r": 4,
    "lora_alpha": 8,
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    "lora_dropout": 0.0,
    "bias": "none",
}


def _config():
    return {
        "dtype": torch.float32,
        "use_latitude_in_learned_flow": False,
        "model": {
            "ft_unembedding_type": "linear",
            "ft_out_chans": 1,
            "lora_config": LORA_CONFIG,
        },
    }


def _model():
    return HelioSpectformer2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=32,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=2,
        n_spectral_blocks=1,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        checkpoint_layers=[],
        rpe=False,
        finetune=True,
        config=_config(),
    )


def test_head_is_discovered_by_the_head_prefix():
    model = _model()
    # The backbone's own children sit alongside the head in this subclass layout.
    assert [n for n, _ in model.named_children()] == ["embedding", "backbone", "head_unembed"]
    assert discover_head_modules(model) == ["head_unembed"]


def test_head_stays_trainable_under_lora_and_backbone_is_frozen():
    model = apply_peft_lora(_model(), _config())

    head = {n: p for n, p in model.named_parameters() if "head_unembed.modules_to_save" in n}
    assert head, "head was not wrapped by PEFT modules_to_save"
    assert all(p.requires_grad for p in head.values())

    frozen_backbone = [
        p.requires_grad
        for n, p in model.named_parameters()
        if "backbone" in n and "lora_" not in n
    ]
    assert frozen_backbone and not any(frozen_backbone)


def test_head_receives_gradients_and_is_updated_by_the_optimizer():
    model = apply_peft_lora(_model(), _config())
    out = model(make_batch())
    assert out.shape == (2, 1, IMG_SIZE, IMG_SIZE)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros_like(out)).backward()
    opt.step()

    updated = [
        n
        for n, p in model.named_parameters()
        if n in before and not torch.equal(before[n], p.detach())
    ]
    assert any("head_unembed" in n for n in updated)


def test_lora_refuses_a_model_whose_head_lacks_the_prefix():
    model = _model()
    model.unembed = model.head_unembed
    del model.head_unembed

    assert discover_head_modules(model) == []
    with pytest.raises(ValueError, match=HEAD_PREFIX):
        apply_peft_lora(model, _config())


def test_pre_fix_checkpoint_still_loads_strictly():
    trained = apply_peft_lora(_model(), _config())

    # A released pre-fix checkpoint: head stored as a plain frozen "unembed",
    # with no modules_to_save entries at all.
    legacy = {}
    for key, value in trained.state_dict().items():
        if "head_unembed.modules_to_save.default." in key:
            continue
        legacy[key.replace("head_unembed.original_module.", "unembed.")] = value
    assert not any("modules_to_save" in k for k in legacy)

    target = apply_peft_lora(_model(), _config())
    target.load_state_dict(remap_legacy_head_keys(legacy, target), strict=True)


def test_post_fix_checkpoint_round_trips_untouched():
    trained = apply_peft_lora(_model(), _config())
    target = apply_peft_lora(_model(), _config())
    state = trained.state_dict()
    target.load_state_dict(remap_legacy_head_keys(state, target), strict=True)
