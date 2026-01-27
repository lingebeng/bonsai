import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Import as a package so relative imports inside the model code resolve.
from transformer.configuration_mimo_v2_flash import MiMoV2FlashConfig
import transformer.modeling_mimo_v2_flash as mimo_flash
from transformer.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def run_smoke_test() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="`rope_config_validation` is deprecated.*",
    )
    torch.manual_seed(0)

    config = MiMoV2FlashConfig.tiny_config()
    # Ensure runtime-required config fields are set for masking + attention.
    config.sliding_window = getattr(config, "sliding_window", 8) or 8
    config._attn_implementation = "eager"

    # Add default RoPE initializer for older Transformers builds.
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) or 1.0
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                config.rope_theta ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
                )
            )
            attention_scaling = 1.0
            return inv_freq, attention_scaling

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    if not hasattr(mimo_flash.MiMoV2FlashRotaryEmbedding, "compute_default_rope_parameters"):
        mimo_flash.MiMoV2FlashRotaryEmbedding.compute_default_rope_parameters = staticmethod(
            ROPE_INIT_FUNCTIONS["default"]
        )

    # Transformers eager attention path forwards extra kwargs; wrap to ignore them.
    if not getattr(mimo_flash.eager_attention_forward, "_accepts_kwargs", False):
        _orig_eager = mimo_flash.eager_attention_forward

        def _eager_wrapper(*args, **kwargs):
            kwargs.pop("position_ids", None)
            return _orig_eager(*args, **kwargs)

        _eager_wrapper._accepts_kwargs = True
        mimo_flash.eager_attention_forward = _eager_wrapper
    model = MiMoV2FlashForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits
    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"unexpected logits shape: {logits.shape}"
    )

    print("smoke test passed: logits shape =", tuple(logits.shape))


if __name__ == "__main__":
    run_smoke_test()
