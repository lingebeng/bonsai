import jax
import jax.numpy as jnp
from flax import nnx

from modeling import ModelConfig, MiMoV2Flash, forward


def run_smoke_test() -> None:
    cfg = ModelConfig.tiny_config()
    rngs = nnx.Rngs(0)
    model = MiMoV2Flash(cfg, rngs=rngs)

    batch_size = 2
    seq_len = 8
    pad_id = 0
    tokens = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, cfg.vocab_size)

    cache = model.init_cache(
        cfg,
        batch_size=batch_size,
        token_len=seq_len,
        generate_steps=0,
        dtype=jnp.float32,
    )
    logits, _ = forward(model, cache, tokens, pad_id)

    assert logits.shape == (batch_size, cfg.vocab_size), (
        f"unexpected logits shape: {logits.shape}"
    )
    print(logits)
    print("jax smoke test passed: logits shape =", tuple(logits.shape))


if __name__ == "__main__":
    run_smoke_test()
