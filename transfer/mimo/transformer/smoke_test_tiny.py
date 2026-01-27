import torch

from configuration_mimo_v2_flash import MiMoV2FlashConfig
from modeling_mimo_v2_flash import MiMoV2FlashForCausalLM


def run_smoke_test() -> None:
    torch.manual_seed(0)

    config = MiMoV2FlashConfig.tiny_config()
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
