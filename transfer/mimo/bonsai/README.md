# MiMo V2 Flash in JAX

This directory contains a pure JAX implementation of the MiMo V2 Flash language model using the Flax NNX API. The code mirrors the PyTorch architecture in `transfer/mimo/transformer` and includes a safetensors weight conversion utility.

## What is implemented

- Hybrid attention with full and sliding-window layers (based on `hybrid_layer_pattern`)
- Partial RoPE (controlled by `partial_rotary_factor`)
- Optional MoE blocks with routing and expert MLPs
- Weight conversion from PyTorch safetensors to JAX parameters

## Running

Hook the model up with your own runner or tests, similar to `transfer/qwen3/bonsai`.
