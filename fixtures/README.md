# Checkpoint slices

These files are extracts of the upstream model weights, converted from BF16
safetensors to NumPy float32 arrays. The conversion preserves every selected
weight value. Each JSON manifest records the source revision, row indices and
hashes of the original row bytes.

- `deepseek-v3-tail.npz`: Copyright (c) 2023 DeepSeek. Distributed under the
  [DeepSeek model licence](licenses/DeepSeek-V3-LICENSE-MODEL), including the
  use-based restrictions in paragraph 5 and Attachment A.
- `deepseek-v4-flash-tail.npz`: Copyright (c) 2023 DeepSeek. Distributed under
  the [upstream MIT licence](licenses/DeepSeek-V4-Flash-LICENSE).
- `kimi-k3-tail.npz` and `kimi-k3-special.npz`: Copyright (c) 2026 Moonshot AI.
  Distributed under the [Kimi K3 licence](licenses/Kimi-K3-LICENSE).

The upstream licences govern use and redistribution of these model slices.
