# Sources

Retrieved 12 November 2026:

- Will DePue, *Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs*, https://unemploymentlabs.com/checkpoint-forensics
- PyTorch `aten/src/ATen/native/cuda/DistributionTemplates.h`, https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistributionTemplates.h
- NVIDIA cuRAND Philox4x32-10 device generator and Box-Muller normal transform, https://docs.nvidia.com/cuda/curand/device-api-overview.html

The report gives DeepSeek seed 953049583 and Moonshot seed 20256586. It reports
57,344 preserved DeepSeek-V3 values reproduced on H100 and 13 Kimi K3 rows
reproduced on GB200. The public article does not give the selected row IDs,
complete RNG offsets or a downloadable reproduction repository. Exact agreement
with its sample counts therefore requires additional source material or independent
row discovery, separated from held-out evaluation.

Each fixture’s JSON manifest records its checkpoint revision and raw-row hashes.

PyTorch main resolved to commit `1be6f926462315354c0b26b4c2d0924a785efbc8` when retrieved.
