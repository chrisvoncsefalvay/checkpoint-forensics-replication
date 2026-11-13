# Checkpoint forensics

Training changes a model's weights, but it need not erase the pattern left by
initialisation. PyTorch distributes random-number streams across GPU threads;
changing the thread count changes where those numbers land. A checkpoint can
retain enough of that arrangement to recover the initialisation layout.

This repository reproduces initialisation-layout comparisons from
[Will DePue's checkpoint-forensics report](https://unemploymentlabs.com/checkpoint-forensics)
using weight slices downloaded from Hugging Face.

| Checkpoint | Best layout | Held-out correlation | Wrong-seed correlation |
|---|---:|---:|---:|
| DeepSeek-V3 | 132 SMs | 0.9999986286 | 0.0058546500 |
| DeepSeek-V4-Flash | 148 SMs | 0.9997838853 | 0.0040199187 |
| Kimi K3 | 152 SMs | 0.9999961306 | 0.0017341414 |

The results recover the reported Hopper-compatible and Blackwell-compatible
layouts. Kimi K3 required a wider row search and PyTorch’s large-tensor split
behaviour. The results, including the earlier unsuccessful searches, are in
[results/REPORT.md](results/REPORT.md).

## Run it

Python 3.10+ and NumPy are sufficient:

```sh
python -m pip install -e .
checkpoint-forensics analyse fixtures/deepseek-v3-tail.npz \
  --seeds 953049583 --sms 108,132,148,152 --offsets 0 \
  --columns 7168 --out results/deepseek-v3.json
checkpoint-forensics analyse fixtures/deepseek-v4-flash-tail.npz \
  --seeds 953049583 --sms 108,132,148,152 --offsets 0 \
  --columns 4096 --out results/deepseek-v4-flash.json
```

The fixtures contain published checkpoint weights, not generated examples. To
fetch the DeepSeek-V3 slice again:

```sh
checkpoint-forensics fetch \
  --repo deepseek-ai/DeepSeek-V3-Base \
  --revision afb92e1fa402c2be2a9eb085312bb02e0384d6c7 \
  --tensor model.embed_tokens.weight \
  --rows 129272:129280 \
  --out fixtures/deepseek-v3-tail.npz
```

The downloader reads safetensors headers and selected rows through HTTP Range
requests. It records the model revision, tensor, row indices and SHA-256 hashes
beside each fixture. `--rows` accepts a half-open range or comma-separated indices.

## Scan beyond the fixtures

The `scan` command streams selected tensors or a whole checkpoint, with byte,
row, tensor and time limits and resumable progress. It keeps the original tensor
indices while reading batches and handles PyTorch’s contiguous TensorIterator
splits above the 32-bit byte-offset limit. See [the streaming guide](docs/streaming.md)
for commands and per-tensor initialisation settings.

## How the comparison works

The analyser generates Philox4x32-10 streams for each candidate seed, thread
layout and RNG offset. It selects the strongest correlation on discovery rows,
then tests that candidate on held-out rows. Wrong-seed and alternative-layout
controls use the same checkpoint data.

For PyTorch's contiguous CUDA normal kernel, with a block size of 256:

```
T = 256 * min(ceil(numel / 256), SMs * floor(max_threads_per_SM / 256))
thread = index % T
call = floor(index / (4*T))
lane = floor(index / T) % 4
```

Each thread starts `curand_init(seed, thread, offset)` and writes successive
`normal4` values at positions separated by `T`. The default is 2,048 threads per
SM, configurable in the Python API. The implementation follows
[PyTorch's execution policy](https://github.com/pytorch/pytorch/blob/1be6f926462315354c0b26b4c2d0924a785efbc8/aten/src/ATen/native/cuda/DistributionTemplates.h).

These experiments use the seeds published in the report. To search RNG offsets,
pass a range such as `--offsets 0:256:4`. Prior initialisation calls, tensor
parallelism and sharded construction can alter the offset or layout, which is
why an unsuccessful offset-zero comparison alone says little about a model.

## GPU comparison

The CPU implementation uses NumPy for the normal transform. This is sufficient
for the correlations above; bitwise comparison requires the GPU implementation.

With PyTorch, a CUDA toolkit and an NVIDIA GPU installed:

```sh
python -m checkpoint_forensics.cuda_verify fixtures/deepseek-v3-tail.npz \
  --seed 953049583 --sms 132 --offset 0 --std .006 \
  --out results/deepseek-v3-curand.json
```

This sparse cuRAND kernel can emulate a chosen thread layout on the available
GPU. To compare against unmodified PyTorch on the physical target GPU:

```sh
python -m checkpoint_forensics.torch_verify fixtures/deepseek-v3-tail.npz \
  --seed 953049583 --offset 0 --std .006 --max-gib 4 \
  --out results/deepseek-v3-torch.json
```

The latter allocates the original BF16 tensor and records the device, SM count,
CUDA and PyTorch versions. A smaller tensor could change the launch layout.
Run it on H100 and B300, then repeat with seed 953049584 as the negative control.
The supplied GPU scripts have not yet been run. They assume BF16, zero mean and
the specified standard deviation.

A layout fingerprint concerns initialisation. Distinguishing NVIDIA from a
specially configured AMD device requires the finer numerical comparison in the
original report. The layout alone does not identify the hardware used for the
rest of training, its location or how it was acquired.

## Tests and data

```sh
python -m unittest discover -s tests -v
```

Eight tests cover the checkpoint hashes, discovery/holdout split and the
DeepSeek and Kimi results with their negative controls. All use the downloaded weights.
Model slices retain their upstream licences; the fixture manifests identify
their source repositories and revisions.

The original report's scripts and selected-row fixtures are available from its
author on request. This is a separate implementation. See [SOURCES.md](SOURCES.md)
for the method and software references.
