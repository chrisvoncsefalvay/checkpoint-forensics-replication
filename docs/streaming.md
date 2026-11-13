# Scanning a checkpoint

`scan` walks the tensors in a Hugging Face checkpoint without downloading the
whole model. It reads batches of rows, generates the requested initialisation
candidates at their original positions and accumulates correlations. The state
file keeps the next row, candidate statistics and strongest row matches.

## Start with a limit

```sh
checkpoint-forensics scan \
  --repo deepseek-ai/DeepSeek-V3-Base \
  --tensor '^model\.embed_tokens\.weight$' \
  --seeds 953049583,953049584 --sms 108,132,148,152 \
  --max-weight-bytes 268435456 \
  --state scans/deepseek-v3.json
```

This reads at most 256 MiB of weight payload. It compares 512 evenly spaced
columns per row by default. Set `--columns 0` to score every value read. The
wrong-seed candidate in this example is a useful control.

Run the same command with a larger limit to continue. Byte, row and tensor
limits are cumulative, so raising the byte limit to 536870912 permits another
256 MiB. A completed scan does not repeat its downloads. The saved revision
remains fixed even if the repository's `main` branch changes.

## Iterate over everything

Omit `--tensor` to visit every tensor. Remove the weight and time limits
explicitly when a complete scan is intended:

```sh
checkpoint-forensics scan \
  --repo deepseek-ai/DeepSeek-V3-Base \
  --seeds 953049583 --sms 108,132,148,152 \
  --columns 0 --max-weight-bytes 0 --max-seconds 0 \
  --state scans/deepseek-v3-full.json
```

This can transfer a very large amount of data and spend substantial CPU time.
It is an available mode, not the default. Only BF16, F16 and F32 tensors are
scored. Packed/quantised tensors and scalars are recorded as skipped; a full
checkpoint walk therefore need not score every stored parameter.

## Limits and memory

| Option | Default | Meaning |
|---|---:|---|
| `--max-weight-bytes` | 268435456 | Cumulative downloaded weight bytes; 0 removes the limit |
| `--max-rows` | 0 | Cumulative rows scanned; 0 removes the limit |
| `--max-tensors` | 0 | Cumulative tensors visited, including skipped tensors |
| `--max-seconds` | 600 | Scanning time for this invocation; 0 removes the limit |
| `--batch-rows` | 32 | Maximum rows per request |
| `--batch-bytes` | 8388608 | Maximum weight payload per batch |
| `--columns` | 512 | Columns scored per row; 0 scores all columns |
| `--keep-hits` | 100 | Strongest qualifying rows retained per candidate and tensor |
| `--threshold` | 0.99 | Minimum row correlation to retain as a hit |

Rows are indivisible download units. If the remaining byte allowance cannot
cover a row, scanning pauses before it. A row larger than `--batch-bytes` also
pauses the scan until that limit is raised. Decoded arrays and candidates use
additional memory beyond the payload limit.

Time is checked between requests and batches; an in-flight request or scoring
batch can finish after the limit. Model/index metadata is fetched during setup,
and safetensors header bytes are counted separately from weight bytes. The
scanner needs Linux or macOS for its process lock.

One-dimensional tensors are treated as a single row. Higher-rank tensors use
the first dimension as rows and flatten the remaining dimensions. Original
linear positions are preserved.

## State and interruption

The state is replaced atomically after each completed batch. A process lock
prevents two writers from using it simultaneously. Downloads are counted before
scoring, so a crash between download and scoring may require reading that batch
again and increasing the byte allowance. Completed rows are not scored twice.

Changing seeds, layouts, offsets, tensor selection, retained-hit count or scoring
columns requires a new state file. Limits and batch sizes can change when resuming. The state
contains all candidate aggregate statistics, total hit counts and the best
`--keep-hits` rows, rather than an unbounded list of every match.

## State the initialisation hypothesis

The default hypothesis is a contiguous PyTorch normal initialisation of each
tensor, using its stored dtype and offset zero. The RNG offset is **not** inferred
from tensor order in the checkpoint. Different tensors may have consumed very
different generator states during construction.

Use `--tensor-config` to override the hypothesis for named tensors:

```json
{
  "model.embed_tokens.weight": {
    "seeds": [953049583],
    "sms": [132, 148],
    "offsets": [0, 4],
    "init_dtype": "BF16",
    "mapping": "pytorch-contiguous"
  }
}
```

The default mapping implements the contiguous TensorIterator split used when
32-bit byte offsets are insufficient. For Kimi K3's [163840, 7168] BF16 embedding,
a 152-SM hypothesis reserves the outer call's 3776 RNG positions, then initialises
the first half at offset 3776 and the second at 5664. This follows the referenced
PyTorch implementation; another framework or software version may differ.

`--mapping single-launch` retains the original emulator's unsplit mapping.
`--init-dtype` describes the dtype at initialisation, which may differ from the
checkpoint's storage dtype. For a known contiguous slice of a larger initialised
tensor, per-tensor `init_numel` and `index_shift` specify that parent tensor's size
and the slice's starting element. Arbitrary reshuffling and tensor-parallel
column layouts need their own mapping.

A high-scoring row found by this scan is a discovery result. Confirm it on fresh
rows or coordinates after fixing the candidate. Weak or absent matches in trained
weights do not establish a hardware class, and changing row batches is not an
independent replication.
