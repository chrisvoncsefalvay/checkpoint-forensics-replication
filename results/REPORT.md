# Checkpoint results

12–13 November 2026

## DeepSeek-V3

Downloaded 8 real embedding rows (129272–129279), 57,344 BF16 values, from
`deepseek-ai/DeepSeek-V3-Base` at pinned revision
`afb92e1fa402c2be2a9eb085312bb02e0384d6c7`.

Using the report's seed 953049583, RNG offset 0 and the full tensor layout
[129280, 7168], four candidate layouts were compared on even-numbered rows.
132 SMs won. The candidate was then evaluated on odd-numbered rows.

| Held-out candidate | Correlation (28,672 values) |
|---|---:|
| 132 SMs, seed 953049583 | 0.9999986286 |
| 132 SMs, seed 953049584 | 0.0058546500 |
| 108 SMs, seed 953049583 | 0.0083042647 |
| 148 SMs, seed 953049583 | 0.0010670692 |
| 152 SMs, seed 953049583 | -0.0022028851 |

Every held-out row individually correlates above 0.9999986. The scale fitted on
discovery rows is approximately 0.00600016. The result reproduces the reported Hopper-compatible layout. The tail rows
were selected independently of the original study. Calculations used the NumPy
CPU transform described in the README.

Command:

```sh
python -m checkpoint_forensics.cli analyse fixtures/deepseek-v3-tail.npz \
  --seeds 953049583 --sms 108,132,148,152 --offsets 0 \
  --columns 7168 --out results/deepseek-v3.json
```

## DeepSeek-V4-Flash

Downloaded 16 real embedding rows (129264–129279), 65,536 BF16 values, from
`deepseek-ai/DeepSeek-V4-Flash` at revision `60d8d70770c6776ff598c94bb586a859a38244f1`.
The full tensor shape is [129280, 4096]. With seed 953049583 and offset zero,
148 SMs wins on the eight discovery rows.

| Held-out candidate | Correlation (32,768 values) |
|---|---:|
| 148 SMs, seed 953049583 | 0.9997838853 |
| 148 SMs, seed 953049584 | 0.0040199187 |
| 108 SMs, seed 953049583 | 0.0047023975 |
| 132 SMs, seed 953049583 | 0.0035816603 |
| 152 SMs, seed 953049583 | -0.0009568257 |

Five of the eight held-out rows individually correlate above 0.9999986. The other
three range from 0.99898 to 0.99966, consistent with small changes. This reproduces
the central V3 132-SM versus V4-Flash 148-SM layout contrast using real weights.
The distinction between layout recovery and hardware attribution is discussed
in the README.

```sh
python -m checkpoint_forensics.cli analyse fixtures/deepseek-v4-flash-tail.npz \
  --seeds 953049583 --sms 108,132,148,152 --offsets 0 \
  --columns 4096 --out results/deepseek-v4-flash.json
```

## Kimi K3

The whole-embedding scan recovers a 152-SM pattern. Fourteen rows selected using
128 coordinates each reproduce on the remaining 7,040 coordinates per row:
**r = 0.9999961306 across 98,560 held-out values**.

| Held-out candidate | Correlation |
|---|---:|
| Seed 20256586, 152 SMs, offset 0 | 0.9999961306 |
| Seed 20256587 | 0.0017341414 |
| 108 SMs | −0.0034190414 |
| 132 SMs | 0.0047935794 |
| 148 SMs | 0.0055033749 |
| Initial offset +4 | 0.0007715995 |

The selected rows are 124, 125, 177–187 and 1699 in
`language_model.model.embed_tokens.weight`, shape [163840, 7168], at revision
`f831ab66814297da540d832a5235f8e904f29d06`. The seed is taken from the original
report. The candidate and selected rows were fixed before scoring the other
7,040 coordinates. Row-level results and the discovery column indices are in
[kimi-k3-confirmation.json](kimi-k3-confirmation.json).

The earlier tests missed this for two reasons. Their fixtures covered the last
256 rows, whereas the confirmed matches are near the beginning of the vocabulary.
The first emulator also assumed a single launch, but Kimi's BF16 embedding
exceeds TensorIterator's 32-bit byte-offset limit. PyTorch reserves the outer
call's RNG state and initialises two halves: for 152 SMs, their offsets are
3776 and 5664. The successful rows are in the first half. On the first five
confirmed rows, the old mapping gives r = −0.00974, compared with 0.99999617
under the corrected mapping.

The surviving pattern's fitted scale is approximately 2.46×10⁻⁶. This is evidence
of a retained initialisation pattern at a small scale, rather than numerically
untouched weights. As with DeepSeek, the CPU calculation establishes the layout
match; the physical-GPU bitwise and NVIDIA-versus-AMD comparisons remain separate.

Reproduce the confirmation from the published checkpoint slice:

```sh
python scripts/confirm_kimi.py
```

The original bounded-search outputs remain in the repository. Their negative
result applies to those rows and mappings, not to the corrected whole-embedding
search. See [kimi-k3-scan.json](kimi-k3-scan.json) for the scan configuration,
coverage and candidate statistics. The completed scan covered all 163,840
embedding rows at 128 columns each (20,971,520 values per candidate), reading
2,348,810,240 weight bytes. Sixteen rows exceeded discovery r = 0.99 under
the 152-SM layout. Fourteen have the held-out confirmation above; rows 57118
and 57119 were also discovery matches but were not independently confirmed.
The 108- and 132-SM layouts each produced one threshold hit above 0.4, and
148 SMs produced none. A threshold hit alone is not a confirmed fingerprint.

## Reproduction

- `python -m compileall -q checkpoint_forensics`: passed.
- `python -m unittest discover -s tests -v`: eight tests passed using actual
  downloaded checkpoints only (fixture provenance/integrity and split/control
  construction and the real DeepSeek/Kimi fingerprint regressions).
- CUDA scripts supplied but not run in this CPU environment.
