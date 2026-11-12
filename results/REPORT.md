# Checkpoint results

12 November 2026

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

Downloaded the final 16 real embedding rows at revision
`f831ab66814297da540d832a5235f8e904f29d06`.
Using seed 20256586 and offset zero, all candidate correlations were near zero.
A bounded exploratory search over 1,024 four-aligned offsets and four layouts
also failed to generalise (best discovery r 0.1247, holdout r -0.0291, using 128
columns per row). The selected rows may differ from the report’s 13 preserved rows; their
construction order, sharding and offset remain unknown. The offset search reused
the first fixture, so its holdout result is exploratory.

## Reproduction

- `python -m compileall -q checkpoint_forensics`: passed.
- `python -m unittest discover -s tests -v`: four tests passed using actual
  downloaded checkpoints only (fixture provenance/integrity and split/control
  construction and the real V3/V4 fingerprint regressions).
- CUDA scripts supplied but not run in this CPU environment.

An exploratory offset-zero check of a further 240 Kimi special-token rows also found no high-correlation row. The corresponding fixture and results are included.
