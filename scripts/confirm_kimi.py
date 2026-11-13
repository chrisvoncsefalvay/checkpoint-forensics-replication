"""Confirm the Kimi candidate on coordinates excluded from the discovery scan."""
import json
from pathlib import Path
import numpy as np
from checkpoint_forensics.analysis import corr
from checkpoint_forensics.streaming import candidate_values, initialise_segments

ROOT = Path(__file__).resolve().parents[1]
fixture = np.load(ROOT / 'fixtures/kimi-k3-confirmed.npz', allow_pickle=False)
values, rows, shape = fixture['values'], fixture['rows'], fixture['shape']
discovery = np.unique(np.linspace(0, shape[1] - 1, 128, dtype=int))
heldout = np.setdiff1d(np.arange(shape[1]), discovery)
indices = (rows[:, None] * shape[1] + np.arange(shape[1])).ravel()
config = {'init_dtype': 'BF16', 'mapping': 'pytorch-contiguous'}
results = []
for seed, sms, offset, label in [
    (20256586, 152, 0, 'selected'),
    (20256587, 152, 0, 'wrong_seed'),
    (20256586, 108, 0, '108_SM'),
    (20256586, 132, 0, '132_SM'),
    (20256586, 148, 0, '148_SM'),
    (20256586, 152, 4, 'offset_plus_4'),
]:
    predicted = candidate_values(indices, shape, seed, sms, offset, config).reshape(values.shape)
    results.append(dict(label=label, seed=seed, sms=sms, offset=offset,
        heldout_r=corr(values[:, heldout].ravel(), predicted[:, heldout].ravel()),
        row_correlations={str(int(row)): corr(x[heldout], y[heldout])
                          for row, x, y in zip(rows, values, predicted)}))
predicted = candidate_values(indices, shape, 20256586, 152, 0, config).reshape(values.shape)
scale = float(np.sum(values[:, discovery].astype(float) * predicted[:, discovery]) /
              np.sum(predicted[:, discovery].astype(float) ** 2))
result = dict(repo='moonshotai/Kimi-K3', revision='f831ab66814297da540d832a5235f8e904f29d06',
    tensor='language_model.model.embed_tokens.weight', selected_rows=rows.tolist(),
    discovery_columns=discovery.tolist(), heldout_columns_per_row=len(heldout),
    heldout_values=len(rows) * len(heldout), discovery_fitted_scale=scale,
    initialisation_segments=initialise_segments(int(np.prod(shape)), 152, 0, itemsize=2),
    method='Rows selected on 128 discovery columns; fixed candidate tested on the remaining 7040 columns',
    results=results)
output = ROOT / 'results/kimi-k3-confirmation.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
