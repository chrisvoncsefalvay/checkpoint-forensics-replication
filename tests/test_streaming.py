import unittest
from pathlib import Path
import numpy as np
from checkpoint_forensics.streaming import initialise_segments, score_rows, merge_statistics, correlation_from_statistics

ROOT = Path(__file__).resolve().parents[1]


class StreamingTests(unittest.TestCase):
    def test_real_deepseek_rows_match_when_batched(self):
        fixture = np.load(ROOT / 'fixtures/deepseek-v3-tail.npz')
        values, rows, shape = fixture['values'], fixture['rows'], fixture['shape']
        config = dict(seeds=[953049583], sms=[132, 148], offsets=[0], init_dtype='BF16', mapping='pytorch-contiguous')
        whole = score_rows(values, rows, shape, config, columns=7168, threshold=.99)
        parts = [score_rows(values[i:i+2], rows[i:i+2], shape, config, columns=7168, threshold=.99) for i in range(0, len(rows), 2)]
        for candidate in whole:
            combined = None
            for part in parts:
                combined = merge_statistics(combined, part[candidate]['statistics'])
            self.assertAlmostEqual(correlation_from_statistics(combined), whole[candidate]['r'], places=12)
        self.assertGreater(whole['953049583:132:0']['r'], .99999)
        self.assertLess(abs(whole['953049583:148:0']['r']), .03)

    def test_actual_kimi_shape_splits_at_byte_limit(self):
        fixture = np.load(ROOT / 'fixtures/kimi-k3-tail.npz')
        total = int(np.prod(fixture['shape']))
        segments = initialise_segments(total, 152, 0, itemsize=2)
        self.assertEqual(segments, [(0, total // 2, 3776), (total // 2, total // 2, 5664)])

    def test_actual_deepseek_shape_keeps_one_launch(self):
        fixture = np.load(ROOT / 'fixtures/deepseek-v3-tail.npz')
        total = int(np.prod(fixture['shape']))
        self.assertEqual(initialise_segments(total, 132, 0, itemsize=2), [(0, total, 0)])

    def test_kimi_confirmed_rows_on_heldout_coordinates(self):
        from checkpoint_forensics.streaming import candidate_values
        from checkpoint_forensics.analysis import corr
        fixture = np.load(ROOT / 'fixtures/kimi-k3-confirmed.npz')
        values, rows, shape = fixture['values'], fixture['rows'], fixture['shape']
        discovery = np.unique(np.linspace(0, shape[1] - 1, 128, dtype=int))
        heldout = np.setdiff1d(np.arange(shape[1]), discovery)
        indices = (rows[:, None] * shape[1] + heldout).ravel()
        observed = values[:, heldout].ravel()
        config = dict(init_dtype='BF16', mapping='pytorch-contiguous')
        matched = candidate_values(indices, shape, 20256586, 152, 0, config)
        self.assertGreater(corr(observed, matched), .99999)
        for seed, sms, offset in [(20256587, 152, 0), (20256586, 132, 0), (20256586, 148, 0), (20256586, 152, 4)]:
            predicted = candidate_values(indices, shape, seed, sms, offset, config)
            self.assertLess(abs(corr(observed, predicted)), .03)

if __name__ == '__main__':
    unittest.main()
