"""Checks use only downloaded real checkpoints, never synthetic observations."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
from checkpoint_forensics.analysis import analyse
ROOT=Path(__file__).resolve().parents[1]

class RealCheckpointTests(unittest.TestCase):
    def test_fixture_integrity(self):
        paths=list((ROOT/'fixtures').glob('*.npz'))
        self.assertTrue(paths,'Download real fixtures before running these tests')
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(meta['fixture_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(meta['revision']),40)
            d=np.load(path,allow_pickle=False)
            self.assertEqual(d['values'].shape,(len(d['rows']),d['shape'][1]))
            self.assertTrue(np.isfinite(d['values']).all())
            self.assertEqual(d['rows'].tolist(),meta['rows'])
    def test_real_discovery_holdout_separation(self):
        paths=list((ROOT/'fixtures').glob('*.npz'));self.assertTrue(paths)
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            seed=20256586 if 'Kimi' in meta['repo'] else 953049583
            r=analyse(path,[seed],[132,148,152],[0],max_columns=32)
            self.assertFalse(set(r['discovery_rows'])&set(r['heldout_rows']))
            self.assertEqual(len(r['heldout_controls']),4)
            self.assertEqual(r['heldout_controls'][0]['seed'],r['selected']['seed'])
            self.assertEqual(r['heldout_controls'][1]['seed'],r['selected']['seed']+1)

    def test_deepseek_v3_real_checkpoint_fingerprint(self):
        path=ROOT/'fixtures/deepseek-v3-tail.npz'
        self.assertTrue(path.exists())
        r=analyse(path,[953049583],[108,132,148,152],[0],max_columns=7168)
        self.assertEqual(r['selected']['sms'],132)
        self.assertGreater(r['heldout_controls'][0]['heldout_r'],0.99999)
        for control in r['heldout_controls'][1:]:
            self.assertLess(abs(control['heldout_r']),0.03)

    def test_deepseek_v4_real_checkpoint_fingerprint(self):
        path=ROOT/'fixtures/deepseek-v4-flash-tail.npz'
        self.assertTrue(path.exists())
        r=analyse(path,[953049583],[108,132,148,152],[0],max_columns=4096)
        self.assertEqual(r['selected']['sms'],148)
        self.assertGreater(r['heldout_controls'][0]['heldout_r'],0.999)
        for control in r['heldout_controls'][1:]:
            self.assertLess(abs(control['heldout_r']),0.03)

if __name__=='__main__':unittest.main()
