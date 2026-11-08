import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posting_index import build, load_manifest
from reverse_index import Reverse, query
from test_posting_index import FILLER, write_source


class ReverseIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.root = Path(cls.tmp.name)
        write_source(t / 'x_source1.tsv.gz', [
            ('S1-1', 'Payne Enterprises', '3315 Fremont St, Peoria, IL', 'US'),
            ('S1-2', 'Payne Enterprises', '12 Oak Lane, Dayton, OH', 'US'),
            ('S1-3', 'Boulangerie Martin', '14 Rue de Rivoli, Paris', 'France')])
        write_source(t / 'x_source2.tsv.gz', [
            ('S2-a', 'Payne Enterpires', '3315 Fremont Street, Peoria', 'US'),
            ('S2-fr', 'Boulangerie Martin', '14 Rue de Rivoli', 'France'),
            ('S2-jp', 'Kyoto Tea', '3 Gion', 'Japan')])
        write_source(t / 'x_source3.tsv.gz', [(f'S3-f{i}', n, a, 'US') for i, (n, a) in enumerate(FILLER)])
        build([t / 'x_source1.tsv.gz'], t / 'refidx', workers=1, chunk_rows=2, max_df_fraction=1.0)
        build([t / 'x_source2.tsv.gz', t / 'x_source3.tsv.gz'], t / 'tidx', workers=1, chunk_rows=2,
              max_df_fraction=1.0)
        query(t / 'refidx', t / 'tidx', t / 'rev', top_k=5, top_terms=64, keep=2, batch=3, threads=1)
        cls.manifest = load_manifest(t / 'tidx')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def reverse(self, country):
        return Reverse(self.root / 'rev', self.root / 'refidx',
                       self.manifest['countries'][country]['dir'], country)

    def test_target_finds_its_reference_first(self):
        out = self.reverse('US').lookup([{'entity_id': 'S1-1'}, {'entity_id': 'S1-2'}, {'entity_id': 'S1-9'}])
        ids = np.load(self.root / 'tidx' / self.manifest['countries']['US']['dir'] / 'ids.npy')
        found = {(int(i), ids[o].decode(), int(k)) for i, o, k in zip(out['ref'], out['t_ord'], out['rank'])}
        self.assertIn((0, 'S2-a', 1), found)  # S2-a's best reference is S1-1 (same address)
        self.assertNotIn(2, {f[0] for f in found})  # Unknown reference: no reverse candidates
        self.assertLessEqual(len(out['ref']), 2 * 8)  # keep=2 references per target

    def test_unseen_country_is_empty(self):
        out = self.reverse('Japan').lookup([{'entity_id': 'S1-1'}])
        self.assertEqual(len(out['ref']), 0)
        fr = self.reverse('France').lookup([{'entity_id': 'S1-3'}])
        self.assertEqual(len(fr['ref']), 1)


if __name__ == '__main__':
    unittest.main()
