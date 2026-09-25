import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posting_index import build, CountryIndex, load_manifest
from submission_pipeline import candidates
from test_posting_index import ref, write_source

SRC = Path(__file__).resolve().parents[1] / 'src'


class CombinedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        # A chain: many branches with the same name; only one matches the address too.
        branches = [(f'S2-b{i}', 'State Bank of India', f'{100 + i} Main Road Sector {i} Pune', 'India')
                    for i in range(30)]
        branches.append(('S2-true', 'State Bank of India', '12 MG Road Camp Pune', 'India'))
        branches += [(f'S2-n{i}', f'Other Shop {i}', '12 MG Road Camp Pune', 'India') for i in range(30)]
        write_source(root / 'train_source2.tsv.gz', branches)
        write_source(root / 'train_source3.tsv.gz', [('S3-x', 'Kyoto Tea', '3 Gion', 'India')])
        write_source(root / 'train_source1.tsv.gz', [('S1-1', 'State Bank of India', '12 MG Road Camp Pune', 'India')])
        with open(root / 'gold.tsv', 'w', encoding='utf-8') as f:
            f.write('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-true\n')
        build([root / 'train_source2.tsv.gz', root / 'train_source3.tsv.gz'], root / 'index',
              workers=1, chunk_rows=20, max_df_fraction=1.0, min_max_df=1000)
        cls.index = CountryIndex(root / 'index', 'India', load_manifest(root / 'index'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_combined_route_ranks_the_right_branch_first(self):
        row = ref('State Bank of India', '12 MG Road Camp Pune', country='India')
        c = candidates(self.index, [row], top_k=5, top_terms=64, keep=5, threads=1, routes=('combined',))
        best = self.index.ids[c['ord'][c['fused_rank'] == 1][0]].decode()
        self.assertEqual(best, 'S2-true')

    def test_blocking_recall_script_runs(self):
        ids = self.root / 'ids.txt'
        ids.write_text('S1-1\n')
        out = self.root / 'recall'
        subprocess.run([sys.executable, str(SRC / 'blocking_recall.py'), '--data', str(self.root),
                        '--index', str(self.root / 'index'), '--gold', str(self.root / 'gold.tsv'),
                        '--reference-ids', str(ids), '--output', str(out), '--top-k', '5',
                        '--threads', '1'], check=True, capture_output=True, cwd=SRC)
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual(report['references'], 1)
        self.assertEqual(report['configs']['plus_combined']['all']['recall_at_50'], 1.0)


if __name__ == '__main__':
    unittest.main()
