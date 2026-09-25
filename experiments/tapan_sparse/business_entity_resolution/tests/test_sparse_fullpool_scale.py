import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from preprocess import RAW, EXTRA, transform


class SparseFullpoolScaleTests(unittest.TestCase):
    def test_additional_exclusion_and_report_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = {
                1: [['S1-old', 'Blue Harbor', '11 Market', 'US'],
                    ['S1-new', 'Blue Harbor', '11 Market', 'US'],
                    ['S1-india', 'Green Store', '22 Road', 'India']],
                2: [['S2-1', 'Blue Harbor', '11 Market', 'US'],
                    ['S2-2', 'Green Store', '22 Road', 'India']],
                3: [['S3-3', 'Blue Harbor', '11 Market', 'US'],
                    ['S3-4', 'Green Store', '22 Road', 'India']]}
            for source, rows in examples.items():
                path = root/f'train_source{source}.tsv.gz'
                with gzip.open(path, 'wt', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=RAW+EXTRA, delimiter='\t')
                    writer.writeheader()
                    writer.writerows(transform(dict(zip(RAW, values))) for values in rows)
                Path(str(path)+'.json').write_text(json.dumps({
                    'input': {'version': 2, 'limit': 0}, 'output_bytes': path.stat().st_size}))
            (root/'old.json').write_text(json.dumps([{'entity_id':'S1-old'}]))
            (root/'other.json').write_text(json.dumps([]))
            (root/'gold.tsv').write_text('source1_entity_id\tmatched_entity_ids\n'
                                         'S1-new\tS2-1,S3-3\nS1-india\tS2-2,S3-4\n')
            script = Path(__file__).resolve().parents[1]/'src/sparse_fullpool.py'
            subprocess.run([sys.executable, str(script), '--data', str(root),
                            '--exclude', str(root/'old.json'),
                            '--exclude-additional', str(root/'other.json'),
                            '--gold', str(root/'gold.tsv'), '--output', str(root/'out'),
                            '--per-country', '1', '--fit-rows', '2',
                            '--chunk-rows', '2', '--top-k', '2', '--threads', '1'],
                           check=True, capture_output=True)
            report = json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['all']['references'], 2)
            self.assertEqual(report['benchmark']['stats']['target_rows'], 4)
            self.assertEqual(report['benchmark']['parameters']['exclude_additional'],
                             [str(root/'other.json')])
            subprocess.run([sys.executable, str(script), '--data', str(root),
                            '--exclude', str(root/'old.json'),
                            '--exclude-additional', str(root/'other.json'),
                            '--gold', str(root/'gold.tsv'), '--output', str(root/'compact'),
                            '--per-country', '1', '--fit-rows', '2',
                            '--chunk-rows', '2', '--top-k', '2', '--threads', '1',
                            '--compact-topk', '--query-batch', '1'],
                           check=True, capture_output=True)
            compact = json.loads((root/'compact/report.json').read_text())
            self.assertEqual(compact['all']['recall_at_50'], report['all']['recall_at_50'])


if __name__ == '__main__':
    unittest.main()
