import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PairTests(unittest.TestCase):
    def test_labels_and_missing_gold_not_injected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'references.json').write_text(json.dumps([{'entity_id': 'q', 'country': 'US'}]))
            result = {'entity_id': 'q', 'country': 'US', 'ranked': ['a'],
                      'channels': {'name_words': ['a', 'b'], 'address_words': [], 'name_trigrams': []}}
            (root/'candidates.jsonl').write_text(json.dumps(result)+'\n')
            (root/'gold.tsv').write_text('source1_entity_id\tmatched_entity_ids\nq\ta,c\n')
            for source, ids in [(2, ['a', 'b']), (3, ['c'])]:
                with gzip.open(root/f'train_source{source}.tsv.gz', 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['entity_id', 'country'])
                    writer.writerows([[identifier, 'US'] for identifier in ids])
            script = Path(__file__).resolve().parents[1]/'src/prepare_saved_pairs.py'
            subprocess.run([sys.executable, str(script), '--run', str(root), '--targets', str(root),
                            '--ground-truth', str(root/'gold.tsv'), '--output', str(root/'out')],
                           check=True, capture_output=True)
            with gzip.open(root/'out/pairs.jsonl.gz', 'rt') as f:
                pairs = [json.loads(line) for line in f]
            self.assertEqual({r['target_entity_id']: r['label'] for r in pairs}, {'a': 1, 'b': 0})
            self.assertEqual(json.loads((root/'out/summary.json').read_text())['missed_gold_links'], 1)
