import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_sparse_matcher import split_references
from preprocess import transform


class SparseMatcherTest(unittest.TestCase):
    def test_true_entity_groups_stay_together(self):
        truth = {f'S1-{i}': {f'S2-{i}'} for i in range(100)}
        truth['S1-other'] = truth['S1-4']
        parts = split_references(truth)
        self.assertEqual(set.union(*parts.values()), set(truth))
        for members in parts.values():
            self.assertEqual('S1-4' in members, 'S1-other' in members)
        self.assertFalse(parts['train'] & parts['dev'])
        self.assertFalse(parts['train'] & parts['holdout'])

    def test_real_training_with_purge_and_unretrieved_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truths = []
            total = 0
            with gzip.open(root / 'pairs.jsonl.gz', 'wt', encoding='utf-8') as stream:
                for i in range(2000):
                    sid = f'S1-{i}'
                    ref = transform({'entity_id': sid, 'business_name': f'Blue shop {i}',
                                     'business_address': f'{i} Market Road', 'country': 'India' if i % 2 else 'US'})
                    gold = [] if i % 10 == 0 else [f'S2-{i}', f'S3-{i}']
                    truths.append({'source1_entity_id': sid, 'country': ref['country'],
                                   'gold_ids': gold + ([f'S2-missing-{i}'] if i % 7 == 1 else [])})
                    for rank, tid in enumerate([f'S2-{i}', f'S3-{i}', 'S2-common-negative', f'S2-negative-{i}'], 1):
                        target = dict(ref, entity_id=tid)
                        label = int(tid in gold)
                        if not label:
                            target = transform(dict(target, business_name='Unrelated bakery', business_address='999 Other Lane'))
                        pair = {'source1_entity_id': sid, 'target_entity_id': tid,
                                'reference': ref, 'target': target, 'label': label,
                                'retrieval_ranks': {'name_words': rank, 'address_words': rank, 'name_trigrams': None}}
                        stream.write(json.dumps(pair) + '\n')
                        total += 1
            (root / 'reference_truth.json').write_text(json.dumps(truths), encoding='utf-8')
            (root / 'summary.json').write_text(json.dumps({'pairs': total}), encoding='utf-8')
            script = Path(__file__).resolve().parents[1] / 'src/train_sparse_matcher.py'
            subprocess.run([sys.executable, str(script), '--data', str(root), '--output', str(root / 'out')],
                           check=True, capture_output=True, timeout=90)
            report = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(report['training_pairs_after_purge'], report['pairs']['train'] - report['references']['train'])
            self.assertEqual(report['dev_holdout_shared_candidate_targets'], 1)
            self.assertLess(report['macro_f05_holdout'], 1)
            self.assertGreater(report['macro_f05_holdout'], .8)
            self.assertTrue((root / 'out/model.txt').is_file())
            subprocess.run([sys.executable, str(script), '--data', str(root), '--output', str(root / 'v2'),
                            '--feature-version', 'v2', '--split-from', str(root / 'out/split.json'), '--dev-only'],
                           check=True, capture_output=True, timeout=90)
            enhanced = json.loads((root / 'v2/report.json').read_text())
            self.assertEqual(enhanced['feature_count'], 57)
            self.assertIsNone(enhanced['macro_f05_holdout'])
            self.assertFalse((root / 'v2/holdout_predictions.jsonl').exists())
            self.assertEqual(json.loads((root / 'v2/split.json').read_text()),
                             json.loads((root / 'out/split.json').read_text()))


if __name__ == '__main__':
    unittest.main()
