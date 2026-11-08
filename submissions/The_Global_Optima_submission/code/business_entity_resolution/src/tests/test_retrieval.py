import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import RAW, EXTRA, transform
from retrieval_experiment import build_index, retrieve, evaluate, sample_references, load_checkpoint


class RetrievalTests(unittest.TestCase):
    def test_checkpoint_recovery_and_wrong_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidates.jsonl'
            result = {'entity_id': 'a', 'country': 'India', 'ranked': [], 'non_ascii_name': False,
                      'channels': dict.fromkeys(['name_words', 'address_words', 'name_trigrams'], [])}
            line = (json.dumps(result)+'\n').encode()
            path.write_bytes(line + b'{"entity_id":')
            self.assertEqual(load_checkpoint(path, [result]), [result])
            self.assertEqual(path.read_bytes(), line)
            self.assertTrue(path.with_suffix('.interrupted_tail').exists())
            with self.assertRaises(ValueError):
                load_checkpoint(path, [{'entity_id': 'b', 'country': 'India'}])

    def test_index_retrieval_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [transform(dict(zip(RAW, values))) for values in [
                ['S2-a', 'Unique Café', '12 Garden Road', 'India'],
                ['S2-b', 'Unique Cafe', '12 Garden Road', 'US'],
                ['S2-c', 'Different', '88 Other Street', 'India']]]
            source = root/'source.tsv.gz'
            with gzip.open(source, 'wt', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=RAW+EXTRA, delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            db = build_index([source], root/'index.sqlite')
            query = transform(dict(zip(RAW, ['S1-q', 'Unique Cafe', '12 Garden Road', 'India'])))
            result = retrieve(db, query)
            self.assertEqual(result['ranked'][0], 'S2-a')
            self.assertNotIn('S2-b', result['ranked'])
            db.close()
            db = build_index([source], root/'index.sqlite')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM entities').fetchone()[0], 3)
            db.close()
            gold = root/'gold.tsv'
            gold.write_text('source1_entity_id\tmatched_entity_ids\nS1-q\tS2-a\n', encoding='utf-8')
            self.assertEqual(evaluate([result], gold)['all']['recall_at_10'], 1)
            self.assertEqual(sample_references(source, 1), sample_references(source, 1))
            self.assertEqual(len(sample_references(source, 1)), 2)

    def test_empty_fields_return_no_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'s.tsv.gz'
            with gzip.open(source, 'wt', encoding='utf-8') as f:
                f.write('\t'.join(RAW+EXTRA)+'\n')
            # Empty input should also be a valid completed index.
            db = build_index([source], root/'index.sqlite')
            row = transform(dict(zip(RAW, ['S1-q', '', '', 'US'])))
            self.assertEqual(retrieve(db, row)['ranked'], [])
            db.close()


if __name__ == '__main__':
    unittest.main()
