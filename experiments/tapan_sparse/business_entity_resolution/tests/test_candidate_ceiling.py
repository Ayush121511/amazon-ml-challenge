import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_candidate_ceiling import audit


class CeilingTest(unittest.TestCase):
    def test_partial_recovery_singleton_and_zero_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {'entity_id': 'a', 'country': 'India', 'ranked': ['x', 'wrong'],
                 'channels': {'name_words': ['x', 'y'], 'address_words': ['x', 'wrong']}},
                {'entity_id': 'b', 'country': 'India', 'ranked': ['wrong'],
                 'channels': {'name_words': ['wrong']}},
                {'entity_id': 'c', 'country': 'US', 'ranked': [], 'channels': {}},
            ]
            candidates = root / 'candidates.jsonl'
            candidates.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            gold = root / 'gold.tsv'
            gold.write_text('source1_entity_id\tmatched_entity_ids\na\tx,y\nb\t\nc\tz\n', encoding='utf-8')
            result = audit(candidates, gold)['groups']
            self.assertAlmostEqual(result['all']['top50_oracle_macro_f05'], (5/6 + 1)/3)
            self.assertAlmostEqual(result['all']['route_union_oracle_macro_f05'], 2/3)
            self.assertEqual(result['all']['route_union_recovered_links'], 2)
            self.assertEqual(result['all']['route_union_positive_refs_with_zero_recovery'], 1)
            self.assertEqual(result['country:India']['route_union_oracle_macro_f05'], 1)
            gold.write_text('source1_entity_id\tmatched_entity_ids\na\tx,y\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing labels'):
                audit(candidates, gold)


if __name__ == '__main__':
    unittest.main()
