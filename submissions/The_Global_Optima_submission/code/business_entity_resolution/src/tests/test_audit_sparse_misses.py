import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_sparse_misses import audit


class SparseMissAuditTests(unittest.TestCase):
    def test_distinguishes_cutoff_loss_from_route_failure(self):
        references = [{'entity_id': 'r', 'country': 'India'}]
        candidates = {'r': {'ranked': ['found'], 'channels': {
            'name_words': ['found', 'trimmed'], 'address_words': [],
            'name_trigrams': []}}}
        gold = {'r': {'found', 'trimmed', 'absent'}}
        missed, counts = audit(references, candidates, gold)
        self.assertEqual(counts['found_top50'], 1)
        self.assertEqual(counts['lost_at_50'], 1)
        self.assertEqual(counts['absent_from_routes'], 1)
        self.assertEqual({row['target_id']: row['kind'] for row in missed},
                         {'trimmed': 'lost_at_50', 'absent': 'absent_from_routes'})


if __name__ == '__main__':
    unittest.main()
