import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from diagnose_retrieval import read_saved


class DiagnosticTests(unittest.TestCase):
    def test_partial_tail_preserved_and_order_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidates.jsonl'
            original = (json.dumps({'entity_id': 'a'})+'\n'+ '{"entity_id":').encode()
            path.write_bytes(original)
            self.assertEqual(len(read_saved(path, [{'entity_id': 'a'}])), 1)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                read_saved(path, [{'entity_id': 'b'}])
