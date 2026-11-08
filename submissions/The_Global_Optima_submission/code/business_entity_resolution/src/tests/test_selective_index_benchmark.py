import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from selective_index_benchmark import query_plan, retrieve


class SelectiveIndexTests(unittest.TestCase):
    def test_conjunctive_routes_find_name_and_address_candidates(self):
        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE entities (id INTEGER PRIMARY KEY, entity_id TEXT, country TEXT)')
        db.execute('CREATE VIRTUAL TABLE words USING fts5(name,address)')
        data = [('name', 'blue infrastructure company', 'somewhere else', 'India'),
                ('address', 'unrelated business', 'door 01 thane maharashtra', 'India'),
                ('other', 'blue infrastructure company', 'door 01 thane', 'US')]
        for index, (identifier, name, address, country) in enumerate(data, 1):
            db.execute('INSERT INTO entities VALUES (?,?,?)', (index, identifier, country))
            db.execute('INSERT INTO words(rowid,name,address) VALUES (?,?,?)',
                       (index, name, address))
        ref = {'entity_id': 'r', 'country': 'India', 'name_norm': 'blue infrastructure',
               'name_folded': 'blue infrastructure private limited',
               'address_folded': 'door no 01 thane maharashtra', 'address_numbers': '01'}
        self.assertEqual(len(query_plan(ref)), 2)
        result, timings = retrieve(db, ref, timeout_seconds=5)
        self.assertEqual(set(result['ranked']), {'name', 'address'})
        self.assertTrue(all(item['status'] == 'complete' for item in timings))


if __name__ == '__main__':
    unittest.main()
