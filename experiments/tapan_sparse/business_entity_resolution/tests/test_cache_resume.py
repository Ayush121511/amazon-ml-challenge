"""Recovery must reuse complete chunks and replace a partially written one."""
import contextlib
import csv
import gzip
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import RAW, EXTRA, transform
from resume_test_cache import resume
from test_pipeline import COUNTRIES, cache_build


class ResumeCacheTest(unittest.TestCase):
    def test_interrupted_final_chunk_and_sqlite_rows(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            for source in (1, 2, 3):
                path = data / f'test_source{source}.tsv.gz'
                rows = [{'entity_id': f'S{source}-{country_index*10+index}',
                         'business_name': 'Blue Harbor Cafe',
                         'business_address': '12 Main Street', 'country': country}
                        for country_index, country in enumerate(COUNTRIES)
                        for index in range(1 if source == 1 else 2)]
                with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=RAW+EXTRA, delimiter='\t')
                    writer.writeheader()
                    writer.writerows(transform(row) for row in rows)
                Path(str(path) + '.json').write_text(json.dumps(
                    {'input': {'version': 2, 'limit': 0}, 'output_bytes': path.stat().st_size}))
            cache = root / 'cache'
            args = SimpleNamespace(data=data, output=cache, fit_rows=2, chunk_rows=2)
            cache_build(args)
            original = json.loads((cache / 'manifest.json').read_text())
            country = 'India'
            last = original['countries'][country]['chunks'][-1]['prefix']
            self.assertEqual(len(np.load(cache / country / f'{last}-ids.npy')), 2)
            expected = sparse.load_npz(cache / country / f'{last}-address_folded.npz').toarray()
            (cache / 'manifest.json').unlink()
            # Simulate a kill after the SQLite commit and ID/name writes.
            (cache / country / f'{last}-address_folded.npz').unlink()
            resume(SimpleNamespace(data=data, cache=cache, fit_rows=2, chunk_rows=2))
            rebuilt = json.loads((cache / 'manifest.json').read_text())
            for name in COUNTRIES:
                self.assertEqual(rebuilt['countries'][name]['targets'], 4)
                with closing(sqlite3.connect(cache / name / 'targets.sqlite')) as database:
                    self.assertEqual(database.execute('SELECT COUNT(*) FROM targets').fetchone()[0], 4)
            np.testing.assert_array_equal(
                sparse.load_npz(cache / country / f'{last}-address_folded.npz').toarray(), expected)
            self.assertEqual(rebuilt['resume_from_targets'][country], 2)
            resume(SimpleNamespace(data=data, cache=cache, fit_rows=2, chunk_rows=2))


if __name__ == '__main__':
    unittest.main()
