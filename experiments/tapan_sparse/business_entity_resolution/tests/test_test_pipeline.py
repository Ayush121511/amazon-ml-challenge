import contextlib
import csv
import gzip
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import RAW, EXTRA, transform
from test_pipeline import cache_build, retrieve, score_country, assemble, COUNTRIES
from validate_final_submission import validate


class FullTestPipelineTest(unittest.TestCase):
    def test_three_countries_resume_scoring_and_official_validator(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data, raw = root / 'data', root / 'raw'
            data.mkdir()
            raw.mkdir()
            for source in (1, 2, 3):
                rows = []
                for country_index, country in enumerate(COUNTRIES):
                    for index in range(3 if source == 1 else 2):
                        rows.append({'entity_id': f'S{source}-{country_index*10+index}',
                                     'business_name': '' if index == 2 else 'Café Blue Harbor',
                                     'business_address': '' if index == 2 else '12 Main Street', 'country': country})
                with (raw / f'test_source{source}.tsv').open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=RAW, delimiter='\t')
                    writer.writeheader()
                    writer.writerows(rows)
                path = data / f'test_source{source}.tsv.gz'
                with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=RAW+EXTRA, delimiter='\t')
                    writer.writeheader()
                    writer.writerows(transform(row) for row in rows)
                Path(str(path) + '.json').write_text(json.dumps({'input': {'version': 2, 'limit': 0},
                                                               'output_bytes': path.stat().st_size}))
            cache = root / 'cache'
            cache_build(SimpleNamespace(data=data, output=cache, fit_rows=2, chunk_rows=1))
            model_dir = root / 'model'
            model_dir.mkdir()
            x = np.vstack([np.zeros((30, 33)), np.ones((30, 33))])
            model = LGBMClassifier(n_estimators=10, min_child_samples=1, verbosity=-1, n_jobs=1)
            model.fit(x, np.array([0]*30 + [1]*30))
            model.booster_.save_model(str(model_dir / 'model.txt'))
            (model_dir / 'report.json').write_text(json.dumps({'threshold_selected_on_dev': .5}))
            for country in COUNTRIES:
                args = SimpleNamespace(cache=cache, country=country, output=root / 'retrieval' / country,
                                       top_k=2, reference_block=2, query_batch=1, threads=1,
                                       checkpoint_chunks=1, max_chunks=1)
                retrieve(args)
                self.assertTrue((args.output / 'block-00000/checkpoint.npz').exists())
                self.assertFalse((args.output / 'complete.json').exists())
                args.max_chunks = 0
                retrieve(args)
                retrieve(args)  # Completed work is reusable.
                self.assertTrue((args.output / 'complete.json').exists())
                uninterrupted = SimpleNamespace(**vars(args))
                uninterrupted.output = root / 'uninterrupted' / country
                retrieve(uninterrupted)
                with np.load(args.output / 'block-00000/candidates.npz') as resumed, \
                     np.load(uninterrupted.output / 'block-00000/candidates.npz') as fresh:
                    np.testing.assert_array_equal(resumed['ids0'], fresh['ids0'])
                    np.testing.assert_array_equal(resumed['scores1'], fresh['scores1'])
                scorer = SimpleNamespace(cache=cache, retrieval=args.output, model=model_dir,
                                         output=root / 'scored' / country, score_batch=2, threads=1)
                score_country(scorer)
                score_country(scorer)
                args.top_k = 3
                with self.assertRaisesRegex(ValueError, 'Resume parameters'):
                    retrieve(args)
            output = root / 'submission'
            assemble(SimpleNamespace(cache=cache, scored=root / 'scored', output=output))
            with (output / 'matching_results.tsv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(rows), 9)
            self.assertEqual({row['matched_entity_ids'] for row in rows if row['source1_entity_id'].endswith('2')}, {''})
            validator = Path(__file__).resolve().parents[2] / '6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py'
            subprocess.run([sys.executable, str(validator), '--matching', str(output / 'matching_results.tsv'),
                            '--candidate', str(output / 'candidate_pairs.tsv'), '--test-dir', str(raw), '--check-ids'],
                           check=True, capture_output=True, timeout=30)
            validate(SimpleNamespace(validator=validator, test_dir=raw, output=output))
            self.assertEqual(json.loads((output / 'VALIDATED.json').read_text())['status'], 'PASS')
            candidate_file = output / 'candidate_pairs.tsv'
            content = candidate_file.read_text(encoding='utf-8')
            first = next(csv.DictReader(io.StringIO(content), delimiter='\t'))['candidate_entity_ids'].split(',')[0]
            candidate_file.write_text(content.replace(first, 'S2-999999', 1), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate/nonexistent'):
                validate(SimpleNamespace(validator=validator, test_dir=raw, output=output))
            self.assertFalse((output / 'VALIDATED.json').exists())


if __name__ == '__main__':
    unittest.main()
