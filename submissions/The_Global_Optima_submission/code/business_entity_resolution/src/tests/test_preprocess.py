import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import RAW, normalize, transform, process_file


class PreprocessingTests(unittest.TestCase):
    def test_indic_marks_survive_both_views(self):
        for word in ['தமிழ்', 'ગુજરાતી', 'हिन्दी', 'বাংলা', 'తెలుగు']:
            with self.subTest(word=word):
                self.assertEqual(normalize(word), word)
                self.assertEqual(normalize(word, True), word)

    def test_mixed_scripts_and_decomposed_latin(self):
        self.assertEqual(normalize('Cafe\u0301 தமிழ் ગુજરાતી', True),
                         'cafe தமிழ் ગુજરાતી')
        self.assertEqual(normalize('Cafe\u0301 தமிழ்'), 'café தமிழ்')
        self.assertEqual(normalize('عَرَبِيّ', True), 'عَرَبِيّ')

    def test_joiners_preserved(self):
        for joiner in ['\u200c', '\u200d']:
            word = 'क्' + joiner + 'ष'
            self.assertEqual(normalize(word, True), word)

    def test_unicode_and_numbers(self):
        self.assertEqual(normalize('  CAFÉ & Co. １２ '), 'café and co 12')
        self.assertEqual(normalize('CAFÉ', True), 'cafe')
        self.assertIn('東京', normalize('東京'))

    def test_originals_and_missingness(self):
        row = dict(zip(RAW, ['S2-001', 'Dréxkor', '', 'France']))
        result = transform(row)
        self.assertEqual({k: result[k] for k in RAW}, row)
        self.assertEqual(result['address_missing'], '1')
        self.assertEqual(result['postal_candidates'], '')

    def test_postal_candidate_not_truth(self):
        row = dict(zip(RAW, ['S1-x', 'Shop', '12 Road 110001', 'India']))
        self.assertEqual(transform(row)['postal_candidates'], '110001')
        row['country'] = 'US'
        self.assertEqual(transform(row)['postal_candidates'], '')

    def test_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'input.tsv'
            dst = Path(directory) / 'output.tsv.gz'
            row = ['S1-01', 'Café\tShop', 'A\nB', 'France']
            with src.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(RAW)
                writer.writerow(row)
            report = process_file((src, dst, 0, False))
            with gzip.open(dst, 'rt', encoding='utf-8', newline='') as f:
                result = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual([result[0][k] for k in RAW], row)
            self.assertEqual(report['rows'], 1)
            self.assertEqual(process_file((src, dst, 0, True)), report)
            with self.assertRaises(FileExistsError):
                process_file((src, dst, 1, True))


if __name__ == '__main__':
    unittest.main()
