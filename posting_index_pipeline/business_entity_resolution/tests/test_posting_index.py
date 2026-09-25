import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import transform, RAW, EXTRA
from posting_index import anchor_keys, build, CountryIndex, load_manifest, search, fuse
from posting_index_benchmark import fresh_sample, listed_references, miss_reasons, scripts


def write_source(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW + EXTRA, delimiter='\t')
        writer.writeheader()
        for entity_id, name, address, country in rows:
            writer.writerow(transform({'entity_id': entity_id, 'business_name': name,
                                       'business_address': address, 'country': country}))


def ref(name, address, country='US', entity_id='S1-1'):
    return transform({'entity_id': entity_id, 'business_name': name,
                      'business_address': address, 'country': country})


FILLER = [('Acme Widgets', '12 Oak Lane, Dayton, OH'), ('Blue River Foods', '400 Pine Road, Austin, TX'),
          ('Summit Legal Group', '77 Elm Street, Boise, ID'), ('Northwind Traders', '5 Harbor Way, Salem, OR'),
          ('Golden Gate Bakery', '909 Market Street, Fresno, CA'), ('Red Rock Motors', '31 Canyon Drive, Mesa, AZ')]


class AnchorKeyTests(unittest.TestCase):
    def test_number_needs_a_neighbouring_word(self):
        self.assertEqual(anchor_keys('630 45th terrace kansas city mo'),
                         ['45|terrace', '630|terrace'])
        self.assertEqual(anchor_keys('12 34'), [])

    def test_leading_zeros_and_reordering_agree(self):
        a = set(anchor_keys('af 0684 nandgram ghaziabad'))
        b = set(anchor_keys('af 684 nandgram near school'))
        self.assertIn('684|nandgram', a & b)


class PostingIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        s2 = [('S2-typo', 'Payne Enterpires', '3315 Fremont St, Peoria, IL', 'US'),
              ('S2-addr', 'Drexkor', '85 Wayne Avenue, Ticonderoga, NY', 'US'),
              ('S2-fr', 'Boulangerie Martin', '14 Rue de Rivoli, Paris', 'France')]
        s3 = [('S3-name', 'Maure Williams Colombier', '', 'US'),
              ('S3-fr-us-name', 'Payne Enterprises', '3315 Rue Fremont, Lyon', 'France')]
        s3 += [(f'S3-f{i}', n, a, 'US') for i, (n, a) in enumerate(FILLER)]
        cls.paths = [root / 'x_source2.tsv.gz', root / 'x_source3.tsv.gz']
        write_source(cls.paths[0], s2)
        write_source(cls.paths[1], s3)
        cls.out = root / 'index'
        cls.manifest = build(cls.paths, cls.out, workers=2, chunk_rows=3, min_max_df=1000)
        cls.us = CountryIndex(cls.out, 'US', load_manifest(cls.out))
        cls.fr = CountryIndex(cls.out, 'France', load_manifest(cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest_counts_every_target_once_per_country(self):
        counts = {c: e['targets'] for c, e in self.manifest['countries'].items()}
        self.assertEqual(counts, {'US': 9, 'France': 2})

    def test_typo_name_found_by_name_route(self):
        hits = search(self.us, [ref('Payne Enterprises LLC', '3315 Fremont Street, Peoria, Illinois')])[0]
        self.assertEqual(hits['name'][0][0], 'S2-typo')
        self.assertIn('S2-typo', fuse(hits)[:1])

    def test_unrelated_name_found_by_address_and_anchor(self):
        hits = search(self.us, [ref('Maure Williams Colombier Inc', '85 Wayne Avenue, Ticonderoga, NY')])[0]
        self.assertIn('S2-addr', [i for i, _ in hits['address']])
        self.assertIn('S2-addr', [i for i, _ in hits['anchor']])
        self.assertIn('S3-name', [i for i, _ in hits['name']])

    def test_country_isolation_and_unseen_country(self):
        us_hits = search(self.us, [ref('Payne Enterprises', '3315 Fremont')])[0]
        self.assertNotIn('S3-fr-us-name', fuse(us_hits))
        fr_hits = search(self.fr, [ref('Boulangerie Martin', '14 rue de rivoli', 'France')])[0]
        self.assertEqual(fr_hits['name'][0][0], 'S2-fr')

    def test_ids_and_text_recoverable_in_ordinal_order(self):
        base = self.out / self.manifest['countries']['US']['dir']
        with gzip.open(base / 'targets.tsv.gz', 'rt', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        self.assertEqual([r['entity_id'] for r in rows], [i.decode() for i in self.us.ids])
        self.assertEqual(rows[0]['business_name'], 'Payne Enterpires')

    def test_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            build(self.paths, self.out)

    def test_empty_query_returns_empty_routes(self):
        hits = search(self.us, [ref('', '')])[0]
        self.assertEqual(fuse(hits), [])

    def test_common_terms_are_dropped_from_postings(self):
        root = Path(self.tmp.name)
        out = root / 'capped'
        manifest = build(self.paths, out, workers=1, chunk_rows=100, max_df_fraction=0.0, min_max_df=1)
        routes = manifest['countries']['US']['routes']
        self.assertGreater(routes['name']['terms_dropped_common'], 0)
        self.assertLess(routes['name']['terms_indexed'], routes['name']['terms_seen'])


class SamplingAndReasonTests(unittest.TestCase):
    def test_fresh_sample_is_deterministic_and_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train_source1.tsv.gz'
            write_source(path, [(f'S1-{i}', f'Name {i}', f'{i} Road', 'US' if i % 2 else 'India')
                                for i in range(200)])
            first = fresh_sample(path, {'S1-1', 'S1-2'}, 10)
            second = fresh_sample(path, {'S1-1', 'S1-2'}, 10)
            self.assertEqual([r['entity_id'] for r in first], [r['entity_id'] for r in second])
            self.assertEqual(len(first), 20)
            self.assertFalse({'S1-1', 'S1-2'} & {r['entity_id'] for r in first})

    def test_listed_references_are_exact_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train_source1.tsv.gz'
            write_source(path, [(f'S1-{i}', f'Shop {i}', f'{i} Main St', 'US') for i in (3, 1, 2)])
            self.assertEqual([r['entity_id'] for r in listed_references(path, {'S1-3', 'S1-1'})],
                             ['S1-1', 'S1-3'])
            with self.assertRaises(ValueError):
                listed_references(path, {'S1-1', 'S1-9'})

    def test_script_mismatch_differs_from_non_ascii(self):
        latin = ref('Raj Investments LLP', '6 Colony Road, Chennai', 'India')
        tamil = ref('ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்', '6 Colony Road, Chennai', 'India')
        accented = ref('Payne Énterprises', '3315 Fremont St')
        plain = ref('Payne Enterprises', '3315 Fremont St')
        flags, primary = miss_reasons(latin, tamil)
        self.assertTrue(flags['name_script_mismatch'])
        self.assertEqual(primary, 'name_script_mismatch')
        flags, _ = miss_reasons(plain, accented)
        self.assertFalse(flags['name_script_mismatch'])
        self.assertTrue(flags['non_ascii_same_script'])
        self.assertEqual(scripts('abc'), {'LATIN'})


if __name__ == '__main__':
    unittest.main()
