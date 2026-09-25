import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pair_features import feature_spec
from preprocess import transform


class PairFeaturesTest(unittest.TestCase):
    def values(self, name_a, name_b, address_a, address_b):
        make = lambda identifier, name, address: transform({'entity_id': identifier, 'business_name': name,
                                                           'business_address': address, 'country': 'US'})
        pair = {'reference': make('S1-1', name_a, address_a), 'target': make('S2-2', name_b, address_b),
                'retrieval_ranks': {'name_words': 1, 'address_words': 2, 'name_trigrams': None}}
        extract, names = feature_spec('v2')
        values = extract(pair)
        self.assertEqual(len(values), len(names))
        baseline, old_names = feature_spec('v1')
        self.assertEqual(values[:len(old_names)], baseline(pair))
        return dict(zip(names, values))

    def test_number_zero_padding_and_conflict(self):
        same = self.values('Harbor Shop LLC', 'Harbor Shop', '07225 Main Street', '7225 Main St')
        different = self.values('Harbor Shop', 'Harbor Shop LLC', '508 Main Street', '509 Main Street')
        self.assertEqual(same['first_address_number:equal'], 1)
        self.assertEqual(same['address_numbers_canonical:jaccard'], 1)
        self.assertEqual(same['name_core:exact'], 1)
        self.assertEqual(different['first_address_number:conflict'], 1)

    def test_alphanumeric_and_missing_addresses(self):
        conflict = self.values('Grameen Designer Pvt Ltd', 'CLC Forge Pvt Ltd', 'B-36A Third Floor', 'S-36A Third Floor')
        self.assertEqual(conflict['first_address_number:equal'], 1)
        self.assertEqual(conflict['first_address_number:prefix_conflict'], 1)
        self.assertEqual(conflict['name_core:no_token_overlap'], 1)
        empty = self.values('Shop', 'Shop', '', '')
        self.assertEqual(empty['first_address_number:equal'], 0)
        self.assertEqual(empty['first_address_number:conflict'], 0)
        self.assertEqual(empty['first_address_number:reference_missing'], 1)


if __name__ == '__main__':
    unittest.main()
