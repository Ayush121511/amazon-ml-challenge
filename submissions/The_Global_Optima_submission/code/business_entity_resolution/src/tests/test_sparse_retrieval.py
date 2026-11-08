import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_retrieval_benchmark import run_channel


class SparseRetrievalTests(unittest.TestCase):
    def test_name_and_address_recover_different_pairs(self):
        targets=[
            {'entity_id':'name-match','name_folded':'cafe de paris','address_folded':'11 distant road'},
            {'entity_id':'address-match','name_folded':'renamed company','address_folded':'5 market street'},
            {'entity_id':'name-second','name_folded':'cafe de paris west','address_folded':'22 unknown lane'},
            {'entity_id':'address-second','name_folded':'other name','address_folded':'5 market street west'},
            {'entity_id':'other','name_folded':'another entity','address_folded':'90 distant avenue'}]
        refs=[{'name_folded':'café de paris','address_folded':'5 market street'}]
        name,_=run_channel(refs,targets,'name_folded',2,1)
        address,_=run_channel(refs,targets,'address_folded',2,1)
        self.assertIn('name-match',[identifier for identifier,_ in name[0]])
        self.assertIn('address-match',[identifier for identifier,_ in address[0]])
