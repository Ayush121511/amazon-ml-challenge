import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from compact_topk import CompactTopK, decode_target_id, encode_target_id


class CompactTopKTests(unittest.TestCase):
    def test_id_roundtrip(self):
        for identifier in ('S2-0', 'S2-123456789', 'S3-987654321'):
            self.assertEqual(decode_target_id(encode_target_id(identifier)), identifier)

    def test_keeps_global_best_across_chunks_and_batches(self):
        store = CompactTopK(3, 2)
        first = csr_matrix(np.array([[.2, .8, 0], [0, .5, .4]], dtype=np.float32))
        store.update(0, first, np.array([encode_target_id(x) for x in
                     ('S2-1','S2-2','S2-3')], dtype=np.uint64))
        second = csr_matrix(np.array([[.9, .1], [.1, .7]], dtype=np.float32))
        store.update(0, second, np.array([encode_target_id(x) for x in
                     ('S3-4','S3-5')], dtype=np.uint64))
        self.assertEqual([identifier for _,identifier in store.ranked(0)],
                         ['S3-4','S2-2'])
        self.assertEqual([identifier for _,identifier in store.ranked(1)],
                         ['S3-5','S2-2'])
        self.assertEqual(store.ranked(2), [])


if __name__ == '__main__':
    unittest.main()
