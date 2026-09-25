import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_matcher_errors import loss_components, feature_names


class LossAttributionTest(unittest.TestCase):
    def test_loss_adds_up_for_all_error_types(self):
        result = loss_components({'a', 'b', 'c'}, {'a', 'b', 'wrong'}, {'a', 'wrong'})
        self.assertGreater(result['blocking_loss'], 0)
        self.assertGreater(result['rejected_true_candidate_loss'], 0)
        self.assertGreater(result['false_positive_loss'], 0)
        self.assertAlmostEqual(sum(result[key] for key in ('blocking_loss', 'rejected_true_candidate_loss', 'false_positive_loss')),
                               1 - result['macro_f05'])

    def test_singleton_and_empty_retrieval(self):
        singleton = loss_components(set(), {'wrong'}, {'wrong'})
        self.assertEqual(singleton['false_positive_loss'], 1)
        self.assertEqual(singleton['blocking_loss'], 0)
        absent = loss_components({'a'}, set(), set())
        self.assertEqual(absent['blocking_loss'], 1)
        self.assertEqual(absent['rejected_true_candidate_loss'], 0)
        self.assertEqual(len(feature_names()), 33)


if __name__ == '__main__':
    unittest.main()
