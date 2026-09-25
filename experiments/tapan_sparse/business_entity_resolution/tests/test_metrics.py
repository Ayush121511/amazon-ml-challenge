import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import entity_f05, macro_f05


class MetricTests(unittest.TestCase):
    def test_official_example(self):
        self.assertAlmostEqual(entity_f05({"a", "b"}, {"a", "b", "c"}), 5 / 7)

    def test_singletons(self):
        self.assertEqual(entity_f05(set(), set()), 1)
        self.assertEqual(entity_f05(set(), {"a"}), 0)
        self.assertEqual(entity_f05({"a"}, set()), 0)

    def test_macro_weights_entities_equally(self):
        gold = {"small": {"x"}, "large": {str(i) for i in range(100)}, "singleton": set()}
        pred = {"small": set(), "large": gold["large"], "singleton": set()}
        self.assertAlmostEqual(macro_f05(gold, pred), 2 / 3)

    def test_false_positive_cost(self):
        self.assertAlmostEqual(entity_f05({"a"}, {"a", "b"}), 5 / 9)
        self.assertAlmostEqual(entity_f05({"a", "b"}, {"a"}), 5 / 6)

    def test_complete_coverage_required(self):
        with self.assertRaises(ValueError):
            macro_f05({"s1": set()}, {})
        with self.assertRaises(ValueError):
            macro_f05({}, {})


if __name__ == "__main__":
    unittest.main()
