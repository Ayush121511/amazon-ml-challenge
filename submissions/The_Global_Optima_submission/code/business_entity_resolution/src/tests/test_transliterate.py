import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transliterate import transliterate


class TransliterateTests(unittest.TestCase):
    def test_ascii_is_unchanged(self):
        self.assertEqual(transliterate('abc traders 12'), 'abc traders 12')

    def test_devanagari_business_names(self):
        self.assertEqual(transliterate('शर्मा इलेक्ट्रिकल्स'), 'sharma ilektrikals')
        self.assertEqual(transliterate('श्री गणेश'), 'shri ganesh')
        self.assertEqual(transliterate('कमल'), 'kamal')  # word-final inherent vowel dropped

    def test_other_scripts_and_digits(self):
        self.assertEqual(transliterate('ਗੁਰੂ ਨਾਨਕ'), 'guru nanak')
        self.assertEqual(transliterate('శ్రీ లక్ష్మి'), 'shri lakshmi')
        self.assertEqual(transliterate('राम १२३'), 'ram 123')

    def test_legal_abbreviations(self):
        self.assertEqual(transliterate('प्रा लि'), 'pvt ltd')


if __name__ == '__main__':
    unittest.main()
