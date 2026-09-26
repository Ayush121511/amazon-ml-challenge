"""Rule-based Brahmic (Indic) script -> Latin transliteration from Unicode character names.

Offline and label-free: no external lookup or data. Unicode names such as
"DEVANAGARI LETTER KA" / "TAMIL VOWEL SIGN II" / "BENGALI SIGN VIRAMA" carry the phonetic
value, which is mapped to a loose ASCII spelling close to how Indian business names are
written in Latin script (long vowels collapsed, retroflex marks dropped, word-final
inherent vowel removed as in Hindi "kamal" rather than "kamala").
"""
import re
import unicodedata
from functools import lru_cache

SCRIPTS = ('DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU',
           'KANNADA', 'MALAYALAM')
CONSONANTS = {'KA': 'k', 'KHA': 'kh', 'GA': 'g', 'GHA': 'gh', 'NGA': 'n', 'CA': 'ch', 'CHA': 'chh',
              'JA': 'j', 'JHA': 'jh', 'NYA': 'n', 'TTA': 't', 'TTHA': 'th', 'DDA': 'd', 'DDHA': 'dh',
              'NNA': 'n', 'TA': 't', 'THA': 'th', 'DA': 'd', 'DHA': 'dh', 'NA': 'n', 'NNNA': 'n',
              'PA': 'p', 'PHA': 'f', 'BA': 'b', 'BHA': 'bh', 'MA': 'm', 'YA': 'y', 'YYA': 'y',
              'RA': 'r', 'RRA': 'r', 'LA': 'l', 'LLA': 'l', 'LLLA': 'l', 'VA': 'v', 'SHA': 'sh',
              'SSA': 'sh', 'SA': 's', 'HA': 'h', 'QA': 'q', 'KHHA': 'kh', 'GHHA': 'g', 'ZA': 'z',
              'DDDHA': 'd', 'RHA': 'r', 'FA': 'f', 'YYYA': 'y'}
VOWELS = {'A': 'a', 'AA': 'a', 'I': 'i', 'II': 'i', 'U': 'u', 'UU': 'u', 'E': 'e', 'EE': 'e',
          'AI': 'ai', 'O': 'o', 'OO': 'o', 'AU': 'au', 'VOCALIC R': 'ri', 'VOCALIC RR': 'ri',
          'VOCALIC L': 'li', 'SHORT E': 'e', 'SHORT O': 'o', 'CANDRA E': 'e', 'CANDRA O': 'o'}


@lru_cache(maxsize=None)
def classify(char):
    """(kind, latin) for one character; kind in consonant/vowel/sign/virama/other."""
    name = unicodedata.name(char, '')
    script = name.split(' ', 1)[0]
    if script not in SCRIPTS:
        return 'other', char
    rest = name[len(script) + 1:]
    if rest.startswith('LETTER '):
        value = rest[7:]
        if value in CONSONANTS:
            return 'consonant', CONSONANTS[value]
        if value in VOWELS:
            return 'vowel', VOWELS[value]
        return 'other', ''
    if rest.startswith('VOWEL SIGN '):
        return 'sign', VOWELS.get(rest[11:], '')
    if rest in ('SIGN VIRAMA', 'SIGN HALANT'):
        return 'virama', ''
    if rest in ('SIGN ANUSVARA', 'SIGN CANDRABINDU', 'SIGN TIPPI', 'SIGN ADAK BINDI'):
        return 'other', 'n'
    if rest == 'SIGN VISARGA':
        return 'other', 'h'
    if rest.startswith('DIGIT '):
        return 'other', str(unicodedata.digit(char))
    return 'other', ''  # nukta, length marks, avagraha, etc.


def transliterate_word(word):
    out, pending = [], False  # pending: last consonant still carries its inherent 'a'
    for char in word:
        kind, latin = classify(char)
        if kind == 'consonant':
            if pending:
                out.append('a')
            out.append(latin)
            pending = True
        elif kind == 'sign':
            out.append(latin)
            pending = False
        elif kind == 'virama':
            pending = False
        else:
            if pending and latin:
                out.append('a')
            pending = False
            out.append(latin)
    # Word-final inherent vowel is dropped (schwa deletion), except after a lone consonant.
    if pending and len(out) == 1:
        out.append('a')
    return ''.join(out)


def has_indic(text):
    return any(classify(c)[0] != 'other' or (ord(c) > 127 and classify(c)[1] != c) for c in text)


# Hindi legal-form abbreviations (pra. li. = pvt. ltd.) and their spelled-out loanwords.
WORDS = {'pra': 'pvt', 'li': 'ltd', 'praivet': 'private', 'praivat': 'private', 'limited': 'limited'}


def transliterate(text):
    """Latin text for Indic words; other words are returned unchanged."""
    if text.isascii():
        return text
    words = (transliterate_word(w) for w in text.split())
    return ' '.join(WORDS.get(w, w) for w in words if w)


# ---------------------------------------------------------------- phonetic ----

PHONETIC_STOP = frozenset('private pvt limited ltd llp llc inc and the co company'.split())
_PHONETIC_RULES = (('ksh', 'ks'), ('x', 'ks'), ('sch', 's'), ('sh', 's'), ('ph', 'f'), ('ck', 'k'),
                   ('q', 'k'), ('z', 'j'), ('w', 'v'), ('th', 't'), ('dh', 'd'), ('bh', 'b'),
                   ('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('ee', 'i'), ('oo', 'u'))


def phonetic_word(word):
    """Loose sound skeleton of one Latin word: spelling and transliteration variants
    ("classic finance" / "klasik fainens", "lakshmi" / "laxmi") map to the same key."""
    w = word
    for a, b in _PHONETIC_RULES:
        w = w.replace(a, b)
    w = re.sub(r'c(?=[eiy])', 's', w)   # soft c
    w = re.sub(r'g(?=[ei])', 'j', w)    # soft g
    w = w.replace('c', 'k')
    if w.startswith('y') and len(w) > 1 and w[1] in 'aeiou':
        w = w[1:]
    head, tail = w[:1], w[1:]
    if head in 'aeiou':
        head = 'a'
    tail = re.sub(r'[aeiouyh]', '', tail)
    return re.sub(r'(.)\1+', r'\1', head + tail)


def phonetic(text):
    """Sound skeleton of a name: transliterated, legal words dropped, vowels removed."""
    words = [w for w in transliterate(text).split() if w not in PHONETIC_STOP]
    return ' '.join(phonetic_word(w) for w in words if not w.isdigit())
