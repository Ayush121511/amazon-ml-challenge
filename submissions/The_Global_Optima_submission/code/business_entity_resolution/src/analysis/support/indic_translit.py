"""Indic-script names to Latin, and consonant skeletons for matching across scripts. No external library.

Every Indic code point has a Unicode name such as 'TELUGU LETTER RA', 'TAMIL VOWEL SIGN II' or 'DEVANAGARI SIGN
VIRAMA'; the letter part gives the Latin sound. Consonants carry an inherent 'a' that a vowel sign replaces and the
virama removes. Covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam.
skeleton() keeps consonants only (vowels are where transliterations of English words disagree), with common
aspirated / retroflex spellings merged, so 'Urban Properties' and 'arban praapartiis' both give 'rbn prprts'.
"""
import re
import unicodedata

SCRIPTS = ('DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM')
INDIC = re.compile(r'[ऀ-෿]')
VOWEL_SOUND = {'AA': 'aa', 'I': 'i', 'II': 'ii', 'U': 'u', 'UU': 'uu', 'E': 'e', 'EE': 'ee', 'AI': 'ai', 'O': 'o',
               'OO': 'oo', 'AU': 'au', 'VOCALIC R': 'ri', 'A': 'a', 'CANDRA E': 'e', 'CANDRA O': 'o', 'SHORT E': 'e',
               'SHORT O': 'o'}


def translit(s):
    out, pending = [], False
    for ch in s:
        if not INDIC.match(ch):
            if pending:
                out.append('a'); pending = False
            out.append(ch)
            continue
        name = unicodedata.name(ch, '')
        m = re.match(r'(?:' + '|'.join(SCRIPTS) + r') (LETTER|VOWEL SIGN|SIGN|DIGIT) (.+)', name)
        if not m:
            continue
        kind, what = m.groups()
        if kind == 'LETTER':
            if pending:
                out.append('a')
            what = what.split(' ')[-1] if what.split(' ')[0] in ('LLLA', 'NNNA', 'RRA') else what
            if what in VOWEL_SOUND:
                out.append(VOWEL_SOUND[what]); pending = False
            else:
                out.append(what.lower().rstrip('a') or what.lower()); pending = True
        elif kind == 'VOWEL SIGN':
            out.append(VOWEL_SOUND.get(what, '')); pending = False
        elif kind == 'SIGN':
            if what == 'VIRAMA':
                pending = False
            elif what in ('ANUSVARA', 'CANDRABINDU'):
                if pending:
                    out.append('a'); pending = False
                out.append('n')
        elif kind == 'DIGIT':
            out.append(str(['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'].index(what)))
    if pending:
        out.append('a')
    return ''.join(out)


MERGE = [('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('jh', 'j'), ('th', 't'), ('dh', 'd'), ('ph', 'f'), ('bh', 'b'),
         ('sh', 's'), ('tt', 't'), ('dd', 'd'), ('nn', 'n'), ('ll', 'l'), ('ss', 's'), ('rr', 'r'), ('ck', 'k'),
         ('q', 'k'), ('x', 'ks'), ('z', 'j'), ('w', 'v'), ('y', ''), ('h', ''), ('p', 'f')]


def skeleton(s):
    s = unicodedata.normalize('NFKD', translit(s))
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    words = []
    for w in re.sub(r'[^a-z0-9]+', ' ', s).split():
        for a, b in MERGE:
            w = w.replace(a, b)
        w = re.sub(r'[aeiou]', '', w)
        w = re.sub(r'(.)\1+', r'\1', w)
        if w:
            words.append(w)
    return ' '.join(words)
