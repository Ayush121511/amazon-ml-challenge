"""Versioned pair features; numeric conflicts are evidence, never hard vetoes."""
import re
from functools import lru_cache

from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from pilot_matcher import features as baseline_features


def baseline_names():
    names = [f'{field}:{measure}' for field in ('name_norm', 'name_folded', 'address_norm', 'address_folded')
             for measure in ('ratio', 'token_sort_ratio', 'token_set_ratio', 'exact', 'length_ratio')]
    names += [f'{field}:{measure}' for field in ('address_numbers', 'postal_candidates')
              for measure in ('jaccard', 'disjoint', 'reference_missing', 'target_missing')]
    return names + ['reference_address_missing', 'target_address_missing'] + [
        f'{field}:inverse_rank' for field in ('name_words', 'address_words', 'name_trigrams')]


EXTRA_NAMES = [
    'name_core:ratio', 'name_core:token_sort_ratio', 'name_core:token_set_ratio',
    'name_core:exact', 'name_core:jaccard', 'name_core:reference_coverage',
    'name_core:target_coverage', 'name_core:no_token_overlap',
    'name_core:initials_equal', 'name_core:reference_initialism', 'name_core:target_initialism',
    'address_numbers_canonical:jaccard', 'address_numbers_canonical:reference_coverage',
    'address_numbers_canonical:target_coverage', 'address_numbers_canonical:disjoint',
    'first_address_number:equal', 'first_address_number:conflict',
    'first_address_number:reference_missing', 'first_address_number:target_missing',
    'first_address_number:prefix_conflict', 'first_address_number:suffix_conflict',
    'address_tokens:jaccard', 'address_tokens:reference_coverage', 'address_tokens:target_coverage',
]

LEGAL_SUFFIXES = frozenset(('inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private',
                           'corp', 'corporation', 'co', 'company', 'llp', 'plc', 'sarl', 'sasu', 'sas', 'sa'))


@lru_cache(maxsize=50000)
def name_parts(value):
    tokens = value.split()
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return ' '.join(tokens), frozenset(tokens), ''.join(token[0] for token in tokens)


@lru_cache(maxsize=50000)
def address_parts(value):
    matches = list(re.finditer(r'\d+', value))
    numbers = frozenset(str(int(match.group())) for match in matches)
    if not matches:
        return numbers, None, '', '', frozenset(value.split())
    first = matches[0]
    before, after = value[:first.start()], value[first.end():]
    prefix = re.search(r'(?<![a-z])([a-z])\s*$', before)
    suffix = re.match(r'([a-z])(?:\b|\s)', after)
    return numbers, str(int(first.group())), prefix.group(1) if prefix else '', suffix.group(1) if suffix else '', frozenset(value.split())


def overlap(a, b):
    common = len(a & b)
    return [common / max(1, len(a | b)), common / max(1, len(a)), common / max(1, len(b))]


def features_v2(pair):
    a, b = pair['reference'], pair['target']
    x, xt, xi = name_parts(a['name_folded'])
    y, yt, yi = name_parts(b['name_folded'])
    valid = bool(x and y)
    extra = [ratio(x, y)/100 if valid else 0, token_sort_ratio(x, y)/100 if valid else 0,
             token_set_ratio(x, y)/100 if valid else 0, int(valid and x == y)]
    extra += overlap(xt, yt)
    extra += [int(bool(xt and yt) and not xt & yt), int(valid and xi == yi),
              int(valid and len(yt) > 1 and x.replace(' ', '') == yi),
              int(valid and len(xt) > 1 and y.replace(' ', '') == xi)]
    an, af, ap, az, at = address_parts(a['address_folded'])
    bn, bf, bp, bz, bt = address_parts(b['address_folded'])
    extra += overlap(an, bn)
    extra += [int(bool(an and bn) and not an & bn), int(af is not None and af == bf),
              int(af is not None and bf is not None and af != bf), int(af is None), int(bf is None),
              int(bool(ap and bp) and ap != bp), int(bool(az and bz) and az != bz)]
    extra += overlap(at, bt)
    return baseline_features(pair) + extra


def feature_spec(version='v1'):
    if version == 'v1':
        return baseline_features, baseline_names()
    if version == 'v2':
        return features_v2, baseline_names() + EXTRA_NAMES
    raise ValueError(f'Unsupported feature version: {version}')
