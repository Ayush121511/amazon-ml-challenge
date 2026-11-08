"""Final submission assembly: take every S1 row from BASE_TSV except the listed countries, which come from OVERRIDE_TSV.

U2 = BASE (T6 decisions for US/India) with France from the France pipeline (FRtrained_w4 -> exclusivity_prune.py ->
france_census.py CENSUS_V2=1). Row order follows BASE_TSV. Env BASE_TSV, OVERRIDE_TSV, OUT_TSV, COUNTRIES (default
France), TEST_RAW (test folder with test_source1.tsv).
"""
import os
from pathlib import Path


def load(f):
    rows, d = [], {}
    with open(f, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            rows.append(s); d[s] = m
    return rows, d


def main():
    raw = Path(os.environ.get('TEST_RAW', str(Path.home() / 'scratch/amazon_ml_2026/data/dataset/test')))
    C = set(os.environ.get('COUNTRIES', 'France').split(','))
    cty = {}
    with open(raw / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t'); cty[q[ie]] = q[ic]
    rows, base = load(os.environ['BASE_TSV'])
    _, over = load(os.environ['OVERRIDE_TSV'])
    Path(os.environ['OUT_TSV']).parent.mkdir(parents=True, exist_ok=True)
    with open(os.environ['OUT_TSV'], 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s in rows:
            f.write(s + '\t' + (over[s] if cty[s] in C else base[s]) + '\n')
    print('wrote', os.environ['OUT_TSV'], 'overridden', sum(cty[s] in C for s in rows), flush=True)


if __name__ == '__main__':
    main()
