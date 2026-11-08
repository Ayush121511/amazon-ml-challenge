"""Merge the France synthetic-alias links into a submission folder.

The links come from `src/postprocessing/synthetic_alias/merge_synthetic_rules.py` (`proposed_synthetic_alias_links_safe.tsv`: S1 id, record id,
channels). CHANNEL=unique_s1_address adds only the pairs found by the unique-S1-address channel; CHANNEL=all adds every
pair. A pair is added only when no business holds the record yet, and a record outside the business's candidate list is
appended to candidate_pairs.tsv. Pairs are added in file order.

Usage: python apply_alias_links.py <base_dir> <links.tsv> <out_dir> <unique_s1_address|all>
Run twice: first with unique_s1_address on the rule-channel file, then with all on its output.
"""
import sys
from collections import Counter
from pathlib import Path


def load_rows(f):
    rows, d = [], {}
    for i, line in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            rows.append(s)
            d[s] = [x for x in m.split(',') if x]
    return rows, d


def main(base, links, out, channel):
    order, u = load_rows(base / 'matching_results.tsv')
    _, cand = load_rows(base / 'candidate_pairs.tsv')
    owner = {t: s for s, ts in u.items() for t in ts}
    R = Counter()
    for i, line in enumerate(open(links, encoding='utf-8')):
        if not i:
            continue
        s, t, channels = line.rstrip('\n').split('\t')[:3]
        if channel != 'all' and channel not in channels.split(','):
            continue
        if t in owner:
            R['already_held'] += 1
            continue
        u[s].append(t)
        owner[t] = s
        R['added'] += 1
        if t not in cand[s]:
            cand[s].append(t)
            R['candidate_added'] += 1
    out.mkdir(parents=True, exist_ok=True)
    for fn, d, col in (('matching_results.tsv', u, 'matched_entity_ids'), ('candidate_pairs.tsv', cand, 'candidate_entity_ids')):
        with open(out / fn, 'w', encoding='utf-8') as fo:
            fo.write(f'source1_entity_id\t{col}\n')
            for s in order:
                fo.write(s + '\t' + ','.join(d[s]) + '\n')
    print(dict(R), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
