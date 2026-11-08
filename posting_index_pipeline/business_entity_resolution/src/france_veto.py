"""France AND-veto: a France match survives only if both the base submission and the self-trained view accept it.

Self-training on a country without labels is used only to veto. US/India matches are the base's, unchanged. The output folder
mirrors a submission folder (rule subfolder + late_scores.npz link) so france_census.py can run on it next.
Env BASE_DIR, SELF_DIR, OUT_DIR. Writes OUT_DIR/<rule>/matching_results.tsv, candidate_pairs.tsv, veto_report.json.
"""
import json
import os
import shutil
from pathlib import Path

HOME = Path.home()
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
BASE, SELF, OUT = Path(os.environ['BASE_DIR']), Path(os.environ['SELF_DIR']), Path(os.environ['OUT_DIR'])


def load(f):
    d = {}
    with open(f, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            d[s] = [x for x in m.split(',') if x]
    return d


def main():
    rd = next(d for d in BASE.iterdir() if (d / 'matching_results.tsv').exists())
    sd = next(d for d in SELF.iterdir() if (d / 'matching_results.tsv').exists())
    base, selfv = load(rd / 'matching_results.tsv'), load(sd / 'matching_results.tsv')
    refs, cty = [], {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[ic]
    out, R = {}, {'france_links_base': 0, 'france_links_vetoed': 0, 'france_refs_emptied_by_veto': 0,
                  'france_links_self_only_ignored': 0}
    for r in refs:
        b = base.get(r, [])
        if cty[r] != 'France':
            out[r] = b
            continue
        s = set(selfv.get(r, []))
        keep = [t for t in b if t in s]
        R['france_links_base'] += len(b)
        R['france_links_vetoed'] += len(b) - len(keep)
        R['france_refs_emptied_by_veto'] += bool(b) and not keep
        R['france_links_self_only_ignored'] += len(s - set(b))
        out[r] = keep
    dest = OUT / rd.name
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs:
            f.write(r + '\t' + ','.join(out[r]) + '\n')
    shutil.copy(rd / 'candidate_pairs.tsv', dest / 'candidate_pairs.tsv')
    link = OUT / 'late_scores.npz'
    if not link.exists():
        os.symlink((BASE / 'late_scores.npz').resolve(), link)
    (OUT / 'veto_report.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
