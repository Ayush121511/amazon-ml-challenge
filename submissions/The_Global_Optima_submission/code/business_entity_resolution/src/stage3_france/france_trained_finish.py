"""Finish the France-trained run from its saved outputs (the training job stopped at a disk-quota error after writing the
model and late_scores.npz): decision rule + global exclusivity on the saved probabilities, France rows merged into T5's
US/India decisions, pre-census output in artifacts/submission_FRtrained/<rule>/ (same layout as FRrobust)."""
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(A / 'business_entity_resolution/src')]
from submission_pipeline import finalize  # noqa: E402

OUT = ME / 'artifacts/submission_FRtrained'
T5TEST = ME / 'artifacts/submission_T5_own'


def main():
    meta = json.load(open(OUT / 'stage3b_frtrained_meta.json'))
    rule = meta['rule']
    Z = np.load(OUT / 'late_scores.npz', allow_pickle=True)
    refs_t = []
    with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            refs_t.append({'entity_id': row['entity_id'], 'country': row['country']})
    cands_out = {}
    with open(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s_, c_ = line.rstrip('\n').split('\t'); cands_out[s_] = c_
    tmp = OUT.parent / 'submission_FRtrained_all_countries_tmp'
    tmp.mkdir(exist_ok=True)
    finalize(refs_t, cands_out, Z['ref_idx'], Z['target'].astype(str).astype(object), Z['prob'].astype(np.float32),
             {'decision': {'rule': rule[0], 'threshold': rule[1]}}, True, tmp, Counter(),
             {'model': 'france_trained', 'started': time.monotonic()}, {})

    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f); return {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in f}
    new, t5 = load(tmp / 'matching_results.tsv'), load(next(T5TEST.glob('expected_*')) / 'matching_results.tsv')
    dest = OUT / f'{rule[0]}_{rule[1]}'
    dest.mkdir(exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs_t:
            e = r['entity_id']
            f.write(e + '\t' + (new[e] if r['country'] == 'France' else t5[e]) + '\n')
    shutil.copy(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', dest / 'candidate_pairs.tsv')
    shutil.rmtree(tmp)
    rep = json.load(open(OUT / 'report.json'))
    rep['chosen'] = {k: v for k, v in meta.items() if k != 'cols'}
    rep['output'] = str(dest)
    rep.pop('verdict', None)
    (OUT / 'report.json').write_text(json.dumps(rep, indent=1))
    print('wrote', dest, 'rule', rule, flush=True)


if __name__ == '__main__':
    main()
