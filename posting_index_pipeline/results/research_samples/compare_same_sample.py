"""Link-level comparison of posting_index v1 and sparse_fullpool on the same 400 references.

Both methods list every missed gold link (with stage), so found/missed sets follow from the
miss files alone: a gold link not listed as missed was found in the top 50.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def load(path, stage_key):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
    return {(r['reference_id'], r['target_id']): r for r in rows}, stage_key


def main(posting_dir, sparse_misses='research/sparse_misses.jsonl'):
    report = json.loads((Path(posting_dir) / 'report.json').read_text(encoding='utf-8'))
    ids = set((Path(posting_dir) / 'reference_ids.txt').read_text(encoding='utf-8').split())
    posting, _ = load(Path(posting_dir) / 'misses.jsonl', 'stage')
    sparse, _ = load(sparse_misses, 'kind')
    stray = {ref for ref, _ in sparse} - ids
    if stray:
        raise SystemExit(f'{len(stray)} sparse-miss references are not in this sample: wrong ID list')
    gold = report['recall']['all']['gold_links']
    p_top, s_top = set(posting), set(sparse)
    p_union = {k for k, v in posting.items() if v['stage'] == 'absent_from_routes'}
    s_union = {k for k, v in sparse.items() if v['kind'] == 'absent_from_routes'}
    out = {'references': len(ids), 'gold_links': gold}
    for name, p_miss, s_miss in (('top50', p_top, s_top), ('union', p_union, s_union)):
        out[name] = {'posting_recall': 1 - len(p_miss) / gold, 'sparse_recall': 1 - len(s_miss) / gold,
                     'both_found': gold - len(p_miss | s_miss), 'only_posting': len(s_miss - p_miss),
                     'only_sparse': len(p_miss - s_miss), 'both_missed': len(p_miss & s_miss),
                     'only_sparse_by_country_reason': dict(Counter(
                         f"{posting[k]['country']}:{posting[k]['primary_reason']}" for k in p_miss - s_miss))}
    out['posting_union_candidates_mean'] = report['recall']['all']['mean_union_candidates']
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
