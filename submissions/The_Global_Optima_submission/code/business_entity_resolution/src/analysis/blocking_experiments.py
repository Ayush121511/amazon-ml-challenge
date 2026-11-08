"""Blocking and retrieval experiments.

Each command is one analysis script. Run `python blocking_experiments.py <command> [args]`; `python blocking_experiments.py` lists the commands.
"""
import sys


def cmd_benchmark_capped():
    """Test capped global retrieval against saved original results; never replace baseline."""
    import argparse
    import json
    import random
    import sqlite3
    import time
    from collections import defaultdict
    from pathlib import Path
    from diagnose_retrieval import read_saved
    from retrieval_experiment import queries, evaluate


    def capped_query(db, table, query, country, cap):
        if table not in {'words', 'chars'}:
            raise ValueError('Invalid table')
        # LIMIT prevents flattening: rank a bounded global shortlist before the join.
        return [r[0] for r in db.execute(f'''SELECT e.entity_id FROM
        (SELECT rowid, rank FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?) AS hits
        JOIN entities e ON e.id=hits.rowid WHERE e.country=?
        ORDER BY hits.rank, e.entity_id LIMIT 50''', (query, cap, country))]


    def main():
        p = argparse.ArgumentParser(description=cmd_benchmark_capped.__doc__)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--ground-truth', type=Path, required=True)
        args = p.parse_args()
        refs = json.loads((args.run/'references.json').read_text(encoding='utf-8'))
        saved = read_saved(args.run/'candidates.jsonl', refs)
        available = {r['entity_id']: r for r in saved}
        rng = random.Random(2027)
        sample = []
        for country in sorted({r['country'] for r in saved}):
            pool = [r for r in refs if r['country'] == country and r['entity_id'] in available]
            sample.extend(rng.sample(pool, min(6, len(pool))))
        db = sqlite3.connect((args.run/'targets.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
        db.execute('PRAGMA cache_size=-131072')
        report = {'sample_ids': [r['entity_id'] for r in sample],
                  'baseline': evaluate([available[r['entity_id']] for r in sample], args.ground_truth),
                  'warning': 'Small development benchmark; global caps can reduce country-specific recall. Timed-out results are unavailable, not measured misses.',
                  'variants': {}}
        for cap in [300, 1000]:
            results, timings = [], []
            for row in sample:
                channels, fusion, complete = {}, defaultdict(float), True
                for label, table, query, enabled in queries(row):
                    start = time.monotonic()
                    db.set_progress_handler(lambda: int(time.monotonic()-start > 3), 1000)
                    try:
                        ids = capped_query(db, table, query, row['country'], cap) if enabled else []
                        status = 'ok'
                    except sqlite3.OperationalError as exc:
                        if 'interrupted' not in str(exc):
                            raise
                        ids, status, complete = [], 'timeout', False
                    finally:
                        db.set_progress_handler(None, 0)
                    channels[label] = ids
                    for rank, identifier in enumerate(ids, 1):
                        fusion[identifier] += 1/(60+rank)
                    timing = {'id': row['entity_id'], 'country': row['country'], 'channel': label,
                              'cap': cap, 'seconds': time.monotonic()-start, 'status': status, 'hits': len(ids)}
                    timings.append(timing)
                    print(json.dumps(timing), flush=True)
                if complete:
                    results.append({'entity_id': row['entity_id'], 'country': row['country'],
                        'non_ascii_name': any(ord(c)>127 for c in row['name_norm']), 'channels': channels,
                        'ranked': sorted(fusion, key=lambda x: (-fusion[x], x))[:50]})
            report['variants'][str(cap)] = {'timings': timings, 'completed_references': len(results),
                 'recall_completed_only': evaluate(results, args.ground_truth) if results else None,
                 'baseline_same_completed': evaluate([available[r['entity_id']] for r in results], args.ground_truth) if results else None}
            (args.run/'capped_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        db.close()
        print('Capped benchmark complete. Baseline files unchanged.', flush=True)


    if __name__ == '__main__':
        main()



def cmd_blocking_cutoff_cost():
    """How many true links each candidate-rank cutoff keeps."""
    import gzip
    import json
    import sys
    from pathlib import Path

    import numpy as np


    def load_pairs(d):
        refs, targets = [], []
        with gzip.open(d / 'pairs.tsv.gz', 'rt') as f:
            for line in f:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r)
                targets.append(t)
        return refs, targets


    def main():
        root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'scratch/vishal_v4_artifacts'
        pairs_dir = root / 'pairs'
        model_dir = root / 'model_xgb'

        x = np.load(pairs_dir / 'x.npy', mmap_mode='r')
        y = np.load(pairs_dir / 'y.npy')
        report = json.loads((model_dir / 'report.json').read_text())
        names = report['features']
        fr_idx = names.index('fused_rank')
        fused_rank = np.asarray(x[:, fr_idx])
        del x

        print(f'total pairs: {len(y)}, total true (y=1): {int(y.sum())}', flush=True)

        refs, targets = load_pairs(pairs_dir)
        ref_arr = np.array(refs, dtype=object)
        del refs, targets

        print('encoding references...', flush=True)
        uniq_refs, ref_codes = np.unique(ref_arr, return_inverse=True)
        n_refs = len(uniq_refs)
        print(f'distinct references: {n_refs}', flush=True)

        total_true = int(y.sum())

        zero_mask = fused_rank == 0
        print(f'fused_rank==0 (reverse-only sentinel) pairs: {int(zero_mask.sum())} '
              f'({int(zero_mask.sum())/len(y):.4%}), true among them: {int((zero_mask & (y==1)).sum())}',
              flush=True)

        rev_idx = names.index('rev_rank') if 'rev_rank' in names else None
        rev_rank = np.asarray(np.load(pairs_dir / 'x.npy', mmap_mode='r')[:, rev_idx]) if rev_idx is not None else None

        # fused_rank is 1-indexed best-first within each reference (lower = better); 0 = not in
        # forward top-K (reverse-only). A real forward-blocking cutoff must exclude the 0 sentinel.
        thresholds = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        print(f'{"K":>5} {"kept_pairs":>12} {"avg_cand/entity":>16} {"true_kept":>10} {"recall":>8}')
        for k in thresholds:
            keep = (fused_rank >= 1) & (fused_rank <= k)
            if rev_rank is not None:
                keep = keep | ((rev_rank >= 1) & (rev_rank <= k))
            kept_pairs = int(keep.sum())
            true_kept = int((keep & (y == 1)).sum())
            recall = true_kept / total_true if total_true else 0.0
            avg_per_entity = kept_pairs / n_refs
            print(f'{k:>5} {kept_pairs:>12} {avg_per_entity:>16.3f} {true_kept:>10} {recall:>8.5f}', flush=True)


    if __name__ == '__main__':
        main()



def cmd_blocking_recall():
    """Forward candidate recall for several route combinations of one index, on fixed references.

Candidates are generated before labels are read; labels only score recall. Used to decide
whether new routes (phonetic, combined) are worth a full retrain.
"""
    import argparse
    import json
    import random
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    from posting_index import CountryIndex, load_manifest, records
    from submission_pipeline import candidates, read_gold

    CONFIGS = {'v2_routes': ('name', 'address', 'anchor'),
               'plus_phonetic': ('name', 'address', 'anchor', 'phonetic'),
               'plus_combined': ('name', 'address', 'anchor', 'combined'),
               'all_routes': ('name', 'address', 'anchor', 'phonetic', 'combined')}


    def main():
        p = argparse.ArgumentParser(description=cmd_blocking_recall.__doc__)
        p.add_argument('--data', type=Path, required=True)
        p.add_argument('--index', type=Path, required=True)
        p.add_argument('--gold', type=Path, required=True)
        p.add_argument('--reference-ids', type=Path, required=True)
        p.add_argument('--sample', type=int, default=20000)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--top-k', type=int, default=200)
        p.add_argument('--top-terms', type=int, default=64)
        p.add_argument('--batch', type=int, default=4096)
        p.add_argument('--threads', type=int, default=8)
        a = p.parse_args()
        started = time.monotonic()
        ids = sorted(a.reference_ids.read_text().split())
        ids = set(random.Random('blocking-recall').sample(ids, min(a.sample, len(ids))))
        refs = [r for r in records(a.data / 'train_source1.tsv.gz') if r['entity_id'] in ids]
        manifest = load_manifest(a.index)
        found = {c: defaultdict(list) for c in CONFIGS}  # config -> ref -> [(fused_rank, target)]
        timing = defaultdict(float)
        by_country = defaultdict(list)
        for r in refs:
            by_country[r['country']].append(r)
        for country, rows in sorted(by_country.items()):
            index = CountryIndex(a.index, country, manifest)
            for start in range(0, len(rows), a.batch):
                batch = rows[start:start + a.batch]
                for config, routes in CONFIGS.items():
                    t = time.monotonic()
                    c = candidates(index, batch, a.top_k, a.top_terms, 200, a.threads, routes=routes)
                    timing[config] += time.monotonic() - t
                    ids_ = index.ids[c['ord']]
                    for i, rank, tid in zip(c['ref'], c['fused_rank'], ids_):
                        found[config][batch[i]['entity_id']].append((int(rank), tid.decode()))
                print(f'{country}: {start + len(batch):,}/{len(rows):,}', flush=True)
            del index
        gold = read_gold(a.gold, {r['entity_id'] for r in refs})  # Labels read only now.
        country_of = {r['entity_id']: r['country'] for r in refs}
        report = {'references': len(refs), 'gold_links': sum(map(len, gold.values())), 'configs': {},
                  'seconds': time.monotonic() - started, 'timing': dict(timing)}
        for config in CONFIGS:
            out = {}
            for group in ['all'] + sorted(by_country):
                subset = [s for s in gold if group == 'all' or country_of[s] == group]
                total = sum(len(gold[s]) for s in subset)
                row = {}
                for k in (50, 100, 200):
                    hit = sum(len({t for rank, t in found[config][s] if rank <= k} & gold[s]) for s in subset)
                    row[f'recall_at_{k}'] = hit / total if total else None
                out[group] = row
            report['configs'][config] = out
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


    if __name__ == '__main__':
        main()



def cmd_export_previous_samples():
    """Writes the earlier diagnostic reference samples, regenerated from their seeds, as ID lists."""
    import argparse
    from pathlib import Path

    from retrieval_experiment import sample_references
    from blocking_v2 import select


    def previous_samples(source1):
        """retrieval_v1 (2,000), then blocking_v2 / sparse_fullpool (400), then selective_index_v1 (400)."""
        retrieval_v1 = [r['entity_id'] for r in sample_references(source1, per_country=1000, seed=2026)]
        sparse = [r['entity_id'] for r in select(source1, set(retrieval_v1), 200)]
        selective = [r['entity_id'] for r in select(source1, set(retrieval_v1) | set(sparse), 200)]
        return {'retrieval_v1': retrieval_v1, 'sparse_fullpool': sparse, 'selective_index_v1': selective}


    def main():
        p = argparse.ArgumentParser(description=cmd_export_previous_samples.__doc__)
        p.add_argument('--source1', type=Path, required=True, help='Preprocessed train_source1.tsv.gz')
        p.add_argument('--output', type=Path, required=True, help='Directory for <sample>_ids.txt files')
        a = p.parse_args()
        a.output.mkdir(parents=True, exist_ok=True)
        for name, ids in previous_samples(a.source1).items():
            (a.output / f'{name}_ids.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
            print(f'{name}: {len(ids):,} IDs', flush=True)


    if __name__ == '__main__':
        main()



def cmd_native_loss():
    """Share of India holdout loss (T6) on records whose name is (mostly) in a native Indian script, split by stage."""
    import os, re
    import pandas as pd
    A = os.environ['A']
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl'); bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
    need = set(df.t) | set(bf.t); name = {}
    for k in (2, 3):
        f = open(f'/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n = h.index('entity_id'), h.index('business_name')
        for l in f:
            q = l.rstrip('\n').split('\t')
            if q[ie] in need: name[q[ie]] = q[n]
    def native(s):
        letters = [c for c in s if c.isalpha()]
        return bool(letters) and sum(ord(c) > 0x0900 for c in letters) / len(letters) > 0.5
    df['native'] = df.t.map(lambda t: native(name.get(t, ''))); bf['native'] = bf.t.map(lambda t: native(name.get(t, '')))
    for p in ('dev', 'holdout'):
        d = df[(df.part == p) & (df.country == 'India')]; b = bf[(bf.part == p) & (bf.country == 'India')] if 'country' in bf else bf[bf.part == p]
        nS = d.s1.nunique()
        print(f'{p} India: pairs {len(d)}, native-script share of pairs {d.native.mean():.3f}')
        for nat in (True, False):
            e = d[d.native == nat]
            print(f'   native={nat}: FN loss pts {100 * e[e.kind == "FN"].loss.sum() / nS:.4f}  FP loss pts {100 * e[e.kind == "FP"].loss.sum() / nS:.4f}  '
                  f'blocking loss pts {100 * b[b.native == nat].loss.sum() / nS:.4f}  | accepted true-rate {e[e.D == 1].y.mean():.4f}  rejected true-rate {e[e.D == 0].y.mean():.4f}')



def cmd_prepare_saved_pairs():
    """Prepare labeled development pairs from saved retrieval; never inject missing gold."""
    import argparse
    import csv
    import gzip
    import json
    import os
    import time
    from pathlib import Path
    from diagnose_retrieval import read_saved
    from retrieval_experiment import records


    def main():
        p = argparse.ArgumentParser(description=cmd_prepare_saved_pairs.__doc__)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--targets', type=Path, required=True)
        p.add_argument('--ground-truth', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        args = p.parse_args()
        start = time.monotonic()
        refs = json.loads((args.run/'references.json').read_text(encoding='utf-8'))
        saved = read_saved(args.run/'candidates.jsonl', refs)
        lookup = {r['entity_id']: r for r in refs}
        selected = {r['entity_id'] for r in saved}
        wanted = set()
        for result in saved:
            wanted.update(set().union(*map(set, result['channels'].values())))
        gold = {}
        owners = {}
        shared = 0
        with args.ground_truth.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                sid = row['source1_entity_id']
                if sid in selected:
                    gold[sid] = set(filter(None, row['matched_entity_ids'].split(',')))
                    for tid in gold[sid]:
                        if tid in owners and owners[tid] != sid:
                            shared += 1
                        owners[tid] = sid
        if set(gold) != selected:
            raise ValueError('Missing labels for selected references')
        targets = {}
        for source in [2, 3]:
            path = args.targets/f'train_source{source}.tsv.gz'
            for count, row in enumerate(records(path), 1):
                if row['entity_id'] in wanted:
                    if row['entity_id'] in targets:
                        raise ValueError('Duplicate target ID')
                    targets[row['entity_id']] = row
                if count % 1000000 == 0:
                    print(f'Scanned source {source}: {count:,}; retained {len(targets):,}', flush=True)
        if set(targets) != wanted:
            raise ValueError(f'{len(wanted-set(targets))} candidate IDs missing from targets')
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output/'pairs.jsonl.gz'
        if destination.exists():
            raise FileExistsError('Pair output already exists; inspect it instead of overwriting')
        partial = args.output/'pairs.jsonl.gz.partial'
        total = positives = 0
        missing = []
        with gzip.open(partial, 'wt', encoding='utf-8', compresslevel=1) as f:
            for result in saved:
                sid = result['entity_id']
                union = set().union(*map(set, result['channels'].values()))
                ranks = {channel: {tid: i for i, tid in enumerate(ids, 1)}
                         for channel, ids in result['channels'].items()}
                final_ranks = {tid: i for i, tid in enumerate(result['ranked'], 1)}
                for tid in sorted(union):
                    label = int(tid in gold[sid])
                    item = {'source1_entity_id': sid, 'target_entity_id': tid,
                        'reference': lookup[sid], 'target': targets[tid], 'label': label,
                        'retrieval_ranks': {k: v.get(tid) for k, v in ranks.items()},
                        'fused_top50_rank': final_ranks.get(tid)}
                    f.write(json.dumps(item, ensure_ascii=False)+'\n')
                    total += 1
                    positives += label
                missing.append({'source1_entity_id': sid, 'gold_ids': sorted(gold[sid]),
                                'unretrieved_gold_ids': sorted(gold[sid]-union)})
        os.replace(partial, destination)
        (args.output/'reference_truth.json').write_text(json.dumps(missing, indent=2), encoding='utf-8')
        report = {'references': len(saved), 'unique_candidate_targets': len(targets),
                  'pairs': total, 'positives': positives, 'negatives': total-positives,
                  'gold_links': sum(map(len, gold.values())),
                  'missed_gold_links': sum(len(r['unretrieved_gold_ids']) for r in missing),
                  'shared_gold_target_occurrences_within_sample': shared,
                  'seconds': time.monotonic()-start,
                  'purpose': 'Development data only. No train/validation split or model trained. Keep reference/entity groups separated; do not randomly split pairs.'}
        (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


    if __name__ == '__main__':
        main()



def cmd_script_recall():
    """Blocking recall by script: does Tamil / Telugu / Malayalam / ... text lose true links at the candidate stage?

Every gold link of the 100k sample references is put in one bucket: stage-1 top 10, ranks 11-20, retrieved but beyond
20, never retrieved. Links are grouped by the scripts of the two business names (reference script -> record script,
from the first letter outside ASCII; LATIN otherwise). Also counts scripts of test records per country.
Writes artifacts/gap_diag/script_recall.json.
"""
    import csv
    import gzip
    import json
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    csv.field_size_limit(10 ** 9)


    def script(s):
        for ch in s:
            if ord(ch) > 127 and ch.isalpha():
                return unicodedata.name(ch, 'OTHER').split(' ')[0]
        return 'LATIN'


    def names(split, keep=None):
        out = {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if keep is None or row['entity_id'] in keep:
                        out[row['entity_id']] = (script(row['business_name']), row['country'])
        return out


    def main():
        gold = json.loads((ART / 'matcher_v4/pairs/gold.json').read_text())
        m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
        m20 = np.load(ART / 'prune_v1/mask_top20.npy').astype(bool)
        bucket = {}
        with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                r, t = line.rstrip('\n').split('\t')
                bucket[(r, t)] = 'top10' if m10[i] else 'r11_20' if m20[i] else 'beyond20'
        ids = set(gold) | {t for v in gold.values() for t in v}
        nm = names('train', ids)
        by = defaultdict(Counter)
        for r, ts in gold.items():
            for t in ts:
                key = f'{nm[r][1]} {nm[r][0]}->{nm[t][0]}'
                by[key][bucket.get((r, t), 'not_retrieved')] += 1
        R = {}
        for k, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(c.values())
            R[k] = {'links': n, **{b: round(c[b] / n, 5) for b in ('top10', 'r11_20', 'beyond20', 'not_retrieved')}}
        # all-India / all-US totals by "any Indic script involved"
        agg = defaultdict(Counter)
        for k, c in by.items():
            cty, pair = k.split(' ', 1)
            agg[f'{cty} {"latin-latin" if pair == "LATIN->LATIN" else "indic involved"}'].update(c)
        R['_summary'] = {k: {'links': sum(c.values()), 'top10': round(c['top10'] / sum(c.values()), 5),
                             'lost_at_blocking': round(1 - c['top10'] / sum(c.values()), 5)} for k, c in sorted(agg.items())}
        test = Counter()
        for eid, (s, cty) in names('test').items():
            test[f'{cty} {eid[:2]} {s}'] += 1
        R['_test_record_scripts'] = dict(test.most_common())
        (ART / 'gap_diag').mkdir(exist_ok=True)
        (ART / 'gap_diag/script_recall.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1), flush=True)


    if __name__ == '__main__':
        main()



def cmd_sparse_fullpool():
    """Streaming full-pool name/address TF-IDF top-k benchmark for fresh references."""
    import argparse
    import heapq
    import json
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    from retrieval_experiment import records, evaluate
    from blocking_v2 import select


    FIELDS = ('name_folded', 'address_folded')


    def fit_models(paths, countries, fit_rows):
        corpus = {country: [] for country in countries}
        for path in paths:
            for row in records(path):
                country = row['country']
                if country in corpus and len(corpus[country]) < fit_rows:
                    corpus[country].append(row)
            if all(len(corpus[c]) >= fit_rows for c in countries):
                break
        models = {}
        for country in countries:
            if len(corpus[country]) < fit_rows:
                raise ValueError(f'Insufficient fitting rows for {country}')
            models[country] = {}
            for field in FIELDS:
                model = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                    min_df=2, max_features=300000, dtype=np.float32)
                model.fit([r[field] for r in corpus[country]])
                models[country][field] = model
                print(f'Fit {country} {field}: {len(model.vocabulary_):,} features',flush=True)
        return models


    def main():
        p=argparse.ArgumentParser(description=cmd_sparse_fullpool.__doc__)
        p.add_argument('--data',type=Path,required=True)
        p.add_argument('--exclude',type=Path,required=True)
        p.add_argument('--gold',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--per-country',type=int,default=200)
        p.add_argument('--fit-rows',type=int,default=20000)
        p.add_argument('--chunk-rows',type=int,default=50000)
        p.add_argument('--top-k',type=int,default=100)
        p.add_argument('--threads',type=int,default=4)
        p.add_argument('--smoke',action='store_true')
        a=p.parse_args()
        if a.output.exists():
            raise FileExistsError('Choose a new output directory')
        if min(a.per_country,a.fit_rows,a.chunk_rows,a.top_k,a.threads)<1:
            p.error('Counts and threads must be positive')
        paths=[a.data/f'train_source{i}.tsv.gz' for i in [1,2,3]]
        for path in paths:
            manifest=json.loads(Path(str(path)+'.json').read_text())
            if manifest['input']['version']!=2 or (manifest['input']['limit'] and not a.smoke):
                raise ValueError('Full v2 preprocessing inputs required')
            if path.stat().st_size!=manifest['output_bytes']:
                raise ValueError('Input size mismatch')
        start=time.monotonic()
        excluded={r['entity_id'] for r in json.loads(a.exclude.read_text(encoding='utf-8'))}
        bycountry=defaultdict(list)
        for row in select(paths[0],excluded,a.per_country):
            bycountry[row['country']].append(row)
        models=fit_models(paths[1:],list(bycountry),a.fit_rows)
        queries={c:{field:models[c][field].transform([r[field] for r in pool])
                    for field in FIELDS} for c,pool in bycountry.items()}
        fitted=time.monotonic()
        heaps={c:{field:[[] for _ in pool] for field in FIELDS}
               for c,pool in bycountry.items()}
        buffers={c:[] for c in bycountry}
        stats={'target_rows':0,'chunks':0,'target_nnz':defaultdict(int),'multiply_seconds':0.0}
        def process(country):
            batch=buffers[country]
            if not batch:
                return
            stats['chunks']+=1
            for field in FIELDS:
                matrix=models[country][field].transform([r[field] for r in batch])
                stats['target_nnz'][field]+=matrix.nnz
                started=time.monotonic()
                scores=sp_matmul_topn(queries[country][field],matrix.T.tocsr(),
                    top_n=a.top_k,threshold=0.0,sort=False,n_threads=a.threads)
                stats['multiply_seconds']+=time.monotonic()-started
                for i,heap in enumerate(heaps[country][field]):
                    for pos in range(scores.indptr[i],scores.indptr[i+1]):
                        pair=(float(scores.data[pos]),batch[scores.indices[pos]]['entity_id'])
                        if len(heap)<a.top_k:
                            heapq.heappush(heap,pair)
                        elif pair>heap[0]:
                            heapq.heapreplace(heap,pair)
            buffers[country]=[]
        for source in [2,3]:
            for row in records(paths[source-1]):
                stats['target_rows']+=1
                country=row['country']
                if country in buffers:
                    buffers[country].append(row)
                    if len(buffers[country])>=a.chunk_rows:
                        process(country)
                if stats['target_rows']%500000==0:
                    print(f'Processed {stats["target_rows"]:,} targets',flush=True)
        for country in buffers:
            process(country)
        queried=time.monotonic()
        results=[]
        for country,pool in bycountry.items():
            for i,row in enumerate(pool):
                channels={}
                fusion=defaultdict(float)
                for field,label in [('name_folded','name_words'),('address_folded','address_words')]:
                    ranked=[identifier for score,identifier in sorted(heaps[country][field][i],reverse=True)]
                    channels[label]=ranked
                    for rank,identifier in enumerate(ranked,1):
                        fusion[identifier]+=1/(60+rank)
                channels['name_trigrams']=[] # Two-route schema compatibility.
                results.append({'entity_id':row['entity_id'],'country':country,
                    'non_ascii_name':any(ord(char)>127 for char in row['name_norm']),
                    'ranked':sorted(fusion,key=lambda identifier:(-fusion[identifier],identifier))[:50],
                    'channels':channels})
        a.output.mkdir(parents=True)
        (a.output/'references.json').write_text(json.dumps([r for pool in bycountry.values() for r in pool],ensure_ascii=False),encoding='utf-8')
        with (a.output/'candidates.jsonl').open('w',encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
        report=evaluate(results,a.gold) # Labels used only after retrieval completes.
        stats['target_nnz']=dict(stats['target_nnz'])
        report['benchmark']={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'stats':stats,'prepare_and_fit_seconds':fitted-start,
            'stream_and_query_seconds':queried-fitted,'total_seconds':time.monotonic()-start,
            'warning':'Full target pool benchmark on a fresh small reference sample. The fitting corpus is an unlabeled prefix; no validation leakage from labels. Still needs all-reference throughput and France test coverage.'}
        try:
            import resource
            report['benchmark']['peak_rss_kib_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


    if __name__=='__main__':
        main()



def cmd_v5raw_align():
    """Align the raw (unfloored) v5 test scores of the two predict shards to candidate_pairs.tsv expanded order
(= artifacts/submission_T5_own/late_scores.npz order), check them against the floored values already used, and save
  artifacts/v5raw/v5_test_raw.npz  keys: v5_prob (v5 matcher probability, raw), stage1_pruner (retrieval-only top-10 ranker)
  artifacts/v5raw/v5_prob_raw.npy  (the same v5_prob array, float32, 17,325,440 values)
"""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    OUT = ME / 'artifacts/v5raw'


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        refs, tgts = [], []
        with open(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                for t in (c.split(',') if c else []):
                    refs.append(s); tgts.append(t)
        pairs = pd.DataFrame({'r': refs, 't': tgts})
        parts = []
        for k in (0, 1):
            z = np.load(ME / f'artifacts/v5raw_s{k}/top10/scores.npz', allow_pickle=True)
            parts.append(pd.DataFrame({'r': z['ref'].astype(str), 't': z['target'].astype(str),
                                       'p': z['prob'].astype(np.float32), 's1': z['stage1_pruner'].astype(np.float32)}))
        table = pd.concat(parts, ignore_index=True)
        dup = int(table.duplicated(['r', 't']).sum())
        m = pairs.merge(table.drop_duplicates(['r', 't']), on=['r', 't'], how='left')
        missing = int(m['p'].isna().sum())
        v5, s1 = m['p'].to_numpy(np.float32), m['s1'].to_numpy(np.float32)
        Z = np.load(ME / 'artifacts/submission_T5_own/late_scores.npz', allow_pickle=True)
        assert (Z['target'].astype(str) == np.array(tgts)).all(), 'late_scores order differs from candidate_pairs order'
        old = Z['stage1'].astype(np.float32)             # v5 probability floored at 0.02 (0 below), as used by T5
        hi = old > 0
        R = {'pairs': len(pairs), 'missing': missing, 'duplicate_shard_rows': dup,
             'check_vs_floored_ge_0.02_max_abs_diff': float(np.nanmax(np.abs(v5[hi] - old[hi]))) if hi.any() else None,
             'check_floored_zero_rows_raw_below_0.02': float(np.mean(v5[~hi] < 0.02)),
             'raw_quantiles': {q: float(np.nanquantile(v5, q)) for q in (0.1, 0.25, 0.5, 0.75)},
             'share_below': {str(c): float(np.mean(v5 < c)) for c in (0.001, 0.003, 0.01, 0.02)}}
        np.savez(OUT / 'v5_test_raw.npz', v5_prob=v5, stage1_pruner=s1)
        np.save(OUT / 'v5_prob_raw.npy', v5)
        (OUT / 'report.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_web_leet_channel():
    """Two deterministic candidate channels for copies that character retrieval misses (US + India), found by reading the
holdout blocking misses:
  web   web / handle names built from the S1 name: 'aesociety.com' = A(chintya) E(ducational) + Society,
        'btprivate.com' = B(ig) T(echnology) + Private, '#LITTLEAUTO', 'energyprivate.c0m'. S1 keys: concatenations of
        contiguous word runs, initials of the other words + one word, and the anagram of all words; the record key is
        its name without www/TLD/punctuation, leet digits mapped back (0->o, 1->l, 5->s ...).
  leet  names with leet digits inside words ('b1ack constructions', '6URU BENEFIT', 'Ltd 5ai It'): both names are
        leet-normalised and matched on their core word set.
A proposal needs address agreement (a shared number or a shared place word) and a single best S1. MODE=train reports
precision, new candidates outside the top 10 and holdout / dev F0.5 of T6 with the extra links; MODE=test writes the
extra pairs for a submission (records owned by another S1 skipped). Env CHANNEL, MODE, OUT, SUB (test).
"""
    import gzip
    import json
    import os
    import re
    import sys
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from native_channel import addr_keys

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '6': 'g', '8': 'b', '@': 'a', '$': 's'})
    LEGAL = {'llc', 'inc', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'corp', 'corporation', 'llp', 'lp', 'pllc',
             'pc', 'the', 'of', 'and', 'm', 's', 'ms', 'smt', 'sri', 'shri', 'mr', 'dr', 'plc', 'incorporated'}
    SUFFIX = {'limited', 'ltd', 'llc', 'inc', 'co', 'corp', 'company', 'llp', 'lp', 'pllc', 'plc', 'corporation'}
    WEBLIKE = re.compile(r'(\.(com|c0m|co|in|net|org|biz|io)\b|www|^[@#]|\s[@#])', re.I)


    def fold(s):
        s = unicodedata.normalize('NFKD', s or '')
        return ''.join(c for c in s if not unicodedata.combining(c)).lower()


    def words(name, drop_legal=True):
        w = re.sub(r'[^a-z0-9]+', ' ', fold(name)).split()
        return [x for x in w if not (drop_legal and x in LEGAL)]


    def web_keys(name):
        ks = set()
        full = words(name, False)
        tail = list(full)
        while tail and tail[-1] in SUFFIX:
            tail.pop()
        for w in (words(name), full, tail):
            w = [x.translate(LEET) for x in w]
            n = len(w)
            if n == 0:
                continue
            for i in range(n):
                for j in range(i + 1, n + 1):
                    if j - i >= 1:
                        ks.add(''.join(w[i:j]))
                ks.add(''.join(x[0] for k, x in enumerate(w) if k != i) + w[i])
            ks.add(''.join(x[0] for x in w))
            ks.add('A:' + ''.join(sorted(''.join(w))))
        return {k for k in ks if len(k.lstrip('A:')) >= 6}


    def web_record_key(name):
        s = fold(name)
        s = re.sub(r'\.(com|c0m|co|in|net|org|biz|io)\b', ' ', s)
        s = re.sub(r'\bwww\b', ' ', s)
        s = re.sub(r'[^a-z0-9]+', '', s).translate(LEET)
        return {s, 'A:' + ''.join(sorted(s))} if len(s) >= 6 else set()


    def leet_key(name):
        w = [x.translate(LEET) if re.search(r'[a-z]', x) else x for x in words(name)]
        return ' '.join(sorted(set(w))) if w else ''


    def read(split):
        D = H / f'amazon_ml_2026/data/dataset/{split}'
        s1, rec = {}, {}
        for k in (1, 2, 3):
            with open(D / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= c or q[c] not in ('US', 'India'):
                        continue
                    if k == 1:
                        s1[q[ie]] = (q[n], q[a], q[c])
                    else:
                        rec[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c])
        return s1, rec


    def propose(s1, rec, channel):
        index = defaultdict(set)
        ak = {}
        if channel == 'web':
            want = {t: web_record_key(v[0]) for t, v in rec.items() if WEBLIKE.search(v[0]) or (' ' not in v[0].strip() and len(v[0]) >= 8)}
            need = set().union(*want.values()) if want else set()
            for s, (n, a, c) in s1.items():
                for k in web_keys(n) & need:
                    index[(c, k)].add(s)
        else:
            want = {t: {leet_key(v[0])} for t, v in rec.items() if re.search(r'[a-z][0-9]|[0-9][a-z]', fold(v[0])) and leet_key(v[0])}
            need = set().union(*want.values()) if want else set()
            for s, (n, a, c) in s1.items():
                k = leet_key(n)
                if k in need and k != ' '.join(sorted(set(words(n)))) or k in need:
                    index[(c, k)].add(s)
        out = {}
        for t, keys in want.items():
            n, a, c = rec[t]
            cands = set().union(*[index.get((c, k), set()) for k in keys]) if keys else set()
            if not cands:
                continue
            nums, ws = addr_keys(a)
            ok = []
            for s in cands:
                if s not in ak:
                    ak[s] = addr_keys(s1[s][1])
                snums, sws = ak[s]
                if (nums & snums) or (ws & sws):
                    ok.append(s)
            if len(ok) == 1:
                out[t] = (ok[0], 100)
        return out


    def main():
        ch, mode = os.environ['CHANNEL'], os.environ.get('MODE', 'train')
        R = {'channel': ch}
        if mode == 'train':
            from distractor_check import decide, f05, partition_of
            gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
            cp = partition_of(gold)
            s1, rec = read('train')
            prop = propose(s1, rec, ch)
            owner = {t: s for s, ts in gold.items() for t in ts}
            mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
            refs, tgts = [], []
            with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
                for i, line in enumerate(f):
                    if mask[i]:
                        r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
            refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
            P = np.load(V / 'fusion/boundary_models_t6/stage3b_prob.npy')
            for part in ('dev', 'holdout'):
                prefs = sorted(s for s in gold if cp[s] == part)
                pset = set(prefs)
                ix = np.flatnonzero(np.isin(refs, prefs))
                pos = {s: i for i, s in enumerate(prefs)}
                chs = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', 0.74, len(prefs))
                pred, cand = defaultdict(set), defaultdict(set)
                for j, c in zip(ix, chs):
                    cand[refs[j]].add(tgts[j])
                    if c:
                        pred[refs[j]].add(tgts[j])
                po = {t: s for s, ts in pred.items() for t in ts}
                C = Counter(); new = defaultdict(set); ex = []
                for t, (s, _) in prop.items():
                    if s not in pset:
                        continue
                    okk = owner.get(t) == s
                    C['proposed'] += 1; C['proposed_true'] += okk
                    if t in cand[s]:
                        C['already_candidate'] += 1
                        continue
                    C['new_candidate'] += 1; C['new_true'] += okk
                    if t in po:
                        C['new_owned_elsewhere'] += 1
                        continue
                    new[s].add(t)
                    if len(ex) < 15:
                        ex.append(f"{int(okk)} {s1[s][0]} | {s1[s][1][:50]} => {rec[t][0]} | {rec[t][1][:50]}")
                base = np.mean([f05(gold[s], pred.get(s, set())) for s in prefs])
                plus = np.mean([f05(gold[s], pred.get(s, set()) | new.get(s, set())) for s in prefs])
                C['gain'] = round(float(plus - base), 6)
                R[part] = dict(C); R[part + '_examples'] = ex
                print(ch, part, dict(C), flush=True)
        else:
            s1, rec = read('test')
            prop = propose(s1, rec, ch)
            from france_census import load
            match = load(os.environ['SUB'])
            owner = {t: r for r, ts in match.items() for t in ts}
            ec, ea = defaultdict(list), defaultdict(list)
            for t, (s, _) in prop.items():
                ec[s].append(t)
                if t not in owner:
                    ea[s].append(t)
            json.dump({'candidates': ec, 'accepted': ea}, open(os.environ['OUT'] + '.pairs.json', 'w'))
            R.update({'proposed': len(prop), 'accepted_links': sum(len(v) for v in ea.values())})
        open(os.environ['OUT'], 'w').write(json.dumps(R, indent=1, ensure_ascii=False))
        print(json.dumps(R, indent=1, ensure_ascii=False), flush=True)


    if __name__ == '__main__':
        main()



COMMANDS = {'benchmark_capped': cmd_benchmark_capped, 'blocking_cutoff_cost': cmd_blocking_cutoff_cost, 'blocking_recall': cmd_blocking_recall, 'export_previous_samples': cmd_export_previous_samples, 'native_loss': cmd_native_loss, 'prepare_saved_pairs': cmd_prepare_saved_pairs, 'script_recall': cmd_script_recall, 'sparse_fullpool': cmd_sparse_fullpool, 'v5raw_align': cmd_v5raw_align, 'web_leet_channel': cmd_web_leet_channel}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('commands:', ', '.join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    name = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {name}'
    COMMANDS[name]()
