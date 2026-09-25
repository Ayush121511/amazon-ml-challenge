"""Fresh-sample recall, end-to-end throughput and miss analysis for posting_index.

Candidates are generated and written for every reference BEFORE ground truth is
opened. Labels are used only for evaluation and the post-hoc miss audit.
"""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import socket
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from retrieval_experiment import records, sample_references
from blocking_v2 import select
from posting_index import CountryIndex, load_manifest, search, fuse, ROUTES

SAMPLE_SEED = 'posting-index-v1'


# ------------------------------------------------------------- sampling ----

def previous_diagnostic_ids(source1):
    """Regenerates every earlier diagnostic sample from its fixed seed."""
    retrieval_v1 = sample_references(source1, per_country=1000, seed=2026)
    excluded = {r['entity_id'] for r in retrieval_v1}
    sparse_and_blocking = select(source1, excluded, 200)
    excluded |= {r['entity_id'] for r in sparse_and_blocking}
    selective = select(source1, excluded, 200)
    sizes = {'retrieval_v1': len(retrieval_v1), 'blocking_v2_and_sparse_fullpool': len(sparse_and_blocking),
             'selective_index_v1': len(selective)}
    return excluded | {r['entity_id'] for r in selective}, sizes


def fresh_sample(source1, excluded, per_country):
    pools, counts, rngs = {}, Counter(), {}
    for row in records(source1):
        if row['entity_id'] in excluded:
            continue
        country = row['country']
        if country not in pools:
            pools[country] = []
            rngs[country] = random.Random(f'{SAMPLE_SEED}:{country}')
        counts[country] += 1
        pool = pools[country]
        if len(pool) < per_country:
            pool.append(row)
        else:
            j = rngs[country].randrange(counts[country])
            if j < per_country:
                pool[j] = row
    return sorted((r for p in pools.values() for r in p), key=lambda r: r['entity_id'])


def listed_references(source1, ids):
    """Exactly the references named in ids, in entity_id order; every ID must exist."""
    refs = [row for row in records(source1) if row['entity_id'] in ids]
    missing = ids - {r['entity_id'] for r in refs}
    if missing or len(refs) != len(ids):
        raise ValueError(f'{len(missing)} listed reference IDs not found in {source1}')
    return sorted(refs, key=lambda r: r['entity_id'])


def first_references(source1, limit):
    refs = []
    for row in records(source1):
        refs.append(row)
        if limit and len(refs) >= limit:
            break
    return refs


# ------------------------------------------------------------- analysis ----

def scripts(text):
    found = set()
    for char in text:
        if char.isalpha():
            found.add(unicodedata.name(char, 'UNKNOWN').split()[0])
    return found


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0


def numbers(text):
    return {t.lstrip('0') or '0' for t in text.split() if t.isdigit()}


def miss_reasons(ref, target):
    """Label-free descriptive features of a (reference, missed target) pair."""
    ref_scripts, target_scripts = scripts(ref['name_folded']), scripts(target['name_folded'])
    non_ascii = any(ord(c) > 127 for c in ref['business_name'] + target['business_name'])
    reasons = {
        'name_script_mismatch': bool(ref_scripts and target_scripts and not (ref_scripts & target_scripts)),
        'non_ascii_same_script': non_ascii and bool(ref_scripts & target_scripts),
        'target_address_missing': not target['address_folded'].strip(),
        'web_domain_name': any(x in target['business_name'].casefold() for x in ('.com', 'www', '.in', '.net', '.org')),
        'shares_address_number': bool(numbers(ref['address_folded']) & numbers(target['address_folded'])),
        'name_token_jaccard_ge_0_5': jaccard(ref['name_folded'], target['name_folded']) >= 0.5,
    }
    for reason in ('name_script_mismatch', 'target_address_missing', 'web_domain_name'):
        if reasons[reason]:
            primary = reason
            break
    else:
        if reasons['name_token_jaccard_ge_0_5']:
            primary = 'similar_name_crowded_out'
        elif reasons['shares_address_number']:
            primary = 'address_anchor_only'
        else:
            primary = 'low_text_similarity'
    return reasons, primary


def load_gold(path, wanted):
    gold = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['source1_entity_id'] in wanted:
                gold[row['source1_entity_id']] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(gold) != wanted:
        raise ValueError('Missing ground truth for sampled references')
    return gold


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {'p10': pick(0.1), 'median': pick(0.5), 'p90': pick(0.9), 'mean': statistics.fmean(values)}


def evaluate(results, gold):
    report = {}
    groups = ['all'] + sorted({r['country'] for r in results}) + ['non_ascii_name']
    for group in groups:
        subset = [r for r in results if group == 'all' or r['country'] == group
                  or (group == 'non_ascii_name' and r['non_ascii_name'])]
        positive = [r for r in subset if gold[r['entity_id']]]
        singletons = [r for r in subset if not gold[r['entity_id']]]
        total = sum(len(gold[r['entity_id']]) for r in positive)
        m = {'references': len(subset), 'positive_references': len(positive),
             'singletons': len(singletons), 'gold_links': total}
        for k in (10, 20, 50):
            found = [len(set(r['ranked'][:k]) & gold[r['entity_id']]) for r in positive]
            m[f'recall_at_{k}'] = sum(found) / total if total else None
            m[f'macro_recall_at_{k}'] = (sum(n / len(gold[r['entity_id']]) for n, r in zip(found, positive))
                                         / len(positive)) if positive else None
        for route in ROUTES:
            m[f'{route}_recall_at_route_k'] = (sum(len({i for i, _ in r['channels'][route]} & gold[r['entity_id']])
                                                   for r in positive) / total) if total else None
        m['union_recall'] = (sum(len(union(r) & gold[r['entity_id']]) for r in positive) / total) if total else None
        m['mean_union_candidates'] = statistics.fmean(len(union(r)) for r in subset) if subset else None
        m['empty_shortlists'] = sum(not r['ranked'] for r in subset)
        m['top1_score_by_route'] = {
            kind: {route: quantiles([r['channels'][route][0][1] for r in rows if r['channels'][route]])
                   for route in ROUTES}
            for kind, rows in (('positive', positive), ('singleton', singletons))}
        report[group] = m
    return report


def union(result):
    return {i for hits in result['channels'].values() for i, _ in hits}


def miss_audit(results, gold, refs_by_id, index_root, manifest, output):
    misses = []
    for r in results:
        truth = gold[r['entity_id']]
        top50, pool = set(r['ranked']), union(r)
        for target in sorted(truth - top50):
            misses.append({'reference_id': r['entity_id'], 'target_id': target, 'country': r['country'],
                           'stage': 'lost_at_50' if target in pool else 'absent_from_routes'})
    wanted = defaultdict(set)
    for m in misses:
        wanted[m['country']].add(m['target_id'])
    texts = {}
    for country, ids in wanted.items():
        if country not in manifest['countries']:
            continue
        path = Path(index_root) / manifest['countries'][country]['dir'] / 'targets.tsv.gz'
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row['entity_id'] in ids:
                    texts[row['entity_id']] = row
    summary = {'misses': len(misses), 'by_country_stage': Counter(), 'primary_reason': Counter(),
               'primary_reason_by_country': defaultdict(Counter), 'overlapping_flags': Counter(),
               'target_in_other_country_pool': 0}
    with (output / 'misses.jsonl').open('w', encoding='utf-8') as f:
        for m in misses:
            summary['by_country_stage'][f"{m['country']}:{m['stage']}"] += 1
            target = texts.get(m['target_id'])
            if target is None:
                m['primary_reason'] = 'target_not_in_same_country_pool'
                summary['target_in_other_country_pool'] += 1
            else:
                flags, primary = miss_reasons(refs_by_id[m['reference_id']], target)
                m.update(flags=flags, primary_reason=primary,
                         reference={k: refs_by_id[m['reference_id']][k] for k in
                                    ('business_name', 'business_address', 'name_folded', 'address_folded')},
                         target={k: target[k] for k in
                                 ('business_name', 'business_address', 'name_folded', 'address_folded')})
                summary['overlapping_flags'].update(k for k, v in flags.items() if v)
            summary['primary_reason'][m['primary_reason']] += 1
            summary['primary_reason_by_country'][m['country']][m['primary_reason']] += 1
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    summary['primary_reason_by_country'] = {k: dict(v) for k, v in summary['primary_reason_by_country'].items()}
    for key in ('by_country_stage', 'primary_reason', 'overlapping_flags'):
        summary[key] = dict(summary[key])
    summary['note'] = ('Flags overlap; primary_reason is a priority assignment so it sums to the total. '
                       'name_script_mismatch means the two names share no alphabetic script; '
                       'non_ascii_same_script means non-ASCII characters appear but the scripts overlap.')
    return summary


# ------------------------------------------------------------------ run ----

def peak_rss_kib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss // 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Preprocessed v2 split directory')
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold', type=Path, help='Train ground truth; evaluation only')
    p.add_argument('--per-country', type=int, default=10000, help='Sample size per country')
    p.add_argument('--reference-ids', type=Path,
                   help='Query exactly these Source 1 IDs (one per line) instead of sampling')
    p.add_argument('--first', type=int, default=0,
                   help='Test split only: query the first N references instead of a per-country sample')
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--top-terms', type=int, default=32)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--test-references', type=int, default=1732544,
                   help='Reference count used for the throughput extrapolation')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a new output directory')
    if a.split == 'train' and not a.gold:
        p.error('--gold is required for the train split')
    started = time.monotonic()
    source1 = a.data / f'{a.split}_source1.tsv.gz'
    sample_info = {}
    exclusion_seconds = 0.0
    if a.reference_ids:
        ids = set(filter(None, a.reference_ids.read_text(encoding='utf-8').split()))
        refs = listed_references(source1, ids)
        sample_info = {'reference_ids_file': str(a.reference_ids)}
    elif a.split == 'train':
        excluded, sizes = previous_diagnostic_ids(source1)
        exclusion_seconds = time.monotonic() - started
        refs = fresh_sample(source1, excluded, a.per_country)
        sample_info = {'excluded_previous_ids': len(excluded), 'previous_samples': sizes,
                       'seed': SAMPLE_SEED, 'per_country': a.per_country}
    elif a.first:
        refs = first_references(source1, a.first)
    else:
        refs = fresh_sample(source1, set(), a.per_country)
        sample_info = {'seed': SAMPLE_SEED, 'per_country': a.per_country}
    sampled = time.monotonic()
    a.output.mkdir(parents=True)
    ids_text = '\n'.join(r['entity_id'] for r in refs) + '\n'
    (a.output / 'reference_ids.txt').write_text(ids_text, encoding='utf-8')
    sample_info['reference_ids_sha256'] = hashlib.sha256(ids_text.encode()).hexdigest()
    sample_info['references_by_country'] = dict(Counter(r['country'] for r in refs))

    manifest = load_manifest(a.index)
    by_country = defaultdict(list)
    for r in refs:
        by_country[r['country']].append(r)
    timing = defaultdict(float)
    results = []
    with (a.output / 'candidates.jsonl').open('w', encoding='utf-8') as out:
        for country in sorted(by_country):
            rows = by_country[country]
            if country not in manifest['countries']:
                chunk = [{'entity_id': r['entity_id'], 'country': country, 'ranked': [],
                          'channels': {route: [] for route in ROUTES},
                          'non_ascii_name': any(ord(c) > 127 for c in r['name_norm'])} for r in rows]
                results.extend(chunk)
                continue
            t = time.monotonic()
            index = CountryIndex(a.index, country, manifest)
            timing[f'load_index_{country}'] = time.monotonic() - t
            t = time.monotonic()
            for start in range(0, len(rows), a.batch):
                batch = rows[start:start + a.batch]
                found = search(index, batch, a.top_k, a.top_terms, a.threads)
                for row, channels in zip(batch, found):
                    result = {'entity_id': row['entity_id'], 'country': country,
                              'non_ascii_name': any(ord(c) > 127 for c in row['name_norm']),
                              'ranked': fuse(channels), 'channels': channels}
                    results.append(result)
                    out.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f'{country}: {start + len(batch):,}/{len(rows):,} references', flush=True)
            timing[f'query_{country}'] = time.monotonic() - t
            timing[f'references_{country}'] = len(rows)
            del index
    finished = time.monotonic()

    queried = len(results)
    load_seconds = sum(v for k, v in timing.items() if k.startswith('load_index_'))
    query_seconds = sum(v for k, v in timing.items() if k.startswith('query_'))
    rate = queried / query_seconds if query_seconds else None
    report = {'split': a.split, 'sample': sample_info, 'parameters': {
                  k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'index': {k: manifest[k] for k in ('build_seconds', 'index_bytes', 'max_df_fraction',
                                                 'min_max_df')},
              'environment': {'host': socket.gethostname(), 'python': platform.python_version(),
                              'pbs_job': __import__('os').environ.get('PBS_JOBID'),
                              'cpus': __import__('os').cpu_count()},
              'throughput': {'references': queried,
                             'regenerate_previous_samples_seconds': exclusion_seconds,
                             'reference_loading_seconds': sampled - started - exclusion_seconds,
                             'index_load_seconds': load_seconds, 'query_seconds': query_seconds,
                             'end_to_end_seconds': finished - started,
                             'references_per_second_query': rate,
                             'timing': dict(timing),
                             'peak_rss_kib': peak_rss_kib()}}
    if rate:
        report['throughput']['extrapolated_test_seconds'] = (
            load_seconds + (sampled - started - exclusion_seconds) + a.test_references / rate)
        report['throughput']['extrapolation_note'] = (
            'Index load + one pass over Source 1 as measured here, plus test references / measured '
            'query rate. The test pool (9.97M targets, three countries) needs its own index build; '
            'check its measured rate before relying on this estimate.')
    if a.gold:
        refs_by_id = {r['entity_id']: r for r in refs}
        gold = load_gold(a.gold, set(refs_by_id))  # First label access, after all candidates are saved.
        report['recall'] = evaluate(results, gold)
        report['miss_audit'] = miss_audit(results, gold, refs_by_id, a.index, manifest, a.output)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
