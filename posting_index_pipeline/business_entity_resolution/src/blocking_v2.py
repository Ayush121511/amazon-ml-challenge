"""Sample-focused inverted blocking benchmark. Bounded postings; no BM25 query scans."""
import argparse
from array import array
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import time
from retrieval_experiment import records, evaluate


def keys(row):
    country = row['country']
    name = row['name_folded']
    address = row['address_folded']
    result = set()
    for token in name.split():
        if len(token) >= 3:
            result.add((country, 'name_words', token))
    for token in address.split():
        if len(token) >= 3:
            result.add((country, 'address_words', token))
    # Four-character pieces within words avoid common three-character fragments.
    for word in name.split():
        for i in range(len(word)-3):
            result.add((country, 'name_trigrams', word[i:i+4]))
    return result


def select(path, excluded, n):
    pools, counts, rngs = {}, Counter(), {}
    for row in records(path):
        if row['entity_id'] in excluded:
            continue
        country = row['country']
        pools.setdefault(country, [])
        rngs.setdefault(country, random.Random('blocking-v2-2028:'+country))
        counts[country] += 1
        if len(pools[country]) < n:
            pools[country].append(row)
        else:
            j = rngs[country].randrange(counts[country])
            if j < n:
                pools[country][j] = row
    return sorted([r for pool in pools.values() for r in pool], key=lambda r:r['entity_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--exclude', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-country', type=int, default=200)
    p.add_argument('--max-postings', type=int, default=200)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.per_country < 1 or args.max_postings < 1:
        p.error('Sample size and posting cap must be positive')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    for i in [1,2,3]:
        path = args.data/f'train_source{i}.tsv.gz'
        manifest = json.loads(Path(str(path)+'.json').read_text())
        if manifest['input']['version'] != 2 or (manifest['input']['limit'] and not args.smoke):
            raise ValueError('Full v2 preprocessing required')
        if path.stat().st_size != manifest['output_bytes']:
            raise ValueError('Input size mismatch')
    started = time.monotonic()
    excluded = {r['entity_id'] for r in json.loads(args.exclude.read_text(encoding='utf-8'))}
    refs = select(args.data/'train_source1.tsv.gz', excluded, args.per_country)
    if not refs:
        raise ValueError('No references after exclusions')
    args.output.mkdir(parents=True)
    (args.output/'references.json').write_text(json.dumps(refs,ensure_ascii=False),encoding='utf-8')
    wanted = set().union(*(keys(r) for r in refs))
    postings = {key: array('I') for key in wanted}
    counts = Counter()
    countries = Counter()
    ordinal = 0
    for source in [2,3]:
        for row in records(args.data/f'train_source{source}.tsv.gz'):
            countries[row['country']] += 1
            for key in keys(row) & wanted:
                counts[key] += 1
                if counts[key] <= args.max_postings:
                    postings[key].append(ordinal)
                elif counts[key] == args.max_postings+1:
                    # Drop the ENTIRE oversized block, never retain a biased prefix.
                    postings[key] = array('I')
            ordinal += 1
            if ordinal % 500000 == 0:
                print(f'Index scan: {ordinal:,} target records',flush=True)
    indexed = time.monotonic()
    results = []
    needed = set()
    for row in refs:
        channel_scores = defaultdict(lambda: defaultdict(float))
        for key in keys(row):
            if 0 < counts[key] <= args.max_postings:
                weight = math.log1p(countries[row['country']]/counts[key])
                for target in postings[key]:
                    channel_scores[key[1]][target] += weight
        channels = {}
        fusion = defaultdict(float)
        for channel in ['name_words','address_words','name_trigrams']:
            scores = channel_scores[channel]
            ranked = sorted(scores,key=lambda x:(-scores[x],x))[:50]
            channels[channel] = ranked
            needed.update(ranked)
            for rank, target in enumerate(ranked,1):
                fusion[target] += 1/(60+rank)
        results.append({'entity_id':row['entity_id'],'country':row['country'],
            'non_ascii_name':any(ord(c)>127 for c in row['name_norm']),
            'channels':channels,'ranked':sorted(fusion,key=lambda x:(-fusion[x],x))[:50]})
    queried = time.monotonic()
    # Second sequential pass maps only shortlisted ordinals to original IDs.
    mapping = {}
    ordinal = 0
    for source in [2,3]:
        for row in records(args.data/f'train_source{source}.tsv.gz'):
            if ordinal in needed:
                mapping[ordinal] = row['entity_id']
            ordinal += 1
    for result in results:
        result['ranked'] = [mapping[i] for i in result['ranked']]
        result['channels'] = {k:[mapping[i] for i in v] for k,v in result['channels'].items()}
    with (args.output/'candidates.jsonl').open('w',encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
    report = evaluate(results,args.gold) # First label access, after candidate generation.
    report['coverage'] = {'empty_shortlists':sum(not r['ranked'] for r in results),
        'mean_union_candidates':sum(len(set().union(*map(set,r['channels'].values()))) for r in results)/len(results)}
    report['benchmark'] = {'smoke':args.smoke,'target_records':ordinal,'query_keys':len(wanted),
        'oversized_blocks_dropped':sum(counts[k]>args.max_postings for k in wanted),
        'max_postings':args.max_postings,'index_and_sample_seconds':indexed-started,
        'query_seconds':queried-indexed,'mapping_and_evaluation_seconds':time.monotonic()-queried,
        'note':'Fresh development sample excludes prior 2000 reference IDs. Sample-focused index, two full target scans. Query time alone is NOT full-pipeline throughput. name_trigrams field contains 4-grams for schema compatibility.'}
    try:
        import resource
        report['benchmark']['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
