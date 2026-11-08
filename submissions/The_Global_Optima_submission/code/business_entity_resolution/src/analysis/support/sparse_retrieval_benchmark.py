"""Bounded bulk TF-IDF retrieval benchmark; sampled-target recall is conditional."""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from preprocess import transform


def stream_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            yield row


def select_references(path, excluded, per_country):
    pools, counts, rngs = {}, Counter(), {}
    for row in stream_tsv(path):
        if row['entity_id'] in excluded:
            continue
        country = row['country']
        pools.setdefault(country, [])
        rngs.setdefault(country, random.Random('sparse-benchmark-2030:'+country))
        counts[country] += 1
        if len(pools[country]) < per_country:
            pools[country].append(transform(row))
        else:
            j = rngs[country].randrange(counts[country])
            if j < per_country:
                pools[country][j] = transform(row)
    return pools


def collect_targets(paths, per_source_country):
    targets = {}
    for source, path in enumerate(paths, 2):
        counts = Counter()
        for row in stream_tsv(path):
            country = row['country']
            if counts[country] >= per_source_country:
                continue
            targets.setdefault(country, []).append(transform(row))
            counts[country] += 1
            if len(counts) >= 2 and all(v >= per_source_country for v in counts.values()):
                break
        print(f'Source {source} target slice: {dict(counts)}', flush=True)
    return targets


def run_channel(references, targets, field, top_n, threads):
    start = time.monotonic()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
        min_df=2, max_features=400000, dtype=np.float32, norm='l2')
    target_matrix = vectorizer.fit_transform([r[field] for r in targets])
    reference_matrix = vectorizer.transform([r[field] for r in references])
    built = time.monotonic()
    matches = sp_matmul_topn(reference_matrix, target_matrix.T.tocsr(),
        top_n=top_n, threshold=0.0, sort=True, n_threads=threads)
    candidates = []
    for i in range(matches.shape[0]):
        candidates.append([(targets[j]['entity_id'],float(score)) for j,score in zip(
            matches.indices[matches.indptr[i]:matches.indptr[i+1]],
            matches.data[matches.indptr[i]:matches.indptr[i+1]])])
    return candidates, {'target_rows':target_matrix.shape[0],
        'vocabulary':len(vectorizer.vocabulary_),'target_nnz':target_matrix.nnz,
        'query_nnz':reference_matrix.nnz,
        'build_seconds':built-start,'multiply_seconds':time.monotonic()-built}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--exclude',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-country',type=int,default=100)
    p.add_argument('--per-source-country',type=int,default=10000)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    if args.per_country<1 or args.per_source_country<1 or args.threads<1:
        p.error('Counts and threads must be positive')
    args.output.mkdir(parents=True)
    clock=time.monotonic()
    excluded={r['entity_id'] for r in json.loads(args.exclude.read_text(encoding='utf-8'))}
    refs=select_references(args.data/'train/train_source1.tsv',excluded,args.per_country)
    targets=collect_targets([args.data/f'train/train_source{i}.tsv' for i in (2,3)],args.per_source_country)
    prepared=time.monotonic()
    result={'parameters':vars(args).copy(),'countries':{},
      'warning':'Targets are a source-file prefix sample. This measures bounded performance, NOT full-pool recall or throughput. Conditional recall includes only gold IDs present in sampled targets.'}
    result['parameters']={k:str(v) if isinstance(v,Path) else v for k,v in result['parameters'].items()}
    gold={}
    selected={r['entity_id'] for pool in refs.values() for r in pool}
    for row in stream_tsv(args.gold):
        if row['source1_entity_id'] in selected:
            gold[row['source1_entity_id']]=set(filter(None,row['matched_entity_ids'].split(',')))
    for country,pool in refs.items():
        target_pool=targets[country]
        target_ids={r['entity_id'] for r in target_pool}
        channels={}
        stats={}
        for field in ('name_folded','address_folded'):
            channels[field],stats[field]=run_channel(pool,target_pool,field,50,args.threads)
            print(country,field,stats[field],flush=True)
        covered=found=0
        for i,ref in enumerate(pool):
            truth=gold[ref['entity_id']] & target_ids
            covered+=len(truth)
            union={identifier for field in channels for identifier,_ in channels[field][i]}
            found+=len(truth & union)
        result['countries'][country]={'references':len(pool),'targets':len(target_pool),
          'gold_links_present_in_target_slice':covered,'gold_links_retrieved_in_union':found,
          'conditional_recall':found/covered if covered else None,
          'channels':stats,'rss_bytes_after_country':psutil.Process().memory_info().rss}
        (args.output/'report.json').write_text(json.dumps(result,indent=2))
    result['prepare_seconds']=prepared-clock
    result['total_seconds']=time.monotonic()-clock
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
