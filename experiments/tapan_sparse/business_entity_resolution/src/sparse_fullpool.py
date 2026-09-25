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
from compact_topk import CompactTopK, encode_target_id


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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--exclude',type=Path,required=True)
    p.add_argument('--exclude-additional',type=Path,action='append',default=[])
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-country',type=int,default=200)
    p.add_argument('--fit-rows',type=int,default=20000)
    p.add_argument('--chunk-rows',type=int,default=50000)
    p.add_argument('--top-k',type=int,default=100)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--compact-topk',action='store_true')
    p.add_argument('--query-batch',type=int,default=5000)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a new output directory')
    if min(a.per_country,a.fit_rows,a.chunk_rows,a.top_k,a.threads,a.query_batch)<1:
        p.error('Counts and threads must be positive')
    paths=[a.data/f'train_source{i}.tsv.gz' for i in [1,2,3]]
    for path in paths:
        manifest=json.loads(Path(str(path)+'.json').read_text())
        if manifest['input']['version']!=2 or (manifest['input']['limit'] and not a.smoke):
            raise ValueError('Full v2 preprocessing inputs required')
        if path.stat().st_size!=manifest['output_bytes']:
            raise ValueError('Input size mismatch')
    start=time.monotonic()
    excluded=set()
    for path in [a.exclude,*a.exclude_additional]:
        excluded.update(r['entity_id'] for r in json.loads(path.read_text(encoding='utf-8')))
    bycountry=defaultdict(list)
    for row in select(paths[0],excluded,a.per_country):
        bycountry[row['country']].append(row)
    print('Selected references:', {c:len(pool) for c,pool in bycountry.items()},flush=True)
    models=fit_models(paths[1:],list(bycountry),a.fit_rows)
    queries={c:{field:models[c][field].transform([r[field] for r in pool])
                for field in FIELDS} for c,pool in bycountry.items()}
    fitted=time.monotonic()
    heaps=({c:{field:CompactTopK(len(pool),a.top_k) for field in FIELDS}
           for c,pool in bycountry.items()} if a.compact_topk else
           {c:{field:[[] for _ in pool] for field in FIELDS}
           for c,pool in bycountry.items()})
    buffers={c:[] for c in bycountry}
    stats={'target_rows':0,'chunks':0,'target_nnz':defaultdict(int),
           'multiply_seconds':0.0,'merge_seconds':0.0}
    def process(country):
        batch=buffers[country]
        if not batch:
            return
        stats['chunks']+=1
        target_ids=(np.asarray([encode_target_id(row['entity_id']) for row in batch],
                               dtype=np.uint64) if a.compact_topk else None)
        for field in FIELDS:
            matrix=models[country][field].transform([r[field] for r in batch])
            stats['target_nnz'][field]+=matrix.nnz
            transposed=matrix.T.tocsr()
            if a.compact_topk:
                query=queries[country][field]
                for lo in range(0,query.shape[0],a.query_batch):
                    started=time.monotonic()
                    scores=sp_matmul_topn(query[lo:lo+a.query_batch],transposed,
                        top_n=a.top_k,threshold=0.0,sort=False,n_threads=a.threads)
                    stats['multiply_seconds']+=time.monotonic()-started
                    started=time.monotonic()
                    heaps[country][field].update(lo,scores,target_ids)
                    stats['merge_seconds']+=time.monotonic()-started
            else:
                started=time.monotonic()
                scores=sp_matmul_topn(queries[country][field],transposed,
                    top_n=a.top_k,threshold=0.0,sort=False,n_threads=a.threads)
                stats['multiply_seconds']+=time.monotonic()-started
                started=time.monotonic()
                for i,heap in enumerate(heaps[country][field]):
                    for pos in range(scores.indptr[i],scores.indptr[i+1]):
                        pair=(float(scores.data[pos]),batch[scores.indices[pos]]['entity_id'])
                        if len(heap)<a.top_k:
                            heapq.heappush(heap,pair)
                        elif pair>heap[0]:
                            heapq.heapreplace(heap,pair)
                stats['merge_seconds']+=time.monotonic()-started
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
                print(f'Processed {stats["target_rows"]:,} targets; '
                      f'elapsed {time.monotonic()-start:.1f}s; '
                      f'multiply {stats["multiply_seconds"]:.1f}s',flush=True)
    for country in buffers:
        process(country)
    queried=time.monotonic()
    results=[]
    for country,pool in bycountry.items():
        for i,row in enumerate(pool):
            channels={}
            fusion=defaultdict(float)
            for field,label in [('name_folded','name_words'),('address_folded','address_words')]:
                ranked=([identifier for score,identifier in heaps[country][field].ranked(i)]
                        if a.compact_topk else
                        [identifier for score,identifier in sorted(heaps[country][field][i],reverse=True)])
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
    parameters={k:([str(item) for item in v] if isinstance(v,list) else
                   str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    report['benchmark']={'parameters':parameters,
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
