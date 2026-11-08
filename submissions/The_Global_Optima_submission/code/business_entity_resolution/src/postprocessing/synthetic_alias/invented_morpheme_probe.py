"""Check a fixed synthetic-syllable grammar on train and remaining France links."""

import csv
import json
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT=Path('artifacts/france_full_audit_20260927/u5_report')
MORPHEMES='arc aria avi belo brix calo cira delta dova drex ecto evo faye flux gild halo iri jax kelo kor lum lyra mira nex novi nyla onyx orbi pyra quo riza sol syn tavo umbra vantage veo vera vio wex xylo yuma zeta zeph'.split()

@lru_cache(None)
def parse(name):
    """Return the count of fixed syllables if the entire one-token name tiles."""
    name=re.sub('[^a-z]','',name.lower())
    if not name:return 0
    state={0:0}
    for i in range(len(name)):
        if i not in state:continue
        for m in MORPHEMES:
            if name.startswith(m,i):state[i+len(m)]=max(state.get(i+len(m),0),state[i]+1)
    return state.get(len(name),0)

def main():
    train=Counter();examples=[]
    with (ROOT/'train_unique_short_names.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if len(r['target_name'].split())!=1:continue
            n=parse(r['target_name']);bucket='morpheme_2plus' if n>=2 else 'other'
            train[(bucket,r['is_true'])]+=1
    test=Counter();chosen=[]
    with (ROOT/'unique_address_novel_names.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if len(r['target_name'].split())!=1:continue
            n=parse(r['target_name']);bucket='morpheme_2plus' if n>=2 else 'other'
            test[bucket]+=1
            if n>=2:chosen.append(r)
    print('TRAIN',dict(train),'TEST',dict(test),flush=True)
    print('TEST EXAMPLES',[(r['ref_name'],r['target_name']) for r in chosen[:30]],flush=True)
    with (ROOT/'morpheme_remaining.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=chosen[0].keys() if chosen else [],delimiter='\t')
        w.writeheader();w.writerows(chosen)
    # Check coverage on the team's U9 invented1 positives.
    ids=[]
    with Path('artifacts/U9_rule_channel_links.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['pattern']=='invented1':ids.append(r['entity_id'])
    conn=sqlite3.connect('artifacts/france_full_audit_20260927/indexed/france_test.sqlite3')
    covered=0;total=0
    for i in range(0,len(ids),500):
        chunk=ids[i:i+500];marks=','.join('?' for _ in chunk)
        for (name,) in conn.execute(f'SELECT name FROM records WHERE entity_id IN ({marks})',chunk):
            total+=1;covered+=parse(name)>=2
    print('U9 invented1 coverage',covered,total,flush=True)
    (ROOT/'morpheme_summary.json').write_text(json.dumps({'train':{str(k):v for k,v in train.items()},'test':dict(test),'u9_coverage':[covered,total]},indent=2),encoding='utf-8')

if __name__=='__main__':main()
