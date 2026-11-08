"""Export unowned France synthetic aliases at unique token-sorted exact address."""

import csv
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

from invented_morpheme_probe import parse

ROOT=Path('artifacts/france_full_audit_20260927')
ART=Path('artifacts')
SUFFIX={'labs','one','sys','co'}

def key(s):
    s=unicodedata.normalize('NFKD',s or '').lower()
    s=''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(sorted(re.findall(r'[a-z0-9]+',s)))

def pattern(name):
    toks=name.lower().split()
    if len(toks)==1 and parse(toks[0])>=2:return 'synthetic_one_token'
    if len(toks)==2 and parse(toks[0])>=2 and toks[1].strip('.,+!') in SUFFIX:return 'synthetic_brand_suffix'
    return ''

def main():
    conn=sqlite3.connect(f"file:{(ROOT/'indexed/france_test.sqlite3').as_posix()}?mode=ro",uri=True)
    unique={};collide=set()
    for ref,address in conn.execute('SELECT entity_id,address FROM records WHERE source=1'):
        a=key(address)
        if not any(c.isdigit() for c in a) or len(a.split())<4:continue
        if a in unique and unique[a]!=ref:collide.add(a)
        else:unique[a]=ref
    for a in collide:unique.pop(a,None)
    owner={}
    with (ART/'final_u5_out/matching_results.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            for t in filter(None,r['matched_entity_ids'].split(',')):owner[t]=r['source1_entity_id']
    planned={}
    for p,col in [(ART/'U7_added_vs_U5.tsv','added_entity_id'),(ART/'U8_added_vs_U7.tsv','added_entity_id'),(ART/'U9_rule_channel_links.tsv','entity_id')]:
        with p.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                planned.setdefault(r[col],set()).add(r['source1_entity_id'])
    rows=[];c=Counter()
    for target,name,address in conn.execute('SELECT entity_id,name,address FROM records WHERE source IN (2,3)'):
        pat=pattern(name)
        if not pat:continue
        ref=unique.get(key(address))
        if not ref:continue
        c['all_synthetic_at_unique_address']+=1
        if target in owner or target in planned:continue
        c['unowned_unplanned']+=1
        rows.append((ref,target,pat,name,address))
    out=ROOT/'u5_report/proposed_morpheme_sorted_links.tsv'
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['source1_entity_id','entity_id','pattern','target_name','target_address'])
        w.writerows(rows)
    c.update(Counter(r[2] for r in rows))
    print(dict(c),'unique pairs',len({(r[0],r[1]) for r in rows}),flush=True)

if __name__=='__main__':main()
