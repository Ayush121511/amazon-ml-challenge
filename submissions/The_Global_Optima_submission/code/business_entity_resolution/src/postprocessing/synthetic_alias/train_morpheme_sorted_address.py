"""Gold precision for synthetic aliases at unique token-sorted full address."""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from invented_morpheme_probe import parse

ROOT=Path('6ab10eb3b23ba_student_resource/student_resource/dataset/train')
OUT=Path('artifacts/france_full_audit_20260927/u5_report')
SUFFIX={'labs','one','sys','co'}

def addr_key(s):
    s=unicodedata.normalize('NFKD',s or '').lower()
    s=''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(sorted(re.findall(r'[a-z0-9]+',s)))

def pattern(name):
    toks=name.lower().split()
    if len(toks)==1 and parse(toks[0])>=2:return 'one'
    if len(toks)==2 and parse(toks[0])>=2 and toks[1].strip('.,+!') in SUFFIX:return 'suffix'
    return ''

def main():
    unique={};collide=set()
    with (ROOT/'train_source1.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            key=(r['country'],addr_key(r['business_address']))
            if not any(c.isdigit() for c in key[1]) or len(key[1].split())<4:continue
            if key in unique and unique[key]!=r['entity_id']:collide.add(key)
            else:unique[key]=r['entity_id']
    for k in collide:unique.pop(k,None)
    print('unique S1 address keys',len(unique),flush=True)
    target_to_ref={};target_pattern={};target_source={}
    for s in (2,3):
        with (ROOT/f'train_source{s}.tsv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                pat=pattern(r['business_name'])
                if not pat:continue
                ref=unique.get((r['country'],addr_key(r['business_address'])))
                if ref:
                    target_to_ref[r['entity_id']]=ref
                    target_pattern[r['entity_id']]=pat
                    target_source[r['entity_id']]=s
        print('source',s,'pairs',len(target_to_ref),flush=True)
    truth=set()
    with (ROOT/'train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ref=r['source1_entity_id']
            for t in filter(None,r['matched_entity_ids'].split(',')):
                if target_to_ref.get(t)==ref:truth.add(t)
    by=Counter()
    for t,pat in target_pattern.items():
        by[(pat,t in truth)]+=1
        by[(pat,target_source[t],t in truth)]+=1
    print('COUNTS',dict(by),flush=True)
    (OUT/'train_morpheme_sorted_address.json').write_text(json.dumps({str(k):v for k,v in by.items()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
