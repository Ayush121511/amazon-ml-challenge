"""Gold precision of synthetic aliases sharing an accepted true target address.

The train anchor uses a different true target from the evaluated alias. This
approximates a high-precision accepted test link without hiding the target's
own label in its anchor.
"""

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

def key(s):
    s=unicodedata.normalize('NFKD',s or '').lower()
    s=''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(sorted(re.findall(r'[a-z0-9]+',s)))

def pattern(name):
    toks=name.lower().split()
    if len(toks)==1 and parse(toks[0])>=2:return 'one'
    if len(toks)==2 and parse(toks[0])>=2 and toks[1].strip('.,+!') in SUFFIX:return 'suffix'
    return ''

def main():
    gold_owner={}
    with (ROOT/'train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ref=r['source1_entity_id']
            for t in filter(None,r['matched_entity_ids'].split(',')):
                gold_owner[t]=ref
    print('gold targets',len(gold_owner),flush=True)
    anchor={};collision=set();anchor_count=Counter();alias=[]
    for s in (2,3):
        with (ROOT/f'train_source{s}.tsv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                address=key(r['business_address'])
                if not any(c.isdigit() for c in address) or len(address.split())<4:continue
                target=r['entity_id'];owner=gold_owner.get(target)
                if owner:
                    if address in anchor and anchor[address]!=owner:collision.add(address)
                    else:anchor[address]=owner
                    anchor_count[(address,owner)]+=1
                pat=pattern(r['business_name'])
                if pat:alias.append((target,address,pat,owner))
        print('source',s,'anchors',len(anchor),'aliases',len(alias),flush=True)
    for address in collision:anchor.pop(address,None)
    by=Counter()
    for target,address,pat,owner in alias:
        ref=anchor.get(address)
        if not ref:continue
        if owner==ref and anchor_count[(address,ref)]<2:continue
        by[(pat,owner==ref)]+=1
    print('COUNTS',dict(by),'colliding addresses',len(collision),flush=True)
    (OUT/'train_accepted_copy_alias.json').write_text(json.dumps({str(k):v for k,v in by.items()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
