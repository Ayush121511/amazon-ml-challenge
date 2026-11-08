"""Find synthetic France aliases sharing a unique U5 accepted target address."""

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
    if len(toks)==1 and parse(toks[0])>=2:return 'one'
    if len(toks)==2 and parse(toks[0])>=2 and toks[1].strip('.,+!') in SUFFIX:return 'suffix'
    return ''

def main():
    conn=sqlite3.connect(f"file:{(ROOT/'indexed/france_test.sqlite3').as_posix()}?mode=ro",uri=True)
    data={t:(name,address,source) for t,source,name,address in conn.execute('SELECT entity_id,source,name,address FROM records WHERE source IN (2,3)')}
    accepted={};owned={}
    with (ART/'final_u5_out/matching_results.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ref=r['source1_entity_id']
            for t in filter(None,r['matched_entity_ids'].split(',')):
                if t in data:
                    accepted[t]=ref
                    owned[t]=ref
    anchors={};collision=set()
    for t,ref in accepted.items():
        a=key(data[t][1])
        if not any(c.isdigit() for c in a) or len(a.split())<4:continue
        if a in anchors and anchors[a]!=ref:collision.add(a)
        else:anchors[a]=ref
    for a in collision:anchors.pop(a,None)
    planned=set();planned_targets=set()
    for p,col in [(ART/'U7_added_vs_U5.tsv','added_entity_id'),(ART/'U8_added_vs_U7.tsv','added_entity_id'),(ART/'U9_rule_channel_links.tsv','entity_id')]:
        with p.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                planned.add((r['source1_entity_id'],r[col]));planned_targets.add(r[col])
    c=Counter();rows=[]
    for t,(name,address,source) in data.items():
        pat=pattern(name)
        if not pat or t in owned or t in planned_targets:continue
        ref=anchors.get(key(address))
        if not ref:continue
        c['unowned_unplanned']+=1
        rows.append((ref,t,pat,source,name,address))
    with (ROOT/'u5_report/accepted_copy_alias_links.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['source1_entity_id','entity_id','pattern','source','target_name','target_address'])
        w.writerows(rows)
    print('anchors',len(anchors),'collisions',len(collision),'rows',len(rows),'patterns',Counter(r[2] for r in rows),flush=True)

if __name__=='__main__':main()
