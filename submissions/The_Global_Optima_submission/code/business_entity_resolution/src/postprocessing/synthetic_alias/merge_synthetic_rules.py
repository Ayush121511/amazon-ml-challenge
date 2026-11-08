"""Union two train-backed alias channels and remove competing S1 claims."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('artifacts/france_full_audit_20260927/u5_report')

def main():
    rows={};claims=defaultdict(set)
    for path,channel in [(ROOT/'proposed_morpheme_sorted_links.tsv','unique_s1_address'),
                         (ROOT/'accepted_copy_alias_links.tsv','accepted_copy_address')]:
        with path.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                ref,target=r['source1_entity_id'],r['entity_id']
                pair=(ref,target)
                rows.setdefault(pair,set()).add(channel)
                claims[target].add(ref)
    ambiguous={target for target,refs in claims.items() if len(refs)>1}
    safe={pair:channels for pair,channels in rows.items() if pair[1] not in ambiguous}
    with (ROOT/'proposed_synthetic_alias_links_safe.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['source1_entity_id','entity_id','channels'])
        for (ref,target),channels in sorted(safe.items()):w.writerow([ref,target,','.join(sorted(channels))])
    counts=Counter()
    for channels in safe.values():counts['+'.join(sorted(channels))]+=1
    report={'input_unique_pairs':len(rows),'ambiguous_targets':len(ambiguous),
            'safe_pairs':len(safe),'by_channel':dict(counts)}
    (ROOT/'synthetic_alias_union_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report,flush=True)

if __name__=='__main__':main()
