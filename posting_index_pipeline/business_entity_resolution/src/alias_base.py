import re, unicodedata, sys
from collections import Counter, defaultdict
D='/Users/ayushsrivastava/Documents/AmazonMLChallenge/data/student_resource/dataset/train/'
fold=lambda s:''.join(c for c in unicodedata.normalize('NFKD',s or '') if not unicodedata.combining(c)).lower()
toks=lambda s:re.sub(r'[^a-z0-9]+',' ',fold(s)).split()
STOP={'inc','llc','ltd','corp','co','pvt','private','limited','the','and','of','company','corporation','lp','llp','pc','group','services','service','center','partners','holdings','enterprises','india'}
def akey(a):
    t=toks(a); nums=[x for x in t if x.isdigit()]
    if not nums: return None
    words=sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x)>2))
    return (str(int(nums[0])), ' '.join(words[:6]))
owner={}
f=open(D+'train_ground_truth.tsv'); next(f)
for l in f:
    q=l.rstrip('\n').split('\t')
    for t in (q[1].split(',') if len(q)>1 else []):
        if t.strip(): owner[t.strip()]=q[0]
s1={}; idx=defaultdict(list)
f=open(D+'train_source1.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
for l in f:
    q=l.rstrip('\n').split('\t'); k=akey(q[a]) if len(q)>a else None
    s1[q[ie]]=q[n]
    if k: idx[(q[c],)+k].append(q[ie])
R=Counter()
for src in (2,3):
    f=open(D+f'train_source{src}.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
    for l in f:
        q=l.rstrip('\n').split('\t')
        if len(q)<=a or not q[a].strip(): continue
        k=akey(q[a]); hit=idx.get((q[c],)+k) if k else None
        if not hit or len(hit)!=1: continue
        s=hit[0]; rt=set(toks(q[n]))-STOP; st=set(toks(s1[s]))-STOP
        if not rt or rt & st: continue
        single = len(toks(q[n]))==1 and toks(q[n])[0].isalpha()
        kind='invented_single_word' if single else 'no_shared_word_multi'
        R[(kind, 'true' if owner.get(q[ie])==s else ('other_S1' if q[ie] in owner else 'nobody'))]+=1
for kind in ('invented_single_word','no_shared_word_multi'):
    tot=sum(v for (k,_),v in R.items() if k==kind)
    print(kind, 'n', tot, {o: f'{R[(kind,o)]/max(tot,1):.3f}' for o in ('true','other_S1','nobody')})
