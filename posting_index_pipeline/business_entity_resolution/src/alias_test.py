import re, unicodedata
from collections import Counter, defaultdict
D='/Users/ayushsrivastava/Documents/AmazonMLChallenge/'
T=D+'data/student_resource/dataset/test/'
fold=lambda s:''.join(c for c in unicodedata.normalize('NFKD',s or '') if not unicodedata.combining(c)).lower()
toks=lambda s:re.sub(r'[^a-z0-9]+',' ',fold(s)).split()
STOP={'inc','llc','ltd','corp','co','pvt','private','limited','the','and','of','company','corporation','lp','llp','pc','group','services','service','center','partners','holdings','enterprises','india','sarl','sas','sasu','eurl','sci','snc','sa','ei','de','la','le','les','des','du','et'}
def akey(a):
    t=toks(a); nums=[x for x in t if x.isdigit()]
    if not nums: return None
    words=sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x)>2))
    return (str(int(nums[0])), ' '.join(words[:6]))
def load(f):
    d={}
    for i,l in enumerate(open(f,encoding='utf-8')):
        if i: s,m=(l.rstrip('\n').split('\t')+[''])[:2]; d[s]=[x for x in m.split(',') if x]
    return d
u7=load('u7_matching.tsv'); cand=load(D+'submissions/final_u5/candidate_pairs.tsv')
link={t:s for s,ts in u7.items() for t in ts}; cs={(s,t) for s,ts in cand.items() for t in ts}
s1={}; cty={}; idx=defaultdict(list)
f=open(T+'test_source1.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
for l in f:
    q=l.rstrip('\n').split('\t'); k=akey(q[a]) if len(q)>a else None
    s1[q[ie]]=(q[n],q[a]); cty[q[ie]]=q[c]
    if k: idx[(q[c],)+k].append(q[ie])
R=Counter(); ex=defaultdict(list); add=[]
for src in (2,3):
    f=open(T+f'test_source{src}.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
    for l in f:
        q=l.rstrip('\n').split('\t')
        if len(q)<=a or not q[a].strip(): continue
        k=akey(q[a]); hit=idx.get((q[c],)+k) if k else None
        if not hit or len(hit)!=1: continue
        s=hit[0]; tk=toks(q[n])
        if not (len(tk)==1 and tk[0].isalpha()): continue
        if set(tk)&(set(toks(s1[s][0]))-STOP): continue
        r=q[ie]
        st='agree' if link.get(r)==s else ('other_S1' if r in link else ('unlinked_in_cand' if (s,r) in cs else 'unlinked_not_cand'))
        R[(q[c],st)]+=1
        if st!='agree' and len(ex[(q[c],st)])<5: ex[(q[c],st)].append(f'{s1[s][0]} | {s1[s][1][:40]}  <=  {q[n]} | {q[a][:40]}' + (f'   [U7 gave to {s1[link[r]][0]} | {s1[link[r]][1][:30]}]' if r in link else ''))
        if st=='unlinked_in_cand': add.append((s,r))
for k in sorted(R): print(k, R[k])
for k,v in ex.items():
    print(k); [print('   ',x) for x in v]
open('alias_adds.tsv','w').write(''.join(f'{s}\t{r}\n' for s,r in add))
