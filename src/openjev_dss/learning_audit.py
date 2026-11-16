"""Conservative image-only family isolation; never reads temporal test labels."""
import json,math,random,hashlib
from pathlib import Path
from collections import defaultdict

def candidate_pairs(items):
    # <=4 changed bits in 64 implies one unchanged block among five blocks.
    buckets=defaultdict(set);pairs=set()
    for i,r in enumerate(items):
        assert len(r['phashes'])>=1, r['name']
        for bits in r['phashes']:
            assert len(bits)==64 and set(bits)<={'0','1'}
            value=int(bits,2)
            for start,end in [(0,13),(13,26),(26,39),(39,52),(52,64)]:
                key=(start,(value>>start)&((1<<(end-start))-1))
                for j in buckets[key]:
                    if j!=i:pairs.add((j,i))
                buckets[key].add(i)
    values=[[int(p,2) for p in r['phashes']] for r in items]
    accepted=[]
    for i,j in sorted(pairs):
        a,b=values[i],values[j]
        def covered(query,source):
            hits=0;matched=set()
            for x in query:
                locations=[k for k,y in enumerate(source) if (x^y).bit_count()<=4]
                if locations:hits+=1;matched.update(locations)
            return hits,len(matched),hits>=max(min(3,len(query)),math.ceil(.75*len(query))) and len(matched)>=min(3,len(source))
        ha,ma,ca=covered(a,b);hb,mb,cb=covered(b,a)
        if ca or cb:
            accepted.append({'a':items[i]['name'],'b':items[j]['name'],'hits':[ha,hb]})
    return accepted

def isolate(root,test_fingerprints,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text());records=manifest['records']
    train=[]
    for r in records:
        m=json.loads((root/'metadata'/(r['name']+'.json')).read_text())
        assert m['sha256']==r['sha256'] and m['fingerprint_version']=='canonical-uniform-plus30s-v3'
        train.append({**r,'phashes':m['phashes']})
    test=json.loads(Path(test_fingerprints).read_text());expected={r['name']:r['sha256'] for r in manifest['test_identity']}
    assert len(test)==290 and {r['name']:r['sha256'] for r in test}==expected
    items=train+test;parent={r['name']:r['name'] for r in items}
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    groups=defaultdict(list)
    for r in records:groups[r['group']].append(r['name'])
    for ns in groups.values():
        for n in ns[1:]:union(ns[0],n)
    candidates=candidate_pairs(items)
    for p in candidates:union(p['a'],p['b'])
    test_components={find(n) for n in expected};quarantine=[];kept=defaultdict(list)
    for r in records:
        if find(r['name']) in test_components:quarantine.append({'name':r['name'],'reason':'conservative_visual_source_test_overlap'})
        else:kept[find(r['name'])].append(r)
    # Mixed weak labels may be a common camera/source, not conflicting duplicate labels.
    strata=defaultdict(list)
    for g,rs in kept.items():strata['+'.join(sorted({r['category'] for r in rs}))].append((g,rs))
    rng=random.Random(42);split=[]
    for cat in sorted(strata):
        gs=sorted(strata[cat]);rng.shuffle(gs);total=sum(len(rs) for _,rs in gs);counts=dict(train=0,val=0,cal=0)
        ratios={'train':.8,'val':.1,'cal':.1}
        for g,rs in gs:
            choice=max(counts,key=lambda x:(ratios[x]*total-counts[x])/max(1,ratios[x]*total))
            split.extend({**r,'group':g,'split':choice} for r in rs);counts[choice]+=len(rs)
    for sp in ['train','val','cal']:
        assert {r['label'] for r in split if r['split']==sp}=={0,1}
    assert not any(set(r['group'] for r in split if r['split']==a)&set(r['group'] for r in split if r['split']==b) for a,b in [('train','val'),('train','cal'),('val','cal')])
    report={'prepared':len(train),'test_images':len(test),'candidates':candidates,'new_quarantine':quarantine,'policy':'All candidate families conservatively grouped; all train families touching test images quarantined. Candidates are not claims of exact duplication. Uniform8 plus every30s fingerprints; >=75% coverage in either direction and >=3 matched candidate positions where available. This cannot rule out every semantic near-duplicate.'}
    (out/'visual_audit.json').write_text(json.dumps(report,indent=2))
    manifest['records']=sorted(split,key=lambda r:r['name']);manifest['smoke']=[]
    manifest['counts']={s:sum(r['split']==s for r in split) for s in ['train','val','cal']}
    manifest['audit']['quarantined']+=quarantine
    manifest['audit']['near_duplicate_visual_audit']='passed';manifest['audit']['visual_method']=report['policy']
    manifest['audit']['split_rebuilt_before_training']=True
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest['counts'],report
