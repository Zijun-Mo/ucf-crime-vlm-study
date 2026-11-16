"""Manifest-only split: no test temporal labels are read here."""
import hashlib,json,re,random,math
from collections import defaultdict
from pathlib import Path
import requests
REV='8d783bb55269300114f9773a3c068b48a4940e4a'
REPO='mteb/ucf-crime'

def source_group(name):
    # Conservative source-family grouping, including split/segment variants.
    stem=re.sub(r'_x264.*$','',name.rsplit('.',1)[0])
    return re.sub(r'(?i)(?:[_-](?:clip|part|segment|scene)?\d+)+$','',stem) if not stem.startswith('Normal_Videos_') else stem

def prepare_manifest(out):
    s=requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    import subprocess
    commit=subprocess.check_output(['git','ls-remote','https://github.com/WaqasSultani/AnomalyDetectionCVPR2018.git','HEAD'],text=True).split()[0]
    prefix=f'https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/{commit}/'
    train=s.get(prefix+'Anomaly_Train.txt',timeout=60);train.raise_for_status()
    test=s.get(prefix+'Temporal_Anomaly_Annotation.txt',timeout=60);test.raise_for_status()
    train_names=[x.strip().split('/')[-1] for x in train.text.splitlines() if x.strip()]
    test_names=[x.strip().split()[0].split('/')[-1] for x in test.text.splitlines() if x.strip()]
    assert len(train_names)==1610 and len(test_names)==290
    assert not set(train_names)&set(test_names)
    found={}
    root=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/videos'
    r=s.get(root,timeout=60);r.raise_for_status();folders=r.json()
    for folder in folders:
        if folder['type']!='directory':continue
        url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/'+folder['path']+'?limit=1000'
        while url:
            r=s.get(url,timeout=60);r.raise_for_status()
            for f in r.json():
                if f['type']=='file':found[Path(f['path']).name]=f
            url=r.links.get('next',{}).get('url')
    missing=set(train_names+test_names)-found.keys()
    assert not missing,sorted(missing)
    test_sha={found[n]['lfs']['oid'] for n in test_names}
    groups=defaultdict(list);records=[]
    # Union source families and exact file duplicates transitively.
    parent={n:n for n in train_names}
    def find(n):
        if parent[n]!=n:parent[n]=find(parent[n])
        return parent[n]
    seen_source={};seen_sha={};quarantined=[]
    for n in train_names:
        f=found[n];sha=f['lfs']['oid']
        if sha in test_sha:
            quarantined.append({'name':n,'reason':'exact_test_file_duplicate','test_names':[t for t in test_names if found[t]['lfs']['oid']==sha]});continue
        g=source_group(n)
        for key,seen in [(g,seen_source),(sha,seen_sha)]:
            if key in seen:parent[find(n)]=find(seen[key])
            else:seen[key]=n
        cat='Normal' if n.startswith('Normal') else re.match('[A-Za-z]+',n)[0]
        records.append(dict(name=n,category=cat,label=int(cat!='Normal'),repo_path=f['path'],sha256=sha,size=f['size'],source_group=g))
    for r in records:groups[find(r['name'])].append(r)
    bycat=defaultdict(list)
    for g,rs in groups.items():
        if len({r['label'] for r in rs})>1:
            quarantined.extend({'name':r['name'],'reason':'conflicting_weak_label_exact_duplicate'} for r in rs);continue
        bycat['+'.join(sorted({r['category'] for r in rs}))].append((g,rs))
    rng=random.Random(42);split=[]
    for cat in sorted(bycat):
        gs=sorted(bycat[cat]);rng.shuffle(gs);total=sum(len(rs) for _,rs in gs);counts=dict(train=0,val=0,cal=0)
        for g,rs in gs:
            choice=max(counts,key=lambda x:({'train':.8,'val':.1,'cal':.1}[x]*total-counts[x])/max(1,{'train':.8,'val':.1,'cal':.1}[x]*total))
            for r in rs:split.append({**r,'group':g,'split':choice})
            counts[choice]+=len(rs)
    # Smoke deterministic category-spread examples, no test data or labels.
    selected=[]
    for sp,n in [('train',8),('val',4)]:
        for y in [0,1]:
            candidates=sorted([r for r in split if r['split']==sp and r['label']==y],key=lambda r:(r['size'],r['name']))
            if y:
                chosen=[];seen=set()
                for r in candidates:
                    if r['category'] not in seen:chosen.append(r);seen.add(r['category'])
                    if len(chosen)==n:break
            else:chosen=candidates[:n]
            assert len(chosen)==n;selected+=chosen
    manifest=dict(dataset=REPO,revision=REV,official_commit=commit,official_train_sha256=hashlib.sha256(train.content).hexdigest(),official_test_sha256=hashlib.sha256(test.content).hexdigest(),records=sorted(split,key=lambda x:x['name']),test_identity=[{'name':t,'sha256':found[t]['lfs']['oid']} for t in sorted(test_names)],smoke=selected,audit=dict(official_train=1610,official_test=290,all_mapped=True,filename_overlap=0,sha256_overlap=len(quarantined),quarantined=quarantined,group_overlap=0,near_duplicate_visual_audit='pending; formal training must refuse until reviewed'),counts={sp:sum(r['split']==sp for r in split) for sp in ['train','val','cal']})
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def sample_starts(n,epoch,video,seed=42):
    if n<=16:return list(range(n))
    digest=int(hashlib.sha256(video.encode()).hexdigest()[:8],16)
    rng=random.Random(seed+epoch*1000003+digest)
    return [rng.randrange(i*n//16,(i+1)*n//16) for i in range(16)]
if __name__=='__main__':
    m=prepare_manifest('artifacts/dss');print(m['counts']);print([(r['name'],r['split']) for r in m['smoke']])
