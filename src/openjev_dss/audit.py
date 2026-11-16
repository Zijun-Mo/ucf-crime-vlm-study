"""Fail-closed visual near-duplicate audit before formal training."""
import json
from pathlib import Path

def near_duplicate_audit(root,test_fingerprints):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());records=manifest['records']
    metadata=[]
    for r in records:
        path=root/'metadata'/(r['name']+'.json')
        if not path.exists():raise ValueError('Full CPU preparation incomplete: '+r['name'])
        m=json.loads(path.read_text());assert m['sha256']==r['sha256'];metadata.append(m)
    # Test fingerprints contain filenames and image hashes only, no temporal labels.
    test=json.loads(Path(test_fingerprints).read_text());candidates=[]
    assert len(test)==290 and len({r['name'] for r in test})==290
    expected={r['name']:r['sha256'] for r in manifest['test_identity']}
    assert {r['name'] for r in test}==set(expected)
    for r in test:assert r['sha256']==expected[r['name']] and len(r['phashes'])>=3
    def near(a,b):
        aa=a['phashes'];bb=b['phashes'];minimum=min(len(aa),len(bb))
        if minimum<3:return False
        hits=sum(min(sum(x!=y for x,y in zip(p,q)) for q in bb)<=4 for p in aa)
        return hits>=max(3,int(.75*minimum))
    for i,a in enumerate(metadata):
        for b in metadata[i+1:]:
            if a['group']!=b['group'] and near(a,b):candidates.append({'a':a['name'],'b':b['name'],'scope':'train_family','needs_group_review':True})
        for b in test:
            if near(a,b):candidates.append({'a':a['name'],'b':b['name'],'scope':'test_overlap','needs_quarantine_review':True})
    report={'prepared':len(metadata),'test_fingerprints':len(test),'candidates':candidates,'method':'multiple pHash hits, <=4/64 Hamming distance, >=75% matching sparse frames; review candidates, not proof of identity'}
    (root/'visual_audit.json').write_text(json.dumps(report,indent=2))
    if candidates:raise ValueError('Review visual candidates, group families/quarantine overlap, then regenerate split before formal training')
    manifest['audit']['near_duplicate_visual_audit']='passed';manifest['audit']['visual_method']=report['method']
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));return report

def make_test_fingerprints(test_root, identity, out):
    """CPU-only sparse images and file identity; no test temporal labels."""
    import cv2,numpy as np,hashlib,shutil
    test_root=Path(test_root);assert len(identity)==290;result=[]
    cfg=json.loads((test_root/'config.json').read_text());canon={r['name']:r for r in cfg['canonical']}
    for rec in identity:
        src=test_root/'canonical'/rec['name'];meta=canon[rec['name']]
        assert meta['source_sha256']==rec['sha256']
        data=src.read_bytes();assert hashlib.sha256(data).hexdigest()==meta['canonical_sha256']
        local=Path('/content/dss_fingerprint.mp4');local.write_bytes(data);cap=cv2.VideoCapture(str(local));n=meta['sampled_frames'];hashes=[]
        for i in sorted(set(range(0,n,60))|set(np.linspace(0,n-1,min(8,n),dtype=int).tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,image=cap.read();assert ok
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8]
            hashes.append(''.join(str(int(v)) for v in (d>d.flatten()[1:].mean()).flatten()))
        cap.release();local.unlink();result.append({**rec,'phashes':hashes})
    Path(out).write_text(json.dumps(result,indent=2));return result


def canonical_phashes(src, sampled_frames):
    import cv2,numpy as np,shutil
    src=Path(src);local=Path('/content/dss_hash_'+src.name);shutil.copyfile(src,local)
    cap=cv2.VideoCapture(str(local));hashes=[]
    try:
        for i in np.linspace(0,sampled_frames-1,min(8,sampled_frames),dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,image=cap.read();assert ok
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8]
            hashes.append(''.join(str(int(v)) for v in (d>d.flatten()[1:].mean()).flatten()))
    finally:cap.release();local.unlink(missing_ok=True)
    return hashes


def dense_canonical_phashes(src,sampled_frames):
    import cv2,numpy as np,shutil
    src=Path(src);local=Path('/content/dss_dense_'+src.name);shutil.copyfile(src,local)
    cap=cv2.VideoCapture(str(local));hashes=[]
    positions=sorted(set(range(0,sampled_frames,60))|set(np.linspace(0,sampled_frames-1,min(8,sampled_frames),dtype=int).tolist()))
    try:
        for i in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,image=cap.read();assert ok
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8]
            hashes.append(''.join(str(int(v)) for v in (d>d.flatten()[1:].mean()).flatten()))
    finally:cap.release();local.unlink(missing_ok=True)
    return hashes

def refresh_dense_fingerprints(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    from concurrent.futures import ThreadPoolExecutor
    def refresh(r):
        p=root/'metadata'/(r['name']+'.json');info=json.loads(p.read_text())
        if info.get('fingerprint_version')=='canonical-uniform-plus30s-v3':return
        info['phashes']=dense_canonical_phashes(root/'canonical'/r['name'],info['sampled_frames'])
        info['fingerprint_version']='canonical-uniform-plus30s-v3';p.write_text(json.dumps(info))
    with ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(refresh,m['records']))
