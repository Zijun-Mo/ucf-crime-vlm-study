"""Reproduce accepted metrics using only saved predictions, on CPU.

Install requirements.txt, then --verify-files; --all --bootstrap 0;
--condition qwen_localization_v4_plateau_test --bootstrap 1000 --paired qwen_zs_p1_4s
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from study_evaluation import evaluate, resampling_weights, interval

HERE=Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE/name).read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-files',action='store_true')
    p.add_argument('--condition',default='qwen_localization_v4_plateau_test')
    p.add_argument('--all',action='store_true')
    p.add_argument('--bootstrap',type=int,default=0,help='0 = point estimates only; 1000 = registered paired video intervals, seed42')
    p.add_argument('--paired',help='Compare against another included condition using the same resampled videos')
    p.add_argument('--output',type=Path,help='Optional destination for recomputed results')
    a=p.parse_args()
    if a.bootstrap<0:p.error('bootstrap must be non-negative')
    if a.verify_files:
        count=0
        for line in (HERE/'data/files.sha256').read_text().splitlines():
            sha,name=line.split('  ',1);path=(HERE/name).resolve()
            if not path.is_relative_to(HERE):raise ValueError('Invalid manifest path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('SHA mismatch: '+name)
            count+=1
        print(json.dumps({'passed':True,'files_verified':count}))
        return
    conditions={c['id']:c for c in load('data/summary.json')}
    if a.condition not in conditions:p.error('Unknown condition: '+a.condition)
    if a.paired and a.paired not in conditions:p.error('Unknown paired baseline')
    records=load('data/test_videos.json')
    weights=resampling_weights(len(records),a.bootstrap,42) if a.bootstrap else np.ones((1,len(records)),dtype=np.int64)
    names=list(conditions) if a.all else [a.condition]
    if a.paired and a.paired not in names:names.append(a.paired)
    reports={};samples={}
    for name in names:
        c=conditions[name];predictions=load('data/predictions/'+name+'.json')
        for r in records:
            value=predictions.get(r['name'])
            if value is None or value.get('name')!=r['name'] or value.get('source_sha256')!=r['sha256']:
                raise ValueError('Prediction identity mismatch: '+r['name'])
        report,boot=evaluate(records,predictions,'raw_logits',weights,duration=c['duration'])
        max_error=max(abs(report['metrics'][k]['value']-v['value']) for k,v in c['metrics'].items())
        if max_error>1e-12:raise ValueError(f'Point estimate mismatch: {name}: {max_error}')
        if a.bootstrap==1000:
            for k,v in c['metrics'].items():
                if v['ci95'] is not None:
                    for endpoint in ['low','high']:
                        if abs(report['metrics'][k]['ci95'][endpoint]-v['ci95'][endpoint])>1e-12:
                            raise ValueError(f'CI mismatch: {name} / {k} / {endpoint}')
        if not a.bootstrap:
            for metric in report['metrics'].values():metric['ci95']=None
        reports[name]={'metrics':report['metrics'],'videos':report['videos'],'frames':report['frames'],'max_point_error':max_error}
        samples[name]=boot
        print(json.dumps({'condition':name,'passed':True,'max_point_error':max_error,'bootstrap':a.bootstrap}),flush=True)
    result={'passed':True,'unit':'video','bootstrap_replicates':a.bootstrap,'seed':42,'conditions':reports}
    if a.paired:
        left=a.condition;right=a.paired
        contrast={k:{'difference':reports[left]['metrics'][k]['value']-reports[right]['metrics'][k]['value'],
                     'ci95':interval(samples[left][k]-samples[right][k]) if a.bootstrap else None} for k in samples[left]}
        if a.bootstrap==1000:
            registered=load('data/comparisons.json')['comparisons']
            matches=[c for c in registered if c['coefficients']=={left:1,right:-1}]
            if matches:
                for k,v in matches[0]['metrics'].items():
                    for endpoint in ['low','high']:
                        if abs(contrast[k]['ci95'][endpoint]-v['ci95'][endpoint])>1e-12:
                            raise ValueError('Paired CI mismatch: '+k)
        result['paired']={'candidate':left,'baseline':right,'metrics':contrast}
        print(json.dumps(result['paired'],ensure_ascii=False,indent=2))
    if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
