"""Recompute V4 development metrics from complete raw scores, without a GPU."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.localization_training import validate_annotations
from openjev_dss.study_evaluation import frame_labels,frame_scores
from openjev_dss.followup_budget import spending,available


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--review-root',type=Path,default=Path('research/localization_v4'))
    parser.add_argument('--output-root',type=Path,default=Path('results/localization_v4_dev'))
    args=parser.parse_args();epoch=args.epoch
    folder=ROOT/args.review_root/f'epoch_{epoch}_review'
    manifest=json.loads((ROOT/'results/learning/manifest.json').read_text())
    records=sorted([r for r in manifest['records'] if r['split']=='val'],key=lambda r:r['name'])
    metadata=json.loads((ROOT/'research/formal_training_metadata.json').read_text())['records']
    annotation_path=ROOT/'research/localization_dev_v2/dev_annotations.json'
    dev=validate_annotations(json.loads(annotation_path.read_text()),manifest,metadata)
    assert len(dev)==34 and len(records)==163
    qual=json.loads((ROOT/'research/localization_v4/cpu_preflight.json').read_text())
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest()==qual['annotation_sha256']
    trained=json.loads((folder/f'epochs/epoch_{epoch}/metrics.json').read_text())
    baseline=json.loads((folder/'baseline_validation.json').read_text())
    checkpoint=json.loads((folder/f'epoch{epoch}_checkpoint_payload_audit.json').read_text())
    assert checkpoint['passed'] and checkpoint['actual_tensor_sha256']==trained['state_sha256']
    prediction_folder=folder/f'validation/epoch_{epoch}'
    assert {p.stem for p in prediction_folder.glob('*.json') if p.name!='metrics.json'}=={r['name'] for r in records}
    yvideo=[];zvideo=[[],[]];yf=[];zf=[[],[]];within=[[],[]];cases=[];hashes={}
    for r in records:
        name=r['name'];teacher_path=ROOT/'.colab-work/localization_v3_teacher'/(name+'.json')
        assert hashlib.sha256(teacher_path.read_bytes()).hexdigest()==qual['teacher_json_sha256'][teacher_path.name]
        old=json.loads(teacher_path.read_text());p=prediction_folder/(name+'.json');new=json.loads(p.read_text())
        assert old['source_sha256']==new['source_sha256']==r['sha256'] and new['state_sha256']==trained['state_sha256']
        scores=[old['logits'],new['logits']]
        assert all(len(z)==metadata[name]['window_count'] and np.isfinite(z).all() for z in scores)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        yvideo.append(r['label'])
        for j,z in enumerate(scores):zvideo[j].append(max(z))
        if r['label'] and name not in dev:continue
        meta=dev[name] if r['label'] else {**metadata[name],'intervals':[]}
        y=frame_labels(meta);yf.append(y);curves=[]
        for j,z in enumerate(scores):
            curve=frame_scores(meta,list(range(len(z))),z);zf[j].append(curve);curves.append(curve)
            if r['label']:within[j].append(float(roc_auc_score(y,curve)))
        if r['label']:cases.append(dict(name=name,zero_shot=within[0][-1],trained=within[1][-1],delta_pp=100*(within[1][-1]-within[0][-1])))
    y=np.concatenate(yf);results=[]
    for j in range(2):
        z=np.concatenate(zf[j]);results.append(dict(dev_pooled_auroc=float(roc_auc_score(y,z)),
            dev_pooled_ap=float(average_precision_score(y,z)),within_macro_auroc=float(np.mean(within[j])),
            video_max_ap=float(average_precision_score(yvideo,zvideo[j])),video_max_auroc=float(roc_auc_score(yvideo,zvideo[j]))))
    errors=[]
    for actual,recorded in zip(results,[baseline,trained]):
        for key,value in actual.items():errors.append(abs(value-recorded[key]))
    assert max(errors)<1e-12
    guards={k:results[1][k]>=results[0][k] for k in ['within_macro_auroc','dev_pooled_ap','video_max_ap']}
    report=dict(passed=True,epoch=epoch,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        baseline=results[0],trained=results[1],delta_pp={k:100*(results[1][k]-results[0][k]) for k in results[0]},
        guards=guards,qualified_candidate=all(guards.values()) and results[1]['dev_pooled_auroc']>results[0]['dev_pooled_auroc'],
        video_validation_count=len(records),frame_dev_count=len(yf),within_video_count=len(cases),frame_count=len(y),
        maximum_recomputation_error=max(errors),raw_prediction_sha256=hashes,checkpoint=checkpoint,
        per_video_within_auroc=cases,annotation_sha256=qual['annotation_sha256'],
        budget=spending(ROOT),remaining_training_cu=available(ROOT,'v3_training'),
        gpu_allocated=False,test_used=False,confidence_intervals_computed=False,
        limitation='Development point estimates after checkpoint selection; approximate reviewed temporal labels; no inference about full-test performance or causality')
    if epoch==1:
        recovery=json.loads((ROOT/'research/localization_v4/teacher_cache_recovery.json').read_text())
        health=json.loads((ROOT/'research/localization_v4/monitor/health.json').read_text());progress=health.get('progress',{})
        if progress.get('epoch_cursor',0)>=1 and progress.get('steps',0)>157:
            stop=datetime.datetime.fromisoformat(health['checked_utc']).timestamp()
            report['time_audit']=dict(reported_epoch_hours=trained['epoch_active_hours'],
                conservative_wall_upper_hours=(stop-recovery['completed'])/3600,
                rationale='Starts before epoch1, after cache repair; ends after epoch2 optimizer updates, includes final epoch1 synchronization',
                under_three_hours=(stop-recovery['completed'])/3600<3)
    out=ROOT/args.output_root/f'epoch_{epoch}_comparison.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','epoch','baseline','trained','delta_pp','guards','qualified_candidate','maximum_recomputation_error','remaining_training_cu','time_audit'] if k in report},ensure_ascii=False))


if __name__=='__main__':main()
