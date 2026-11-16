"""CPU-only full-test scoring after a selected localization GPU job settles."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import configure_drive_environment
from openjev_dss.learning import atomic_json
from openjev_dss.study_evaluation import evaluate,interval,resampling_weights,frame_labels,frame_scores
from openjev_dss.followup_budget import spending

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('receipt',type=Path);parser.add_argument('--offline',action='store_true',help='Recompute from already verified local raw predictions;no Drive access or GPU')
    args=parser.parse_args();args.receipt=args.receipt.resolve();receipt=json.loads(args.receipt.read_text())
    assert receipt['status']=='complete' and receipt.get('finished_utc'),'Only finalize a settled complete inference run'
    assert 'Active assignments: 0' in receipt['usage_after'],'Verify GPU release before CPU scoring'
    policy=json.loads((args.receipt.parent/'policy.json').read_text());name=policy['name'];dest=ROOT/'results'/name;dest.mkdir(exist_ok=True)
    assert len(policy['records'])==290 and all(r['split']=='test' for r in policy['records'])
    if receipt.get('formal_test') is False:
        receipt.setdefault('metadata_corrections',[]).append(dict(field='formal_test',old=False,new=True,
            reason='Inference launcher scaffold used pilot flag; frozen pre-inference policy has all290 official test identities. Metadata correction only, no prediction or selection changes.',
            corrected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        receipt['formal_test']=True;atomic_json(args.receipt,receipt)
    raw=dest/'raw';raw.mkdir(exist_ok=True)
    if not args.offline:
        configure_drive_environment(ROOT)
        subprocess.run(['rclone','copy',policy['output_remote'],str(raw),'--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID',
            '--transfers','12','--checkers','12','--log-level','ERROR'],check=True,timeout=600)
    if policy['name'].startswith('qwen_localization_v4_'):
        from openjev_dss.followup_test import completed_test_gate
        audit=completed_test_gate(policy,raw)
        atomic_json(args.receipt.parent.parent/'complete_prediction_audit.json',audit)
    records=policy['records'];predictions={};hashes={}
    for row in records:
        name=row['name'];path=raw/'predictions'/f'{name}.json';p=json.loads(path.read_text())
        assert p['fingerprint']==policy['fingerprint'] and p['checkpoint_tensor_sha256']==policy['checkpoint_tensor_sha256']
        assert p['source_sha256']==row['sha256'] and p['start_s']==list(range(len(p['raw_logits'])))
        baseline=json.loads((ROOT/policy['baseline_source']/(name+'.json')).read_text())
        assert baseline['source_sha256']==row['sha256'] and baseline['start_s']==p['start_s']
        predictions[name]={**p,'candidate':p['raw_logits'],'baseline':baseline['raw_logits']}
        hashes[str(path.relative_to(dest))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(predictions)==290 and sum(len(p['candidate']) for p in predictions.values())==37188
    weights=resampling_weights(290,1000,42);reports={};samples={}
    for arm in ['baseline','candidate']:reports[arm],samples[arm]=evaluate(records,predictions,arm,weights)
    existing=json.loads((ROOT/'results/study_formal_v2/qwen_zs_p1_4s.json').read_text())
    for key,value in reports['baseline']['metrics'].items():
        assert abs(value['value']-existing['metrics'][key]['value'])<1e-12,'Reused baseline differs from accepted full-test metrics'
    contrasts={k:dict(difference=reports['candidate']['metrics'][k]['value']-reports['baseline']['metrics'][k]['value'],
        ci95=interval(samples['candidate'][k]-samples['baseline'][k])) for k in samples['candidate']}
    from sklearn.metrics import roc_auc_score
    cases=[];distributions={a:{k:[] for k in ['normal_video','abnormal_background','anomaly']} for a in reports}
    for row in records:
        y=frame_labels(row);p=predictions[row['name']];z={a:frame_scores(row,p['start_s'],p[a]) for a in reports}
        mixed=0<y.sum()<len(y)
        cases.append(dict(name=row['name'],category=row['category'],duration_s=row['n_frames']/row['fps'],
            anomaly_fraction=float(y.mean()),within_delta=(float(roc_auc_score(y,z['candidate'])-roc_auc_score(y,z['baseline'])) if mixed else None)))
        for a in reports:
            for key,mask in [('normal_video',np.ones(len(y),bool) if not y.any() else np.zeros(len(y),bool)),
                             ('abnormal_background',(y==0) if y.any() else np.zeros(len(y),bool)),('anomaly',y==1)]:
                if mask.any():distributions[a][key].append(dict(name=row['name'],frames=int(mask.sum()),mean=float(z[a][mask].mean()),q90=float(np.quantile(z[a][mask],.9))))
    ordered=sorted((r for r in cases if r['within_delta'] is not None),key=lambda r:(r['within_delta'],r['name']))
    selected=dict(worst=ordered[0],best=ordered[-1],median=ordered[len(ordered)//2])
    result=dict(experiment=policy['name'],status='complete',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metrics={a:{k:v['value'] for k,v in report['metrics'].items()} for a,report in reports.items()},
        paired_contrasts=contrasts,reports=reports,bootstrap=dict(replicates=1000,seed=42,unit='paired video'),
        selected_epoch=policy['selected_epoch'],checkpoint_tensor_sha256=policy['checkpoint_tensor_sha256'],
        configuration_selected_on_test=False,gpu_released=True,optimizer_updates=0,cost=spending(ROOT))
    atomic_json(dest/'summary.json',result);atomic_json(dest/'prediction_sha256.json',hashes)
    atomic_json(dest/'per_video.json',cases);atomic_json(dest/'score_distribution.json',distributions)
    atomic_json(dest/'case_selection.json',dict(rule='min/max/median within-video AUROC difference, deterministic filename ties; descriptive only',cases=selected))
    registry_path=ROOT/'research/formal_experiment_registry.json';registry=json.loads(registry_path.read_text())
    registry['experiments'][policy['name']]=dict(status='complete',policy=str((args.receipt.parent/'policy.json').relative_to(ROOT)),
        receipt=str(args.receipt.relative_to(ROOT)),summary=str((dest/'summary.json').relative_to(ROOT)),videos=290,windows=37188,frames=1111808,gpu_released=True)
    atomic_json(registry_path,registry)
    if policy['name'].startswith('qwen_localization_v4_'):
        # Preserve V4's selected-checkpoint history; the legacy V3 renderer
        # predates this method and would otherwise erase its current status.
        plateau=policy['name']=='qwen_localization_v4_plateau_test'
        continuation=policy['name']=='qwen_localization_v4_epoch5_test'
        path=ROOT/('research/localization_v4_plateau/result_summary.json' if plateau else 'research/localization_v4_continuation/result_summary.json' if continuation else 'research/localization_v4/result_summary.json');state=json.loads(path.read_text())
        state.update(status='complete',full_test_complete=True,full_test_summary=str((dest/'summary.json').relative_to(ROOT)),budget=spending(ROOT))
        atomic_json(path,state)
        path=ROOT/'results/future_work_v1/inventory.json';index=json.loads(path.read_text())
        index['v4_plateau' if plateau else 'v4_continuation' if continuation else 'v4'].update(status='complete',full_test_complete=True,selected_epoch=policy['selected_epoch'],summary=str((dest/'summary.json').relative_to(ROOT)))
        index.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),budget=spending(ROOT));atomic_json(path,index)
        path=args.receipt.parent.parent/'registration.json';registration=json.loads(path.read_text())
        registration.update(status='complete',test_predictions_produced=True,full_test_summary=str((dest/'summary.json').relative_to(ROOT)))
        atomic_json(path,registration)
    else:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_frame_transfer.py'),'--include-teacher'],cwd=ROOT,check=True,timeout=180)
        subprocess.run([sys.executable,str(ROOT/'scripts/refresh_experiment_inventory.py')],cwd=ROOT,check=True,timeout=90)
        subprocess.run([sys.executable,str(ROOT/'scripts/render_followup_handoff.py')],cwd=ROOT,check=True,timeout=90)
    print(json.dumps(result['metrics'],indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
