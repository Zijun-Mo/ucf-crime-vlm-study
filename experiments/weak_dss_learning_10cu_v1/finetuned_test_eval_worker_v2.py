import gc,hashlib,json,os,shutil,sys,time,traceback,subprocess
from pathlib import Path

DRIVE=Path('/content/drive/MyDrive/OpenJev_UCFCrime')
TRAIN=DRIVE/'weak_dss_learning_10cu_v1'
FORMAL=DRIVE/'formal_stride1'
DATA=DRIVE/'weak_dss_smoke_v1'
BASE=DRIVE/'models/base/15852e8c16360a2fea060d615a32b45270f8a8fc'
ORIGINAL=DRIVE/'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint'
OUT=TRAIN/'finetuned_test_v1'
OUT.mkdir(parents=True,exist_ok=True)
STATUS={'status':'starting','started':time.time(),'stage':'dependencies'}
def save_status():
    STATUS['elapsed_s']=time.time()-STATUS['started']
    tmp=OUT/'status.json.tmp';tmp.write_text(json.dumps(STATUS,indent=2));tmp.replace(OUT/'status.json')
try:
    dep=Path('/content/finetuned_test_deps');dep.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','--target',str(dep),'--upgrade','--no-deps','transformers==5.10.2','peft==0.19.1','av==19.0.0','einops==0.8.2','fla-core==0.5.2','tokenizers==0.22.2'],check=True,timeout=300)
    sys.path.insert(0,str(dep));sys.path.insert(0,str(TRAIN))
    import torch,numpy as np
    from openjev_dss.runtime import Runtime,VisualStore
    from openjev_dss.training import restore
    from openjev_dss.evaluation import freeze_test_lock,paired_report
    from openjev_dss.experiment import score_windows,calibrate_videos
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and BASE.exists() and ORIGINAL.exists()
    train_manifest=json.loads((TRAIN/'manifest.json').read_text())
    train_records=train_manifest['records']
    assert len([r for r in train_records if r['split']=='cal'])==163
    training_out=TRAIN/'training'
    checkpoints={};selections={}
    for arm in ['A','B']:
        folder=training_out/arm
        rows=json.loads((folder/'training.json').read_text())
        selection=json.loads((folder/'selection.json').read_text())
        assert len(rows)==160 and rows[-1].get('val_video_ap') is not None
        assert selection.get('test_used') is False and selection.get('epoch')==1
        checkpoints[arm]=folder/'best_state.pt'
        assert checkpoints[arm].is_file()
        selections[arm]=selection
    assert len({len(json.loads((training_out/a/'training.json').read_text())) for a in ['A','B']})==1
    save_status();STATUS.update(stage='load_model',gpu=torch.cuda.get_device_name(0),torch=torch.__version__);save_status()
    work=Path('/content/finetuned_eval_models');work.mkdir(exist_ok=True)
    for arm in ['A','B']:
        d=work/arm;d.mkdir(exist_ok=True)
        link=d/'adapter'
        if link.exists() or link.is_symlink():link.unlink()
        link.symlink_to(training_out/arm/'best_adapter',target_is_directory=True)
        shutil.copyfile(training_out/arm/'best_head.pt',d/'head.pt')
    rt=Runtime(str(BASE),str(ORIGINAL),str(FORMAL/'template.pt'),checkpointing=False)
    assert rt.full.config._name_or_path
    # Local manifests and symlinked canonical videos avoid duplicating large video corpora in Drive.
    calroot=Path('/content/finetuned_eval_cal');testroot=Path('/content/finetuned_eval_test')
    for root,canonical in [(calroot,DATA/'canonical'),(testroot,FORMAL/'canonical')]:
        root.mkdir(exist_ok=True)
        c=root/'canonical'
        if c.exists() or c.is_symlink():
            if c.is_symlink() or c.is_file():c.unlink()
            elif not any(c.iterdir()):c.rmdir()
        if not c.exists():c.symlink_to(canonical,target_is_directory=True)
        (root/'metadata').mkdir(exist_ok=True)
    cfg=json.loads((FORMAL/'config.json').read_text())
    calmeta=calroot/'metadata'
    for r in train_records:
        if r['split']=='cal':
            src=DATA/'metadata'/(r['name']+'.json')
            target=calmeta/(r['name']+'.json')
            if not target.exists():shutil.copyfile(src,target)
    testmeta=testroot/'metadata'
    canonical={x['name']:x for x in cfg['canonical']}
    # Freeze both selected checkpoint hashes and independent-calibration fits before reading test labels.
    calibrations={}
    cal_provider=VisualStore(rt,calroot,local_cache='/content/finetuned_feature_scratch')
    STATUS.update(stage='calibration',calibration_videos=163);save_status()
    for arm in ['A','B']:
        restore(rt.model,rt.head,torch.load(checkpoints[arm],map_location='cpu',weights_only=True))
        rt.model.eval();rt.head.eval()
        calibrations[arm]=calibrate_videos(rt.forward,train_records,cal_provider)
        STATUS['calibration_'+arm]=calibrations[arm];save_status()
        gc.collect();torch.cuda.empty_cache()
    cp_hash={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in checkpoints.items()}
    lock=freeze_test_lock(cp_hash,calibrations,OUT/'selection_lock.json')
    STATUS.update(stage='selection_locked',selection_lock_sha256=lock['sha256'],checkpoint_sha256=cp_hash);save_status()
    # Access official test temporal annotations only after the A/B checkpoint and calibration lock is durable.
    test_manifest=json.loads((FORMAL/'manifest.json').read_text())
    test_records=test_manifest['videos'];expected={x['name']:x['sha256'] for x in train_manifest['test_identity']}
    assert len(test_records)==290 and {r['name']:r['sha256'] for r in test_records}==expected
    for r in test_records:
        m=canonical[r['name']];assert m['source_sha256']==r['sha256']
        target=testmeta/(r['name']+'.json')
        if not target.exists():target.write_text(json.dumps({**m,'sha256':r['sha256'],'n_frames':r['n_frames'],'fps':r['fps']}))
    test_provider=VisualStore(rt,testroot,local_cache='/content/finetuned_feature_scratch')
    arms={a:{} for a in ['A','B']}
    for arm in ['A','B']:
        restore(rt.model,rt.head,torch.load(checkpoints[arm],map_location='cpu',weights_only=True))
        rt.model.eval();rt.head.eval();folder=OUT/arm;folder.mkdir(exist_ok=True)
        STATUS.update(stage='test_inference',model=arm,test_videos=0);save_status()
        for i,r in enumerate(test_records):
            target=folder/(r['name']+'.json')
            if target.exists():
                p=json.loads(target.read_text());assert p['selection_sha256']==lock['sha256']
            else:
                windows=test_provider(r,0,True)
                raw=score_windows(rt.forward,windows,4).tolist()
                p={'name':r['name'],'start_s':list(range(len(raw))),'raw_logits':raw,'selection_sha256':lock['sha256'],'temperature':calibrations[arm]['temperature'],'checkpoint_sha256':cp_hash[arm]}
                tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(p));tmp.replace(target)
            arms[arm][r['name']]=p
            STATUS.update(model=arm,test_videos=i+1,elapsed_s=time.time()-STATUS['started']);save_status()
        gc.collect();torch.cuda.empty_cache()
    report=paired_report(test_records,arms,OUT/'paired_test_report.json',bootstrap=1000,seed=42)
    summary={'complete':True,'status':'complete','training_epochs':1,'training_steps_per_arm':160,'training_checkpoints':selections,'test_videos':290,'calibration_videos':163,'checkpoint_sha256':cp_hash,'calibration':calibrations,'paired_report':report,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'elapsed_s':time.time()-STATUS['started'],'test_used_for_any_selection':False,'evaluation_note':'A and B were selected only on validation; calibration was fit on disjoint calibration videos; official test annotations were read only after the checkpoint/calibration lock.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));STATUS.update(status='complete',stage='done',summary=summary);save_status()
    print(json.dumps(summary,indent=2))
except Exception:
    STATUS.update(status='error',stage='failed',error=traceback.format_exc());save_status();print(STATUS['error']);raise
