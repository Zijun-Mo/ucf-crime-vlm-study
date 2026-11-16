"""Registered complete-test storyboard comparison, with unchanged pilot scoring."""
import concurrent.futures,hashlib,json,os,sys,time,traceback
from pathlib import Path
work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import numpy as np,requests,torch
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.learning import atomic_json
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.storyboard_context import make_sheets,NativeContextScorer
from openjev_dss.storyboard_stream import StreamingWindows
from openjev_dss.storyboard_cached import cache_signature,PairFeatureView,score_cached_video
from openjev_dss.study_evaluation import evaluate,interval

out=Path('/content/ucf_storyboard');out.mkdir(exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
policy=json.loads((work/'research/storyboard_context_formal/policy.json').read_text())
assert hashlib.sha256((work/'src/openjev_dss/storyboard_context.py').read_bytes()).hexdigest()==policy['context_module_sha256']
deadline=float(os.environ['UCF_DEADLINE']);remote=policy['output_remote'];started=time.time()
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','2','--retries-sleep','2s']
state=dict(status='running',stage='initializing',completed_videos=0,total_videos=290,scored_windows=0,
    optimizer_updates=0,started=started,test_used=True,fingerprint=policy['fingerprint'])
pool=None
def status(stage,**extra):
    state.update(stage=stage,updated=time.time(),**extra);atomic_json(out/'status.json',state)
def guard():
    if time.time()>deadline-240:raise TimeoutError('Formal experiment durable-save reserve')
def publish():
    checked(rc+['copy',str(out),remote,'--transfers','8','--checkers','8','--checksum'],timeout=240)
def original(record):
    path=Path('/content/storyboard_original')/record['name'];path.parent.mkdir(exist_ok=True)
    url='https://huggingface.co/datasets/mteb/ucf-crime/resolve/'+policy['dataset_revision']+'/'+record['repo_path']
    for attempt in range(3):
        try:
            if not path.exists():
                response=requests.get(url,stream=True,timeout=(20,60));response.raise_for_status();temp=path.with_suffix('.partial')
                with temp.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):guard();f.write(chunk)
                with temp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']
                temp.replace(path)
            return path
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))
try:
    assert torch.cuda.get_device_name()=='NVIDIA L4'
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    # A fixed remote directory permits exact per-video/arm resume, never a
    # second model or a different context policy in the same result artifact.
    checked(rc+['mkdir',remote],timeout=60)
    checked(rc+['copy',remote,str(out),'--transfers','8','--checkers','8'],timeout=300)
    if (out/'policy.json').exists():assert json.loads((out/'policy.json').read_text())['fingerprint']==policy['fingerprint']
    atomic_json(out/'policy.json',policy);atomic_json(out/'registration.json',json.loads((work/'research/storyboard_context_formal/registration.json').read_text()))
    predictions={}
    for record in policy['records']:
        path=out/'predictions'/f'{record["name"]}.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['fingerprint']==policy['fingerprint'] and row['source_sha256']==record['sha256']
            assert row['start_s']==list(range(policy['metadata'][record['name']]['window_count']))
            predictions[record['name']]=row
            state['scored_windows']+=sum(len(row[a]) for a in policy['arms'] if a in row)
            state['completed_videos']+=int(all(a in row for a in policy['arms']))
    remaining=[r for r in policy['records'] if not all(a in predictions.get(r['name'],{}) for a in policy['arms'])]
    data=Path('/content/storyboard_canonical');data.mkdir(exist_ok=True)
    if remaining:
        status('staging_canonical')
        listing=Path('/content/storyboard_canonical_files.txt');listing.write_text('\n'.join(r['name'] for r in remaining)+'\n')
        checked(rc+['copy',policy['canonical_remote'],str(data),'--files-from',str(listing),'--transfers','8','--checkers','8'],timeout=900)
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=3);futures={}
        for i in range(min(3,len(remaining))):futures[i]=pool.submit(original,remaining[i])
        status('loading_model');os.environ['HF_HUB_DISABLE_XET']='1'
        from huggingface_hub import snapshot_download
        base=snapshot_download(policy['model_id'],revision=policy['revision'],token=False,
            allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'],max_workers=4)
        runtime=ControlledRuntime(base,policy['prompt'],train=False);assert runtime.trainable_parameters==0
        runtime.model.eval();runtime.head.eval();template=Path('/content/ucf_templates/template_4s.mp4')
        cache_gates=[];cache_admitted=set();cache_rejected=set();cache_hits=0;cache_fallbacks=0
        for number,record in enumerate(remaining):
            guard();name=record['name'];m=policy['metadata'][name];status('staging_video',active_video=name)
            raw=futures.pop(number).result()
            if number+3<len(remaining):futures[number+3]=pool.submit(original,remaining[number+3])
            source=data/name
            with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['canonical_sha256']
            images,layout=make_sheets(raw,out/'sheets'/name);windows=StreamingWindows(source,m)
            row=predictions.get(name,dict(name=name,source_sha256=record['sha256'],fingerprint=policy['fingerprint'],start_s=list(range(len(windows)))))
            row['baseline']=policy['baseline_predictions'][name]['raw_logits']
            row['layout']=layout
            feature_view=None
            signature,cache_name=cache_signature(runtime,record,m,policy['revision'])
            cache_path=Path('/content/storyboard_features')/cache_name
            status('staging_visual_cache',active_video=name)
            try:
                checked(rc+['copyto',policy['visual_cache_remote']+'/'+cache_name,str(cache_path)],timeout=300)
                feature_view=PairFeatureView(cache_path,signature,len(windows))
            except Exception as e:
                row['visual_cache_read_error']=sanitize(str(e))[-500:]
                guard()
            last_update=[0.]
            def scoring_guard():
                guard()
                if time.time()-last_update[0]>20:
                    status('scoring',active_window_cursor=windows.cursor);last_update[0]=time.time()
            for arm in policy['arms']:
                if arm in row:continue
                windows.cursor=0
                status('scoring',active_arm=arm,active_window_cursor=0)
                scorer=NativeContextScorer(runtime,images if arm=='storyboard' else [],policy['prompt'],template,context=arm!='baseline')
                shape=scorer.info['image_tokens']
                if feature_view is not None and shape not in cache_admitted and shape not in cache_rejected:
                    native_probe=scorer.score(windows[:4],guard=guard);windows.close();windows.cursor=0
                    cached_probe=score_cached_video(scorer,feature_view[:len(native_probe)],guard=guard)
                    error=float(np.max(np.abs(np.asarray(native_probe)-np.asarray(cached_probe))))
                    gate=dict(name=name,image_tokens=shape,max_abs_logit_error=error,native_logits=native_probe,cached_logits=cached_probe,allowed=error<1e-4,threshold=1e-4)
                    cache_gates.append(gate);atomic_json(out/'visual_cache_gates.json',cache_gates)
                    (cache_admitted if gate['allowed'] else cache_rejected).add(shape)
                use_cache=feature_view is not None and shape in cache_admitted
                arm_started=time.time()
                row[arm]=(score_cached_video(scorer,feature_view,guard=scoring_guard) if use_cache else scorer.score(windows,guard=scoring_guard))
                row['visual_cache_reused']=use_cache;row['visual_cache_key']=cache_name
                cache_hits+=int(use_cache);cache_fallbacks+=int(not use_cache)
                row.setdefault('arm_seconds',{})[arm]=time.time()-arm_started
                assert len(row[arm])==len(windows)
                if arm=='storyboard':row['context_input']=scorer.info
                state['scored_windows']+=len(row[arm]);atomic_json(out/'predictions'/f'{name}.json',row)
                if len(windows)>512:publish()
            windows.close();predictions[name]=row;state['completed_videos']+=1
            status('scoring',active_window_cursor=len(windows))
            if (number+1)%4==0 or number+1==len(remaining):publish()
            print('FORMAL_CONTEXT_VIDEO',state['completed_videos'],name,row['context_input'],flush=True)
            raw.unlink();source.unlink();del images,windows,scorer,feature_view
            cache_path.unlink(missing_ok=True)
        pool.shutdown(wait=True);pool=None
    assert len(predictions)==290 and all(all(a in p for a in policy['arms']) for p in predictions.values())
    status('evaluating');publish()
    records=[dict(name=r['name'],category=r['category'],fps=r['fps'],n_frames=r['n_frames'],intervals=r['intervals']) for r in policy['records']]
    rng=np.random.default_rng(42);weights=rng.multinomial(290,np.ones(290)/290,size=1000)
    reports={};samples={}
    for arm in policy['evaluation_arms']:reports[arm],samples[arm]=evaluate(records,predictions,arm,weights)
    contrasts={}
    for a,b in [('storyboard','baseline')]:
        contrasts[a+'_minus_'+b]={k:dict(difference=reports[a]['metrics'][k]['value']-reports[b]['metrics'][k]['value'],ci95=interval(samples[a][k]-samples[b][k])) for k in ['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap','video_max_auroc']}
    tokens=[p['context_input']['image_tokens'] for p in predictions.values()]
    result=dict(status='complete',scope=policy['scope'],formal_test=True,fingerprint=policy['fingerprint'],
        reports=reports,paired_contrasts=contrasts,registered_policy_sha256=policy['fingerprint'],
        input_summary=dict(min_image_tokens=min(tokens),max_image_tokens=max(tokens),local_video_tokens=196,
                           uniform_sampling=True,original_aspect_ratio=True,context_module_sha256=policy['context_module_sha256']),
        cache_audit=dict(baseline_reused=True,baseline_source=policy['baseline_source'],visual_feature_hits=sum(p.get('visual_cache_reused',False) for p in predictions.values()),native_fallbacks=sum(not p.get('visual_cache_reused',False) for p in predictions.values())),
        gpu=torch.cuda.get_device_name(),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
        allocation_worker_elapsed_s=time.time()-started,optimizer_updates=0,test_used=True)
    assert all(r['videos']==290 and r['frames']==1111808 for r in reports.values())
    atomic_json(out/'result.json',result);status('finished',status='complete');publish()
    print('FORMAL_CONTEXT_COMPLETE',json.dumps({a:r['metrics']['pooled_auroc'] for a,r in reports.items()}),flush=True)
except Exception:
    status('failed',status='error',error=sanitize(traceback.format_exc()));print(state['error'],flush=True)
    try:publish()
    except Exception:pass
    raise
finally:
    if pool:pool.shutdown(wait=False,cancel_futures=True)
