"""Frozen 0-shot overview pilot; all video/weight bytes stay on the L4 VM."""
import concurrent.futures,hashlib,json,os,sys,time,traceback,zipfile
from pathlib import Path
work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import cv2,numpy as np,requests,torch
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.learning import atomic_json
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.storyboard_context import make_sheets,NativeContextScorer,CONTEXT_INSTRUCTION
from openjev_dss.study_pair_visuals import _pairs,_windows
from openjev_dss.experiment import score_windows
from openjev_dss.study_evaluation import evaluate,interval

out=Path('/content/ucf_storyboard');out.mkdir(exist_ok=True)
(out/'predictions').mkdir(exist_ok=True)
policy=json.loads((work/'research/storyboard_context_v1/policy.json').read_text())
deadline=float(os.environ['UCF_DEADLINE']);remote=policy['output_remote']
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','45s','--retries','2','--retries-sleep','2s']
state=dict(status='running',stage='initializing',completed_videos=0,total_videos=len(policy['records']),scored_windows=0,optimizer_updates=0,started=time.time(),test_used=False)

def status(stage,**extra):
    state.update(stage=stage,updated=time.time(),**extra);atomic_json(out/'status.json',state)
def guard():
    if time.time()>deadline-180:raise TimeoutError('Overview experiment durable-save reserve')
def publish():
    guard();checked(rc+['copy',str(out),remote,'--transfers','8','--checkers','8','--checksum'],timeout=180)
def download_original(record):
    path=Path('/content/storyboard_original')/record['name'];path.parent.mkdir(exist_ok=True)
    url='https://huggingface.co/datasets/mteb/ucf-crime/resolve/'+policy['dataset_revision']+'/'+record['repo_path']
    for attempt in range(3):
        try:
            if not path.exists():
                response=requests.get(url,stream=True,timeout=(20,60));response.raise_for_status()
                temp=path.with_suffix('.partial')
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
    atomic_json(out/'policy.json',policy)
    status('staging_canonical')
    data=Path('/content/storyboard_canonical');data.mkdir(exist_ok=True);archive=data/'canonical.zip'
    asset=json.loads((work/'research/formal_input_archive.json').read_text())
    checked(rc+['copyto',asset['remote'],str(archive)],timeout=600)
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==asset['zip_sha256']
    with zipfile.ZipFile(archive) as z:
        for r in policy['records']:z.extract(r['name'],data)
    archive.unlink()
    status('staging_originals')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        original=dict(zip([r['name'] for r in policy['records']],pool.map(download_original,policy['records'])))
    status('loading_model')
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import snapshot_download
    base=snapshot_download(policy['model_id'],revision=policy['revision'],token=False,
        allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'],max_workers=4)
    runtime=ControlledRuntime(base,policy['prompt'],train=False)
    assert runtime.trainable_parameters==0
    runtime.model.eval();runtime.head.eval()
    template_video=Path('/content/ucf_templates/template_4s.mp4')
    predictions={};reference_checks=[];native_checks=[];image_tokens=[]
    for number,record in enumerate(policy['records']):
        guard();name=record['name'];m=policy['metadata'][name]
        status('scoring',active_video=name,active_arm='baseline')
        source=data/name
        with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['canonical_sha256']
        cap=cv2.VideoCapture(str(source));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release();assert len(frames)==m['sampled_frames']+1
        windows=[np.stack([frames[min(2*t+j,m['sampled_frames'])] for j in range(8)]) for t in range(m['window_count'])]
        with torch.inference_mode():
            baseline=NativeContextScorer(runtime,[],policy['prompt'],template_video,context=False)
            z0=baseline.score(windows,guard=guard)
            state['scored_windows']+=len(z0)
            images,layout=make_sheets(original[name],out/'sheets'/name)
            status('scoring',active_arm='prompt_only')
            text_control=NativeContextScorer(runtime,[],policy['prompt'],template_video)
            ztext=text_control.score(windows,guard=guard);state['scored_windows']+=len(ztext)
            status('scoring',active_arm='storyboard')
            contextual=NativeContextScorer(runtime,images,policy['prompt'],template_video);info=contextual.info
            assert info['image_tokens']>0;image_tokens.append(info['image_tokens'])
            if number==0:
                native_checks.append(dict(name=name,execution='All arms invoke official multimodal forward directly; no manually injected image embeddings',image_tokens=info['image_tokens']))
            zcontext=contextual.score(windows,guard=guard);state['scored_windows']+=len(zcontext)
        row=dict(name=name,source_sha256=record['sha256'],start_s=list(range(len(z0))),
                 baseline=z0,prompt_only=ztext,storyboard=zcontext,layout=layout,context_input=info)
        predictions[name]=row;atomic_json(out/'predictions'/f'{name}.json',row)
        status('scoring',completed_videos=number+1)
        if (number+1)%4==0:publish()
        del frames,windows,images,baseline,text_control,contextual;original[name].unlink();source.unlink()
        print('OVERVIEW_VIDEO',number+1,name,info,flush=True)
    status('evaluating')
    records=[];ann={r['name']:r for r in policy['annotations']}
    for r in policy['records']:
        m=policy['metadata'][r['name']]
        records.append(dict(name=r['name'],category=r['category'],fps=m['fps'],n_frames=m['n_frames'],
                            intervals=ann[r['name']]['intervals'] if r['label'] else []))
    rng=np.random.default_rng(42);weights=rng.multinomial(len(records),np.ones(len(records))/len(records),size=1000)
    reports={};samples={}
    for arm in ['baseline','prompt_only','storyboard']:
        reports[arm],samples[arm]=evaluate(records,predictions,arm,weights)
    contrasts={}
    for a,b in [('storyboard','baseline'),('prompt_only','baseline'),('storyboard','prompt_only')]:
        contrasts[a+'_minus_'+b]={key:dict(difference=reports[a]['metrics'][key]['value']-reports[b]['metrics'][key]['value'],
            ci95=interval(samples[a][key]-samples[b][key])) for key in ['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap','video_max_auroc']}
    result=dict(status='complete',scope='44-video development pilot; duration-matched normal videos; not a full test-set result',
        reports=reports,paired_contrasts=contrasts,native_execution_audit=native_checks,
        input_summary=dict(min_image_tokens=min(image_tokens),max_image_tokens=max(image_tokens),
            local_video_tokens=196,uniform_sampling=True,original_aspect_ratio=True),
        gpu=torch.cuda.get_device_name(),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
        elapsed_s=time.time()-state['started'],optimizer_updates=0,test_used=False)
    atomic_json(out/'result.json',result);status('finished',status='complete');publish()
    print('OVERVIEW_RESULT',json.dumps({k:{m:d['value'] for m,d in v['metrics'].items()} for k,v in reports.items()}),flush=True)
except Exception:
    status('failed',status='error',error=sanitize(traceback.format_exc()));print(state['error'],flush=True)
    try:publish()
    except Exception:pass
    raise
