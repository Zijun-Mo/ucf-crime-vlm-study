"""CPU-only exact RGB-input comparison against the original decoder."""
import hashlib,importlib.util,inspect,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
started=time.time();report={'status':'running','new_gpu_allocation':False,'model_forward_calls':0,'training_steps':0}
out=Path('/content/ucf_stream_input_probe.json')
try:
    payload=json.loads(Path('/content/ucf_stream_probe_payload.json').read_text())
    data=Path('/content/ucf_stream_input_data');data.mkdir(exist_ok=True);canonical=data/'canonical';canonical.mkdir(exist_ok=True)
    listing=data/'files.txt';listing.write_text('\n'.join(r['name'] for r in payload['fixtures'])+'\n')
    rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--log-level','ERROR',
        '--contimeout','15s','--timeout','60s','--retries','2']
    subprocess.run(rc+['copy','gdrive:weak_dss_smoke_v1/canonical',str(canonical),'--files-from',str(listing),
                       '--transfers','4','--checkers','4'],check=True,timeout=180)
    for r in payload['fixtures']:
        with (canonical/r['name']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==payload['metadata'][r['name']]['canonical_sha256']
    path=Path('/content/formal_features_stream_candidate.py')
    spec=importlib.util.spec_from_file_location('openjev_dss.stream_candidate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate=module.stream_input_gate(canonical,payload['metadata'],payload['fixtures'])
    assert gate['allowed']
    report.update(status='passed',gate=gate,fixtures=[r['name'] for r in payload['fixtures']],
        pair_batches_sha256=hashlib.sha256(inspect.getsource(module.pair_batches).encode()).hexdigest(),
        original_decoder_sha256=hashlib.sha256(Path('/content/ucf_project/src/openjev_dss/runtime.py').read_bytes()).hexdigest(),
        bytes=sum((canonical/r['name']).stat().st_size for r in payload['fixtures']))
except Exception:
    report.update(status='error',error=traceback.format_exc())
report['elapsed_s']=time.time()-started;out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
