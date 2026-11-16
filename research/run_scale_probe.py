"""Bounded scale memory/throughput check using only fixed training fixtures."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

begin = time.time()
work = Path('/content/ucf_project')
sys.path.insert(0, '/content/ucf_deps')
sys.path.insert(0, str(work / 'src'))
import torch
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.runtime import decode_canonical_frames
from openjev_dss.study_pair_visuals import GatedPairVisualStore
from openjev_dss.cloud_io import checked, sanitize

protocol = json.loads((work / 'research/protocol.json').read_text())
job = json.loads((work / 'research/inference_jobs.json').read_text())['groups'][os.environ['UCF_GROUP']][0]
resolution = json.loads((work / 'research/model_resolution.json').read_text())[job['model_id']]
deadline = float(os.environ['UCF_DEADLINE'])
path = Path('/content/ucf_scale_probe.json')
report = {'status': 'running', 'job': job, 'started': begin, 'test_labels_used': False,
          'dtype': 'bf16', 'batch': 4, 'prefix_cache': False,
          'model_weight_bytes': sum(r['size'] for r in resolution['weight_files']),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def persist():
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)

def guard():
    if time.time() > deadline - 120:
        raise TimeoutError('Scale probe time reserve reached')

try:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    report.update(stage='public_model_download', gpu=torch.cuda.get_device_name(),
                  gpu_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3,
                  versions={name: importlib.metadata.version(name) for name in ['torch','transformers','peft','fla-core','triton']})
    persist()
    guard()
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    start = time.time()
    base = Path(snapshot_download(job['model_id'], revision=job['revision'], token=False,
            allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'], max_workers=4))
    report['download_s'] = time.time() - start
    # Verify the actual downloaded model, without exposing any credentials.
    report['weight_checks'] = []
    for file in resolution['weight_files']:
        guard()
        source = base / file['name']
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        assert source.stat().st_size == file['size'] and digest.hexdigest() == file['lfs']['sha256']
        report['weight_checks'].append({'file': file['name'], 'sha256': digest.hexdigest(), 'size': file['size']})
    report['stage'] = 'load_model'
    persist()
    guard()
    start = time.time()
    runtime = ControlledRuntime(str(base), protocol['prompts']['P1'], 4)
    runtime.model.eval()
    report.update(load_s=time.time() - start, template_sha256=runtime.template_sha256)
    root = Path('/content/ucf_drive')
    manifest = json.loads((root / 'weak_dss_learning_10cu_v1/manifest.json').read_text())
    assert hashlib.sha256((root / 'weak_dss_learning_10cu_v1/manifest.json').read_bytes()).hexdigest() == protocol['train_manifest_sha256']
    fixtures = [min((r for r in manifest['records'] if r['split']=='train' and r['label']==y),
                    key=lambda r:(r['size'],r['name'])) for y in [0,1]]
    metadata = {r['name']: {**json.loads((root / 'weak_dss_smoke_v1/metadata' / (r['name']+'.json')).read_text()), 'source_sha256':r['sha256']} for r in fixtures}
    report.update(stage='train_fixture_probe', fixtures=[r['name'] for r in fixtures])
    persist()
    features=[]
    with torch.inference_mode():
        for record in fixtures:
            guard()
            local=Path('/content') / record['name']
            shutil.copyfile(root / 'weak_dss_smoke_v1/canonical' / record['name'], local)
            m=metadata[record['name']]
            assert hashlib.sha256(local.read_bytes()).hexdigest()==m['canonical_sha256']
            frames=decode_canonical_frames(local,m['sampled_frames'],[0],True)
            starts=list(range(min(4,m['window_count'])))
            windows=[np.stack([frames[min(2*s+j,m['sampled_frames'])] for j in range(8)]) for s in starts]
            windows += [windows[-1]]*(4-len(windows))
            torch.cuda.synchronize();start=time.time()
            f=runtime.vision(windows)
            torch.cuda.synchronize()
            report.setdefault('vision_batch4_s',[]).append(time.time()-start)
            assert torch.isfinite(f).all()
            features.append(f)
            local.unlink()
        provider=GatedPairVisualStore(runtime,root/'weak_dss_smoke_v1/canonical',metadata,
                                     '/content/ucf_scale_gate',job['revision'],guard=guard)
        report['pair_feature_gate']=provider.gate(fixtures,4)
        for _ in range(3):runtime.forward(features[0])
        torch.cuda.synchronize()
        elapsed=[]
        for _ in range(10):
            guard();start=time.time()
            score=runtime.forward(features[_%len(features)])
            torch.cuda.synchronize()
            assert torch.isfinite(score).all()
            elapsed.append(time.time()-start)
    guard()
    upload = Path('/content/ucf_scale_upload_fixture.pt')
    torch.save({'signature': {'training_fixture_probe': True}, 'features': torch.cat(features).repeat(2,1,1)}, upload)
    upload_started = time.time()
    checked(['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','copyto',str(upload),
             'gdrive:remaining_20261108_v1/preflight/io_bench/'+job['name']+'_'+report['gpu'].replace(' ','_')+'.pt',
             '--contimeout','15s','--timeout','30s','--retries','2'],timeout=60)
    report['upload_benchmark'] = {'source_bytes': upload.stat().st_size,
        'trials': [{'mode':'single_request','exit_code':0,'seconds':time.time()-upload_started}]}
    upload.unlink()
    throughput=40/sum(elapsed)
    report.update(status='passed',stage='complete',language_windows_s=throughput,
                  language_batch4_seconds=elapsed, full_test_language_estimate_s=37188/throughput,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                  elapsed_s=time.time()-begin,
                  note='Training fixtures only; full-test wall time must additionally include vision, I/O and setup')
except Exception:
    report.update(status='error', error=sanitize(traceback.format_exc()), elapsed_s=time.time()-begin)
finally:
    persist()
    checked(['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','copyto',str(path),
             'gdrive:remaining_20261108_v1/preflight/'+job['name']+'_scale_'+report.get('gpu','unknown').replace(' ','_')+'.json',
             '--contimeout','15s','--timeout','30s','--retries','2'],timeout=60)
print(json.dumps(report,indent=2),flush=True)
assert report['status']=='passed', 'Scale probe failed; inspect saved evidence before a full run'
