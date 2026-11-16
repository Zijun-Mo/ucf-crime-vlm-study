"""Read an immutable epoch checkpoint on the existing runtime; no optimizer writes."""
import hashlib,json,os,sys,time
from pathlib import Path
import subprocess
sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
import torch
from openjev_dss.controlled_training import tensor_digest
torch.set_num_threads(2)
epoch=int(os.environ['UCF_REVIEW_EPOCH']);out=Path('/content/ucf_localization_train')
folder=out/'epochs'/f'epoch_{epoch}';metrics=json.loads((folder/'metrics.json').read_text())
path=folder/'state.pt';state=torch.load(path,map_location='cpu',weights_only=True)
actual=tensor_digest(state);assert actual==metrics['state_sha256']
predictions=list((out/'validation'/f'epoch_{epoch}').glob('*.json'));predictions=[p for p in predictions if p.name!='metrics.json']
assert len(predictions)==163
assert all(json.loads(p.read_text())['state_sha256']==actual for p in predictions)
report=dict(passed=True,epoch=epoch,actual_tensor_sha256=actual,
    actual_checkpoint_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_bytes=path.stat().st_size,
    validation_videos=len(predictions),weights_kept_remote=True,optimizer_unmodified=True,time=time.time(),
    epoch_metrics_mtime=(folder/'metrics.json').stat().st_mtime,
    previous_epoch_metrics_mtime=(out/'epochs'/f'epoch_{epoch-1}'/'metrics.json').stat().st_mtime)
target=out/f'epoch{epoch}_checkpoint_payload_audit.json';target.write_text(json.dumps(report,indent=2)+'\n')
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','2']
subprocess.run(rc+['copyto',str(target),os.environ['UCF_REVIEW_REMOTE']+'/'+target.name,'--checksum'],check=True,timeout=120)
print('READ_ONLY_EPOCH_CHECKPOINT_VERIFIED',epoch,actual)
