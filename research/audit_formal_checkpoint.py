import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
import torch
from openjev_dss.controlled_training import tensor_digest
root=Path('/content/ucf_formal_train');state=json.loads((root/'status.json').read_text());report={'generation':state['generation'],'fingerprint':state['fingerprint'],'test_used':state['test_used'],'arms':{}}
for name,arm in state['arms'].items():
 path=root/name/f"resume_{state['generation']%2}.pt"
 saved=torch.load(path,map_location='cpu',weights_only=True)
 digest=tensor_digest(saved['trainable']);steps=[float(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s]
 assert saved['steps']==arm['steps'] and digest!=arm['initial_digest'] and steps
 assert min(steps)==max(steps)==saved['steps']
 report['arms'][name]={'steps':saved['steps'],'initial_trainables_sha256':arm['initial_digest'],'current_trainables_sha256':digest,'parameters_changed':True,'optimizer_state_entries':len(steps),'optimizer_step_min':min(steps),'optimizer_step_max':max(steps),'optimizer_group_lrs':[g['lr'] for g in saved['optimizer']['param_groups']],'checkpoint_md5':hashlib.md5(path.read_bytes()).hexdigest(),'checkpoint_bytes':path.stat().st_size,'weights_downloaded_to_desktop':False}
print(json.dumps(report,indent=2))
