"""One-epoch native Qwen LoRA+MIL training using the frozen historical split."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

work = Path('/content/ucf_project')
sys.path.insert(0, '/content/ucf_deps')
sys.path.insert(0, str(work / 'src'))
import torch
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.controlled_training import run_single_mil
from openjev_dss.runtime import VisualStore
from openjev_dss.cloud_io import checked, sanitize
from openjev_dss.checkpoint_sync import publish_training_bundle
from openjev_dss.training_cache_gate import verify_training_cache

root = Path('/content/ucf_drive')
protocol = json.loads((work / 'research/protocol.json').read_text())
preflight = json.loads((root / 'remaining_20261108_v1/preflight/gpu_preflight.json').read_text())
assert preflight['status'] == 'passed', 'Training requires a successful, durable numerical preflight'
deadline = float(os.environ['UCF_DEADLINE'])
rc = ['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf', '--contimeout', '15s', '--timeout', '60s',
      '--retries', '3', '--low-level-retries', '10', '--retries-sleep', '2s']
out = Path('/content/ucf_qwen_mil_train')
out.mkdir(exist_ok=True)
remote = 'gdrive:remaining_20261108_v1/qwen_mil_train'
existing = root / 'remaining_20261108_v1/qwen_mil_train'
if existing.exists():
    # Resume only under the same frozen protocol.
    old = json.loads((existing / 'protocol.json').read_text())
    assert old == protocol
    checked(rc + ['copy', remote, str(out)], timeout=240)
(out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')

def sync():
    # Only this run's output namespace is mutable; preserve all historical folders.
    publish_training_bundle(out, remote, lambda command: checked(rc + command, timeout=180))


def guard():
    if time.time() > deadline - 240:
        raise TimeoutError('Reserve for checkpoints and runtime release')


torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
guard()
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
base = snapshot_download('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc', token=False,
                        allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', '*.model'], max_workers=4)
runtime = ControlledRuntime(base, protocol['prompts']['P1'], train=True)

manifest_path = root / 'weak_dss_learning_10cu_v1/manifest.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol['train_manifest_sha256']
manifest = json.loads(manifest_path.read_text())
assert manifest['counts'] == {'train': 1256, 'val': 163, 'cal': 163}
data = Path('/content/ucf_training_data')
data.mkdir(exist_ok=True)
for name in ['metadata', 'canonical']:
    target = data / name
    if not target.exists():
        target.symlink_to(root / 'weak_dss_smoke_v1' / name, target_is_directory=True)


class ReusableVisualStore(VisualStore):
    def __init__(self):
        super().__init__(runtime, data, local_cache='/content/ucf_train_feature_cache')
        self.guard = guard
        self.old = root / 'weak_dss_smoke_v1/window_cache'
        self.reuse_enabled = False

    def __call__(self, record, epoch, exhaustive=False):
        if self.reuse_enabled and not exhaustive:
            _, _, signature, target = self.descriptor(record, epoch, exhaustive)
            value = self.cache.get(self.old / target.name, signature)
            if value is not None:
                return value
        # Fresh features always persist locally, never to the read-only source.
        return super().__call__(record, epoch, exhaustive)


provider = ReusableVisualStore()
cache_gate = verify_training_cache(runtime, provider, manifest['records'], provider.old,
                                   '/content/ucf_train_cache_gate', guard=guard)
(out / 'cache_gate.json').write_text(json.dumps(cache_gate, indent=2) + '\n')
provider.reuse_enabled = cache_gate['reuse_recommended']
# Scientific resume identity excludes the transport wrapper, so upload tuning
# cannot silently change the numerical model, schedule or input sampler.
contract = {'protocol_sha256': hashlib.sha256((work/'research/protocol.json').read_bytes()).hexdigest(),
            'numerical_source_sha256': {name: hashlib.sha256((work/name).read_bytes()).hexdigest() for name in [
                'src/openjev_dss/controlled_runtime.py', 'src/openjev_dss/controlled_training.py',
                'src/openjev_dss/runtime.py', 'src/openjev_dss/learning.py', 'src/openjev_dss/training.py',
                'src/openjev_dss/losses.py', 'src/openjev_dss/data.py', 'src/openjev_dss/training_cache_gate.py']},
            'torch': str(torch.__version__), 'initial_trainable_parameters': runtime.trainable_parameters}
contract_path = out/'training_contract.json'
if contract_path.exists():
    assert json.loads(contract_path.read_text()) == contract, 'Scientific training contract changed; do not combine resumed optimizer states'
else:
    assert not (out/'status.json').exists(), 'A previous training state has no numerical contract; inspect it before reuse'
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')

receipt = {'started': time.time(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
           'runtime_code_sha256': hashlib.sha256((work / 'src/openjev_dss/controlled_runtime.py').read_bytes()).hexdigest(),
           'training_code_sha256': hashlib.sha256((work / 'src/openjev_dss/controlled_training.py').read_bytes()).hexdigest(),
           'trainable_parameters': runtime.trainable_parameters, 'cache_reuse_allowed': provider.reuse_enabled,
           'historical_cache_numerically_compatible': cache_gate['cache_reuse_allowed'],
           'cache_gate': 'cache_gate.json',
           'reuse_scope': 'sampled training bags only; validation features are regenerated',
           'prior_two_window_preflight_allowed_cache': preflight['cache_reuse_allowed'],
           'template_sha256': runtime.template_sha256, 'seed': 42, 'test_used': False,
           'window_sampling': 'up to 16 per training bag; short videos use all available starts; original unpadded vision tail',
           'checkpoint_publication': 'bulk data first; status pointer last; two checkpoint generations'}
(out / 'runtime_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
sync()
result = run_single_mil(runtime, manifest, provider, out, deadline, sync)
receipt.update(status=result['status'], elapsed_s=time.time() - receipt['started'],
               peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
(out / 'runtime_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
sync()
print(json.dumps(receipt, indent=2))
assert result['status'] == 'complete', 'Training not complete; resume its saved checkpoint'
