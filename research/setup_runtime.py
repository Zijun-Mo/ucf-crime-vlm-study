"""Install the tested minimal stack on a fresh Colab runtime."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

work = Path('/content/ucf_project')
work.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/ucf_project.zip') as archive:
    for name in archive.namelist():
        assert (work / name).resolve().is_relative_to(work.resolve())
    archive.extractall(work)
deps = Path('/content/ucf_deps')
deps.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--target', str(deps),
                'transformers==5.10.2', 'peft==0.19.1', 'tokenizers==0.22.2',
                'av==19.0.0', 'einops==0.8.2', 'fla-core==0.5.2'], check=True, timeout=240)
sys.path.insert(0, str(deps))
sys.path.insert(0, str(work / 'src'))
import torch
assert torch.cuda.is_available()
report = {'gpu': torch.cuda.get_device_name(), 'gpu_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
          'versions': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'triton', 'fla-core']}}
required = json.loads((work / 'research/gpu_preflight_result.json').read_text())['versions']
report['ready'] = report['versions'] == required
report['required_versions'] = required
if os.environ.get('UCF_GROUP') == 'train_formal':
    reference = json.loads((work / 'research/formal_schedule_environment_reference.json').read_text())
    assert reference['status'] == 'verified_actual_colab_and_cpu_plan_match'
    report['formal_required_versions'] = reference['versions']
    report['formal_versions'] = {name: importlib.metadata.version(name) for name in reference['versions']}
    report['ready'] = report['ready'] and report['formal_versions'] == report['formal_required_versions']
Path('/content/ucf_environment_ready.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

assert report['ready'], 'Runtime differs from the recorded numerical environment; do not start a full experiment'
