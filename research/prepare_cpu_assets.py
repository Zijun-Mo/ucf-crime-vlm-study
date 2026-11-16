"""Prepare processors and input templates on a cheap CPU runtime before GPU use."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

work = Path('/content/ucf_project')
work.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/ucf_project.zip') as z:
    for name in z.namelist():
        if not (work / name).resolve().is_relative_to(work.resolve()):
            raise ValueError('Unsafe project archive')
    z.extractall(work)
deps = Path('/content/ucf_deps')
deps.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--target', str(deps),
                'transformers==5.10.2', 'peft==0.19.1', 'tokenizers==0.22.2',
                'av==19.0.0', 'einops==0.8.2', 'fla-core==0.5.2'], check=True, timeout=240)
sys.path.insert(0, str(deps))
sys.path.insert(0, str(work / 'src'))
import torch
from transformers import AutoProcessor
from openjev_dss.controlled_runtime import build_template

root = Path('/content/ucf_drive')
base = root / 'models/base/15852e8c16360a2fea060d615a32b45270f8a8fc'
local = Path('/content/ucf_cpu_base')
local.mkdir(exist_ok=True)
rclone = ['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf']
subprocess.run(rclone + ['copy', 'gdrive:models/base/15852e8c16360a2fea060d615a32b45270f8a8fc', str(local),
               '--exclude', '*.safetensors', '--exclude', '*.safetensors.index.json', '--transfers', '4'], check=True, timeout=180)
processor = AutoProcessor.from_pretrained(local)
processor.tokenizer.padding_side = 'right'
protocol = json.loads((work / 'research/protocol.json').read_text())
out = Path('/content/ucf_prepared')
out.mkdir(exist_ok=True)
report = {'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'tokenizers', 'av']},
          'templates': {}, 'source_code_sha256': hashlib.sha256((work / 'src/openjev_dss/controlled_runtime.py').read_bytes()).hexdigest()}
for name, duration in [('P1', 4), ('P2', 4), ('P3', 4), ('P1', 2), ('P1', 8)]:
    template = build_template(processor, protocol['prompts'][name], duration, out)
    template = {k: v for k, v in template.items() if isinstance(v, torch.Tensor)}
    key = f'{name}_{duration}s'
    torch.save(template, out / (key + '.pt'))
    report['templates'][key] = {'shapes': {k: list(v.shape) for k, v in template.items()},
                                'sha256': hashlib.sha256((out / (key + '.pt')).read_bytes()).hexdigest()}
    if key == 'P1_4s':
        original = torch.load(root / 'formal_stride1/template.pt', map_location='cpu', weights_only=True)
        for field in ['input_ids', 'attention_mask', 'mm_token_type_ids', 'video_grid_thw']:
            torch.testing.assert_close(template[field], original[field][:1], rtol=0, atol=0)
        report['P1_matches_historical_template_exactly'] = True
candidates = {word: processor.tokenizer.encode(word, add_special_tokens=False) for word in ['Yes', 'No']}
assert candidates == {'Yes': [9175], 'No': [2665]}, candidates
report['candidate_tokens'] = candidates
(out / 'cpu_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
subprocess.run(rclone + ['copy', str(out), 'gdrive:remaining_20261108_v1/preflight', '--exclude', '*.mp4'], check=True, timeout=180)
print(json.dumps(report, indent=2))
