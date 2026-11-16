"""Persist compiler artifacts, never activations or trainable-weight caches."""
import hashlib,importlib.metadata,json,os,shutil,subprocess
from pathlib import Path

def environment():
    import torch
    values={'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())}
    for package in ['triton','fla-core','transformers']:
        values[package]=importlib.metadata.version(package)
    driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True)
    values['driver']=driver.stdout.strip();return values

def paths(root,local):
    env=environment();key=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()
    return env,Path(root)/key,Path(local)

def restore_kernel_cache(root,local='/content/dss_formal_triton'):
    env,persistent,scratch=paths(root,local);scratch.mkdir(parents=True,exist_ok=True)
    manifest=persistent/'environment.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==env
        shutil.copytree(persistent/'cache',scratch,dirs_exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(scratch)
    return manifest.exists()

def save_kernel_cache(root,local=None):
    env,persistent,scratch=paths(root,local or os.environ['TRITON_CACHE_DIR'])
    persistent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(scratch,persistent/'cache',dirs_exist_ok=True)
    (persistent/'environment.json').write_text(json.dumps(env,indent=2));return env
