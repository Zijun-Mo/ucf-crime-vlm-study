"""Read-only preflight for the project Colab runtime; never allocates a GPU."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

root = Path('/content/drive/MyDrive/OpenJev_UCFCrime')
report = {'time': time.time(), 'python': platform.python_version(),
          'drive_mounted': Path('/content/drive/MyDrive').is_dir(),
          'project_available': root.is_dir(),
          'disk_free_gib': shutil.disk_usage('/content').free / 1024**3, 'versions': {}}
for name in ['torch', 'transformers', 'peft', 'fla-core', 'triton', 'numpy', 'opencv-python', 'scikit-learn']:
    try:
        report['versions'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['versions'][name] = None
for relative in ['formal_stride1/template.pt', 'formal_stride1/config.json',
                 'weak_dss_learning_10cu_v1/training/A/best_state.pt',
                 'weak_dss_learning_10cu_v1/manifest.json',
                 'weak_dss_smoke_v1/canonical', 'weak_dss_smoke_v1/window_cache']:
    p = root / relative
    report[relative] = {'exists': p.exists(), 'bytes': p.stat().st_size if p.is_file() else None}
Path('/content/ucf_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
