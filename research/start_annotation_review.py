"""Start only the CPU annotation-sheet producer on the existing allocation."""
import json,subprocess,sys
from pathlib import Path
handle=Path('/content/annotation_review_handle.json')
if not handle.exists():
    log=open('/content/annotation_review.log','w')
    worker=subprocess.Popen([sys.executable,'/content/prepare_annotation_review.py'],stdout=log,stderr=subprocess.STDOUT)
    handle.write_text(json.dumps(dict(pid=worker.pid,role='CPU only; JPEG preparation, no model or CUDA')))
print('ANNOTATION_REVIEW_PROCESS',handle.read_text())
