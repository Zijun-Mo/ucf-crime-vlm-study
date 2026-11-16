"""Read pinned validation videos on Colab CPU, emit timestamped review sheets."""
from pathlib import Path
import hashlib
import json
import math
import time
import zipfile
import requests
import cv2
from PIL import Image,ImageDraw

ROOT=Path('/content/localization_dev');ROOT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'candidates.json').read_text())
review=ROOT/'sheets';review.mkdir(exist_ok=True)
source='https://huggingface.co/datasets/mteb/ucf-crime/resolve/8d783bb55269300114f9773a3c068b48a4940e4a/'
for number,row in enumerate(manifest['records']):
    path=ROOT/row['name']
    if not path.exists():
        response=requests.get(source+row['repo_path'],stream=True,timeout=(30,90));response.raise_for_status()
        temporary=path.with_suffix('.partial')
        with temporary.open('wb') as out:
            for chunk in response.iter_content(1024*1024):out.write(chunk)
        with temporary.open('rb') as inp:assert hashlib.file_digest(inp,'sha256').hexdigest()==row['source_sha256']
        temporary.replace(path)
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps==row['fps'] and n==row['n_frames']
    times=list(range(0,math.ceil(n/fps),2))
    for page,start in enumerate(range(0,len(times),24)):
        target=review/(path.stem+f'_{page:02d}.jpg')
        if target.exists():continue
        canvas=Image.new('RGB',(1280,6*208),'white');draw=ImageDraw.Draw(canvas)
        for j,t in enumerate(times[start:start+24]):
            frame=min(round(t*fps),n-1);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((318,180))
            x=(j%4)*320;y=(j//4)*208;canvas.paste(im,(x,y+25))
            draw.text((x+3,y+3),f'{path.stem} {t}s f{frame+1}',fill='black')
        canvas.save(target,quality=90)
    cap.release()
    (ROOT/'status.json').write_text(json.dumps(dict(status='running',completed=number+1,name=row['name'],total=len(manifest['records']))))
    print('REVIEW_SHEETS',number+1,row['name'],flush=True)
with zipfile.ZipFile(ROOT/'review_sheets.zip','w',zipfile.ZIP_STORED) as archive:
    for path in sorted(review.glob('*.jpg')):archive.write(path,path.name)
(ROOT/'status.json').write_text(json.dumps(dict(status='complete',completed=len(manifest['records']),sheets=len(list(review.glob('*.jpg'))))))
print('REVIEW_PACK_COMPLETE',flush=True)
