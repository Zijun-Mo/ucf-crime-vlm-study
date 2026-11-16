"""Colab CPU: make small timestamped sheets; originals never leave Colab."""
import hashlib,json,math,sys,time
from pathlib import Path
import cv2,requests
from PIL import Image,ImageDraw

root=Path('/content/annotation_dev_v2');root.mkdir(exist_ok=True)
candidates=json.loads(Path('/content/annotation_candidates.json').read_text())
for row in candidates['records']:
    path=root/row['name'];url='https://huggingface.co/datasets/mteb/ucf-crime/resolve/'+candidates['dataset_revision']+'/'+row['repo_path']
    if not path.exists():
        for attempt in range(3):
            try:
                response=requests.get(url,stream=True,timeout=(20,60));response.raise_for_status()
                with path.open('wb') as out:
                    for chunk in response.iter_content(1024*1024):out.write(chunk)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];break
            except Exception:
                path.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2)
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps==row['fps'] and n==row['n_frames']
    # Coarse sheets cover the entire video; separate requests later refine
    # every proposed boundary and adjacent normal context at 2fps.
    step=max(2,int(math.ceil((n/fps)/72)))
    times=list(range(0,math.ceil(n/fps),step))
    folder=root/'sheets'/row['name'];folder.mkdir(parents=True,exist_ok=True)
    for page,start in enumerate(range(0,len(times),24)):
        output=folder/f'overview_{page:02d}.jpg'
        if output.exists():continue
        canvas=Image.new('RGB',(1440,6*228),'white');draw=ImageDraw.Draw(canvas)
        for j,t in enumerate(times[start:start+24]):
            frame=min(n-1,round(t*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((358,202));x=(j%4)*360;y=(j//4)*228
            canvas.paste(im,(x,y+24));draw.text((x+3,y+3),f'{row["name"]} t={t}s f={frame+1}',fill='black')
        canvas.save(output,quality=88)
    cap.release();(root/'status.json').write_text(json.dumps(dict(status='running',completed=row['name'])))
(root/'status.json').write_text(json.dumps(dict(status='complete',videos=len(candidates['records']))))
print('ANNOTATION_SHEETS_COMPLETE',flush=True)
