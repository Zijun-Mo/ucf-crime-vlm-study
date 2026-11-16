"""Remote CPU-only 2fps boundary evidence on the original video timeline."""
import json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
root=Path('/content/annotation_dev_v2');data=json.loads(Path('/content/annotation_refined.json').read_text())
output=root/'boundary_sheets';output.mkdir(exist_ok=True);inventory=[]
for row in data['records']:
    cap=cv2.VideoCapture(str(root/row['name']));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps==row['fps'] and count==row['n_frames']
    intervals=row['intervals_seconds'];folder=output/row['name'];folder.mkdir(exist_ok=True)
    for page,start in enumerate(range(0,len(intervals),2)):
        pairs=intervals[start:start+2];canvas=Image.new('RGB',(1960,len(pairs)*2*190),'white');draw=ImageDraw.Draw(canvas);points=[]
        for j,(a,b) in enumerate(pairs):
            for kind,center,offset in [('start',a,0),('end',b,1)]:
                y=(j*2+offset)*190
                for col,delta in enumerate(np.arange(-1.5,1.51,.5)):
                    t=max(0.,min((count-1)/fps,center+float(delta)));frame=min(count-1,round(t*fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    gamma=.5 if row['name'].startswith('Shooting') else 1.
                    if gamma!=1.:rgb=np.clip(255*(rgb.astype(float)/255)**gamma,0,255).astype(np.uint8)
                    im=Image.fromarray(rgb);im.thumbnail((278,160));x=col*280
                    canvas.paste(im,(x,y+28));draw.text((x+2,y+2),f'{start+j+1} {kind} t={frame/fps:.2f} f={frame+1}'+(' gamma.5' if gamma!=1 else ''),fill='black')
                    points.append(dict(interval=start+j,kind=kind,frame=frame+1,time=frame/fps))
        target=folder/f'boundary_{page:02d}.jpg';canvas.save(target,quality=91)
        inventory.append(dict(name=row['name'],path=str(target.relative_to(output)),source_sha256=row['source_sha256'],points=points))
    cap.release()
(output/'inventory.json').write_text(json.dumps(dict(status='complete',videos=len(data['records']),sampling_step_s=.5,display_gamma_for_dark_shooting=.5,records=inventory),indent=2)+'\n')
import zipfile
with zipfile.ZipFile('/content/annotation_boundary_sheets.zip','w',zipfile.ZIP_STORED) as z:
    for p in output.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(output))
print('BOUNDARY_EVIDENCE_COMPLETE',len(inventory),flush=True)
