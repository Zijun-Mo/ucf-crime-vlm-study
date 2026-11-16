"""Remote CPU extraction of fixed descriptive cases from pinned model inputs."""
import json,hashlib,subprocess,os,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
config=json.loads(Path('/content/v4_case_config.json').read_text());out=Path('/content/v4_case_frames');out.mkdir(exist_ok=True)
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','10s','--timeout','30s','--retries','2']
report=dict(source='pinned224x224 2fps canonical model inputs, not full-resolution source videos',
    selector='8 uniform points, first/last annotation boundaries +/-1s, and zero-shot/V4 highest-score window centers; nearest canonical frame',cases=[])
for item in config['cases']:
    row=item['record'];meta=item['metadata'];name=row['name'];path=Path('/content')/name
    subprocess.run(rc+['copyto',config['canonical_remote']+'/'+name,str(path)],check=True,timeout=120)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['canonical_sha256']
    cap=cv2.VideoCapture(str(path));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert abs(fps-2.)<1e-6
    duration=row['n_frames']/row['fps'];intervals=[(a,b) for a,b in row['intervals'] if a>=1 and b>=a]
    assert intervals
    begin=(intervals[0][0]-1)/row['fps'];end=intervals[-1][1]/row['fps']
    times=list(np.linspace(0,max(0,duration-.5),8))+[begin-1,begin+1,end-1,end+1]+item['peak_window_centers_s']
    indices=sorted({min(count-2,max(0,round(float(t)*fps))) for t in times})
    canvas=Image.new('RGB',(960,52+len(range(0,len(indices),4))*260),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,6),f'{item["role"]}: {name}',fill='black');draw.text((8,26),'Canonical model input; red border = benchmark anomaly at sampled time',fill='black');frames=[]
    for i,index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
        t=index/fps;original=min(row['n_frames'],round(t*row['fps'])+1)
        anomaly=any(a<=original<=b for a,b in intervals);x=(i%4)*240;y=52+(i//4)*260
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((224,224));canvas.paste(im,(x+8,y+27))
        draw.rectangle((x+5,y+24,x+235,y+254),outline='#bf3a42' if anomaly else '#808080',width=3)
        draw.text((x+8,y+5),f't={t:.1f}s | '+('GT anomaly' if anomaly else 'GT background'),fill='black')
        frames.append(dict(canonical_frame=index,time_s=t,original_frame=original,benchmark_anomaly=anomaly))
    cap.release();path.unlink();target=out/(name+'.jpg');canvas.save(target,quality=90)
    report['cases'].append(dict(name=name,role=item['role'],image=target.name,image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_sha256=row['sha256'],canonical_sha256=meta['canonical_sha256'],sampled_frames=frames))
(out/'manifest.json').write_text(json.dumps(report,indent=2))
subprocess.run(rc+['copy',str(out),config['output_remote'],'--checksum','--transfers','4'],check=True,timeout=120)
print('CASE_FRAMES_COMPLETE',len(report['cases']),flush=True)
