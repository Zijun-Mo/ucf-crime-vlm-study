"""Remote CPU images for predetermined descriptive cases; no model inference."""
import hashlib,json,math,zipfile
from pathlib import Path
import cv2,requests,numpy as np
from PIL import Image,ImageDraw
root=Path('/content/followup_case_images');root.mkdir(exist_ok=True);cv2.setNumThreads(1)
requests_data=json.loads(Path('/content/followup_case_requests.json').read_text());inventory=[]
for row in requests_data['records']:
    source=root/row['name'];url='https://huggingface.co/datasets/mteb/ucf-crime/resolve/'+requests_data['dataset_revision']+'/'+row['repo_path']
    response=requests.get(url,stream=True,timeout=(20,90));response.raise_for_status()
    with source.open('wb') as out:
        for chunk in response.iter_content(1024*1024):out.write(chunk)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
    duration=row['n_frames']/row['fps'];times=list(np.linspace(0,max(0,duration-.1),6))
    for a,b in row['intervals']:
        if a<1 or b<a:continue
        start=(a-1)/row['fps'];end=b/row['fps'];times.extend([max(0,start-1),start,(start+end)/2,end,min(duration-.1,end+1)])
    times=sorted(set(round(float(t),2) for t in times));canvas=Image.new('RGB',(1440,math.ceil(len(times)/4)*230),'white');draw=ImageDraw.Draw(canvas)
    cap=cv2.VideoCapture(str(source));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==row['n_frames']
    for i,t in enumerate(times):
        frame=min(row['n_frames']-1,round(t*row['fps']));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((358,202));x=i%4*360;y=i//4*230
        canvas.paste(im,(x,y+26));draw.text((x+3,y+3),f'{row["name"]} t={frame/row["fps"]:.2f}s f={frame+1}',fill='black')
    cap.release();target=root/(source.stem+'.jpg');canvas.save(target,quality=90);source.unlink()
    inventory.append(dict(name=row['name'],source_sha256=row['sha256'],image=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),times_s=times))
(root/'manifest.json').write_text(json.dumps(dict(status='complete',selection_rule=requests_data['selection_rule'],records=inventory),indent=2)+'\n')
with zipfile.ZipFile('/content/followup_case_images.zip','w',zipfile.ZIP_STORED) as z:
    for path in root.iterdir():
        if path.suffix in ['.jpg','.json']:z.write(path,path.name)
print('FOLLOWUP_CASE_IMAGES_COMPLETE',len(inventory),flush=True)
