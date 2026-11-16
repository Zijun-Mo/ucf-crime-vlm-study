"""CPU only: stream 4 pinned mirror videos to Colab, emit small contact sheets."""
from pathlib import Path
import requests,cv2,numpy as np
from PIL import Image,ImageDraw
root='https://huggingface.co/datasets/mteb/ucf-crime/resolve/8d783bb55269300114f9773a3c068b48a4940e4a/videos/'
cases={'Stealing019_x264.mp4':('Stealing',[10,14,16,30,88,91,92,94,136,140,143,146]),'RoadAccidents009_x264.mp4':('RoadAccidents',[3,5,6,7,7.5,8,9,11,20,24,26,29]),'Shooting013_x264.mp4':('Shooting',[0,1,2,4,10,20,27,28.7,30,31.3,32,34]),'Explosion004_x264.mp4':('Explosion',[0,1,2.5,4,6,7.4,9,15,30,45,47,50])}
for name,(category,times) in cases.items():
 path=Path('/content/'+name);response=requests.get(root+category+'/'+name,stream=True,timeout=(30,90));response.raise_for_status()
 with path.open('wb') as f:
  for chunk in response.iter_content(1024*1024):f.write(chunk)
 cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));canvas=Image.new('RGB',(1280,3*206),'white');draw=ImageDraw.Draw(canvas)
 for i,t in enumerate(times):
  cap.set(cv2.CAP_PROP_POS_FRAMES,min(round(t*fps),frames-1));ok,frame=cap.read();assert ok,(name,t)
  im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((310,176));x=(i%4)*320;y=(i//4)*206;canvas.paste(im,(x,y+22));draw.text((x+4,y+3),f'{name}  t={t:.1f}s',fill='black')
 cap.release();path.unlink();out='/content/'+name+'.jpg';canvas.save(out,quality=88);print(name,'fps',fps,'frames',frames,'output',out,flush=True)
