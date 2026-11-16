import requests,hashlib,json,time
pairs=[('Normal_Videos221_x264.mp4','Normal_Videos_597_x264.mp4','29f926b752ed9bdf2108def4542aabd43d18587fa600bb100ac0cea794f40e0a'),('Normal_Videos594_x264.mp4','Normal_Videos_217_x264.mp4','73684bf2ee4e9c825c0a019e245a7bf94208ac8c9e2a3a20e995e3ceddbefa93')]
root='https://huggingface.co/datasets/mteb/ucf-crime/resolve/8d783bb55269300114f9773a3c068b48a4940e4a/videos/'
results=[]
for a,b,expected in pairs:
 row=[]
 for folder,n in [('Training_Normal',a),('Testing_Normal',b)]:
  h=hashlib.sha256();size=0
  with requests.get(root+folder+'/'+n,stream=True,timeout=(20,90)) as r:
   r.raise_for_status()
   for chunk in r.iter_content(1024*1024):h.update(chunk);size+=len(chunk)
  digest=h.hexdigest();row.append({'name':n,'sha256':digest,'bytes':size});assert digest==expected
 results.append(row);print(json.dumps(row),flush=True)
print('ALL_PASS',len(results),flush=True)
