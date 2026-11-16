import requests,json,hashlib
m=json.load(open('results/learning/manifest.json'));pairs=[(x['name'],x['test_names'][0]) for x in m['audit']['quarantined'] if x['reason']=='exact_test_file_duplicate'];s=requests.Session();s.mount('https://',requests.adapters.HTTPAdapter(max_retries=4));root='https://huggingface.co/api/datasets/mteb/ucf-crime/tree/'+m['revision']+'/videos/'
allfiles={}
for d in ['Training_Normal','Testing_Normal']:
 u=root+d+'?limit=1000'
 while u:
  r=s.get(u,timeout=50);r.raise_for_status();allfiles.update({x['path'].split('/')[-1]:x for x in r.json() if x['type']=='file'});u=r.links.get('next',{}).get('url')
base='https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/'+m['official_commit']+'/'
tr=s.get(base+'Anomaly_Train.txt',timeout=30);tr.raise_for_status();te=s.get(base+'Temporal_Anomaly_Annotation.txt',timeout=30);te.raise_for_status();train={x.strip().split('/')[-1] for x in tr.text.splitlines() if x.strip()};test={x.split()[0].split('/')[-1] for x in te.text.splitlines() if x.strip()}; print('OFFICIAL_LIST_SHA_MATCH',hashlib.sha256(tr.content).hexdigest()==m['official_train_sha256'],hashlib.sha256(te.content).hexdigest()==m['official_test_sha256'])
for a,b in pairs:
 x=allfiles[a];y=allfiles[b];print(a,b,'official',a in train,b in test,'sha_same',x['lfs']['oid']==y['lfs']['oid'],'size_same',x['size']==y['size'],'sha',x['lfs']['oid'],'size',x['size'])
