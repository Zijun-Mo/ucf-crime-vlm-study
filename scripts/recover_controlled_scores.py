"""Recover only small complete score ZIPs; leave model weights and data on Drive."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import checked,configure_drive_environment
configure_drive_environment(ROOT)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-optional',action='store_true')
parser.add_argument('--jobs',nargs='+',help='Recover only these named conditions')
parser.add_argument('--formal-v2',action='store_true',help='Recover the formal primary matrix; leave historical one-epoch MIL auxiliary')
parser.add_argument('--available-only',action='store_true',help='Skip conditions without a cloud completion marker')
parser.add_argument('--force',action='store_true',help='Preserve and replace even an existing complete local bundle')
args=parser.parse_args()
registry=json.loads((ROOT/'research/inference_jobs.json').read_text())['groups']
if args.formal_v2:
    registry['formal_mil']=json.loads((ROOT/'research/formal_inference_jobs.json').read_text())['groups']['formal_mil']
jobs=[j for group in registry.values() for j in group if args.include_optional or not j.get('optional')]
if args.formal_v2:
    jobs=[j for j in jobs if j['mode'] not in ['qwen_mil','openjev_mil']]
if args.jobs:
    assert set(args.jobs)<={j['name'] for j in jobs},'Unknown or optional-excluded condition'
    jobs=[j for j in jobs if j['name'] in args.jobs]
rc=['rclone','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID',
    '--contimeout','15s','--timeout','60s','--retries','3','--log-level','ERROR']
(ROOT/'.colab-work').mkdir(exist_ok=True)
log=ROOT/'.colab-work/recovery_errors.log'
for job in jobs:
    name=job['name'];target=ROOT/'results/controlled'/name
    marker=target/'complete.json'
    if not args.force and marker.exists() and len(list((target/'predictions').glob('*.json')))==290:
        m=json.loads(marker.read_text())
        if m.get('complete') is True and m.get('windows')==37188:
            print('LOCAL_COMPLETE',name,flush=True);continue
    remote='gdrive:remaining_20261108_v1/jobs/'+name+'/results.zip'
    if args.available_only:
        try:
            checked(rc+['lsjson',remote.rsplit('/',1)[0]+'/complete.json','--stat'],timeout=60,log=log)
        except RuntimeError as error:
            if not any(term in str(error).lower() for term in ['object not found', 'directory not found']):raise
            print('NOT_YET_COMPLETE',name,flush=True);continue
    metadata=json.loads(checked(rc+['lsjson',remote,'--stat'],timeout=90,log=log).stdout)
    assert 0 < metadata['Size'] <= 10*1024**2,'Leave oversized bundle on Drive: '+remote
    raw=ROOT/'results/controlled_raw'/(name+'.zip');raw.parent.mkdir(parents=True,exist_ok=True)
    checked(rc+['copyto',remote,str(raw)],timeout=180,log=log)
    scratch=Path(tempfile.mkdtemp(prefix=name+'_',dir=ROOT/'.colab-work'))
    with zipfile.ZipFile(raw) as archive:
        assert archive.testzip() is None
        assert sum(e.file_size for e in archive.infolist()) <= 30*1024**2
        for entry in archive.namelist():
            assert (scratch/entry).resolve().is_relative_to(scratch.resolve())
        archive.extractall(scratch)
    m=json.loads((scratch/'complete.json').read_text())
    c=json.loads((scratch/'config.json').read_text())
    assert m=={'complete':True,'fingerprint':c['fingerprint'],'videos':290,'windows':37188,'frames':1111808}
    assert len(list((scratch/'predictions').glob('*.json')))==290
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=ROOT/'.colab-work/superseded_scores'/(name+'_'+stamp)
        backup.parent.mkdir(parents=True,exist_ok=True);target.rename(backup)
    scratch.rename(target)
    print('RECOVERED_SMALL_COMPLETE',name,metadata['Size'],flush=True)
