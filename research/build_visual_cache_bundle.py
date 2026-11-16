"""CPU-only remote packaging of existing visual bytes, with resumable parts."""
import hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
work=Path('/content/ucf_project');sys.path.insert(0,str(work/'src'))
from openjev_dss.cache_bundle import groups,pack
config=json.loads(Path('/content/cache_bundle_config.json').read_text())
inventory=json.loads((work/config['inventory']).read_text())
assert hashlib.sha256((work/config['inventory']).read_bytes()).hexdigest()==config['inventory_sha256']
out=Path('/content/cache_bundle_work');out.mkdir(exist_ok=True)
status=Path('/content/cache_bundle_status.json');start=time.time()
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','3','--retries-sleep','2s']
def save(**kw):status.write_text(json.dumps(dict(updated=time.time(),elapsed_s=time.time()-start,**kw))+'\n')
def run(args,timeout=1800):return subprocess.run(rc+args,capture_output=True,text=True,check=True,timeout=timeout)
try:
    run(['mkdir',config['output_remote']]);parts=[];batches=groups(inventory['files'])
    save(status='running',stage='starting',parts=0,total_parts=len(batches),total_bytes=inventory['bytes'])
    for i,files in enumerate(batches):
        if time.time()>config['deadline']-180:raise TimeoutError('Cache packaging reserve')
        part_name=f'part_{i:03d}.zip';sidecar=f'part_{i:03d}.json'
        old=subprocess.run(rc+['cat',config['output_remote']+'/'+sidecar],capture_output=True,text=True,timeout=60)
        if old.returncode==0:
            part=json.loads(old.stdout)
            assert part['files']==files and part['inventory_sha256']==config['inventory_sha256']
            check=json.loads(run(['lsjson',config['output_remote']+'/'+part_name,'--stat','--hash'],90).stdout)
            assert check['Size']==part['bytes'] and check['Hashes']['md5']==part['md5']
            parts.append(part);continue
        save(status='running',stage='copying_existing_shards',part=i,total_parts=len(batches),completed_bytes=sum(p['source_bytes'] for p in parts),total_bytes=inventory['bytes'])
        raw=out/'raw';raw.mkdir(exist_ok=True)
        assert 2*sum(f['bytes'] for f in files)+1024**3<shutil.disk_usage(out).free
        listing=out/'files.txt';listing.write_text('\n'.join(f['path'] for f in files)+'\n')
        run(['copy',inventory['source_remote'],str(raw),'--files-from',str(listing),'--transfers','8','--checkers','8','--checksum'])
        archive=out/part_name;part=pack(raw,files,archive)
        part.update(name=part_name,source_bytes=sum(f['bytes'] for f in files),inventory_sha256=config['inventory_sha256'])
        save(status='running',stage='uploading_verified_bundle',part=i,total_parts=len(batches),completed_bytes=sum(p['source_bytes'] for p in parts),total_bytes=inventory['bytes'])
        run(['copyto',str(archive),config['output_remote']+'/'+part_name,'--checksum'])
        check=json.loads(run(['lsjson',config['output_remote']+'/'+part_name,'--stat','--hash'],90).stdout)
        assert check['Size']==part['bytes'] and check['Hashes']['md5']==part['md5']
        meta=out/sidecar;meta.write_text(json.dumps(part,indent=2)+'\n')
        run(['copyto',str(meta),config['output_remote']+'/'+sidecar,'--checksum'],120)
        parts.append(part);archive.unlink();shutil.rmtree(raw)
    manifest=dict(status='complete',format='zip_stored_visual_bytes_v1',inventory_sha256=config['inventory_sha256'],source_remote=inventory['source_remote'],count=inventory['count'],bytes=inventory['bytes'],parts=parts,source_tensors_not_recomputed=True)
    path=out/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n');run(['copyto',str(path),config['output_remote']+'/manifest.json','--checksum'],120)
    save(status='complete',stage='finished',parts=len(parts),count=inventory['count'],bytes=inventory['bytes'],manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
except BaseException:
    save(status='error',stage='failed',error=traceback.format_exc());raise
