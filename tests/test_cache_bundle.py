import hashlib
import pytest
from openjev_dss.cache_bundle import groups, pack, restore


def test_bundle_preserves_every_byte_and_rejects_modified_archive(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();rows=[]
    for name,data in [('ab/first.pt',b'feature_one'),('bc/second.pt',b'feature_two')]:
        p=raw/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
        rows.append(dict(path=name,bytes=len(data),md5=hashlib.md5(data).hexdigest()))
    assert len(groups(rows,limit=15))==2
    target=tmp_path/'bundle.zip';part=pack(raw,rows,target)
    assert restore(target,part,tmp_path/'restored')==2
    for row in rows:assert (raw/row['path']).read_bytes()==(tmp_path/'restored'/row['path']).read_bytes()
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(AssertionError):restore(target,part,tmp_path/'bad')


def test_invalid_cache_path_or_source_hash_fails_closed(tmp_path):
    with pytest.raises(AssertionError):groups([dict(path='../escape.pt',bytes=1,md5='bad')])
    p=tmp_path/'x.pt';p.write_bytes(b'x')
    with pytest.raises(AssertionError):pack(tmp_path,[dict(path='x.pt',bytes=1,md5='bad')],tmp_path/'bad.zip')


def test_remote_bundle_restore_checks_pinned_manifest_and_preserves_shards(tmp_path):
    import json,shutil
    from openjev_dss.cache_bundle import file_digest,restore_remote_bundle
    source=tmp_path/'cloud';source.mkdir();raw=tmp_path/'raw';raw.mkdir()
    data=b'immutable feature bytes';(raw/'x.pt').write_bytes(data)
    row=dict(path='x.pt',bytes=len(data),md5=hashlib.md5(data).hexdigest())
    part=pack(raw,[row],source/'part.zip');part['name']='part.zip'
    manifest=dict(status='complete',format='zip_stored_visual_bytes_v1',inventory_sha256='fixed',count=1,bytes=len(data),parts=[part])
    (source/'manifest.json').write_text(json.dumps(manifest))
    config=dict(manifest_remote='cloud/manifest.json',manifest_sha256=file_digest(source/'manifest.json'),inventory_sha256='fixed')
    def transport(args,timeout):shutil.copy2(source/args[1].split('/')[-1],args[2])
    report=restore_remote_bundle(config,tmp_path/'restored',transport)
    assert report['all_shard_md5_verified'] and report['source_tensors_recomputed'] is False
    assert (tmp_path/'restored/x.pt').read_bytes()==data
    with pytest.raises(AssertionError):restore_remote_bundle({**config,'manifest_sha256':'changed'},tmp_path/'bad',transport)
