"""Byte-preserving archives of immutable visual shards; no tensor operations."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile


def safe_relative(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and name.endswith('.pt')
    return p


def groups(files, limit=2 * 1024**3):
    result, current, size = [], [], 0
    assert files and len({r['path'] for r in files}) == len(files)
    for row in sorted(files, key=lambda r:r['path']):
        safe_relative(row['path'])
        if current and size + row['bytes'] > limit:
            result.append(current); current, size = [], 0
        current.append(row); size += row['bytes']
    if current: result.append(current)
    return result


def file_digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def pack(root, files, target):
    root, target = Path(root), Path(target)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for row in files:
            path = root / safe_relative(row['path'])
            assert path.stat().st_size == row['bytes']
            digest = hashlib.md5()
            with path.open('rb') as source, archive.open(row['path'], 'w', force_zip64=True) as dest:
                while chunk := source.read(4 * 1024**2):
                    digest.update(chunk); dest.write(chunk)
            assert digest.hexdigest() == row['md5'], 'Source shard differs from pinned inventory'
    return dict(bytes=target.stat().st_size, sha256=file_digest(target), md5=file_digest(target, 'md5'), files=files)


def restore(archive_path, part, destination):
    archive_path, destination = Path(archive_path), Path(destination)
    assert archive_path.stat().st_size == part['bytes'] and file_digest(archive_path) == part['sha256']
    expected = {row['path']:row for row in part['files']}
    assert len(expected) == len(part['files'])
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
        for name, row in expected.items():
            path = destination / safe_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.bundle_partial')
            digest, size = hashlib.md5(), 0
            with archive.open(name) as source, temporary.open('wb') as out:
                while chunk := source.read(4 * 1024**2):
                    digest.update(chunk); size += len(chunk); out.write(chunk)
            assert size == row['bytes'] and digest.hexdigest() == row['md5']
            temporary.replace(path)
    return len(expected)


def restore_remote_bundle(config, destination, run, guard=lambda:None, progress=lambda row:None):
    """Restore the pinned CPU-built bytes using existing native rclone transport."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    work=destination.parent/'visual_bundle_download';work.mkdir(exist_ok=True)
    manifest_path=work/'manifest.json';remote=config['manifest_remote'].rsplit('/',1)[0]
    guard();run(['copyto',config['manifest_remote'],str(manifest_path)],120)
    assert file_digest(manifest_path)==config['manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='complete' and manifest['format']=='zip_stored_visual_bytes_v1'
    assert manifest['inventory_sha256']==config['inventory_sha256']
    files=[f for part in manifest['parts'] for f in part['files']]
    assert len(files)==manifest['count'] and len({f['path'] for f in files})==len(files)
    assert sum(f['bytes'] for f in files)==manifest['bytes']
    start=time.time();done=0
    for i,part in enumerate(manifest['parts']):
        guard();assert Path(part['name']).name==part['name'] and part['name'].endswith('.zip')
        target=work/part['name']
        progress(dict(stage='restoring_visual_bundle',part=i,total_parts=len(manifest['parts']),restored_files=done,elapsed_s=time.time()-start))
        run(['copyto',remote+'/'+part['name'],str(target),'--checksum'],1800)
        done+=restore(target,part,destination);target.unlink();guard()
    report=dict(complete=True,count=done,bytes=manifest['bytes'],manifest_sha256=config['manifest_sha256'],
                inventory_sha256=config['inventory_sha256'],all_shard_md5_verified=True,elapsed_s=time.time()-start,
                source_tensors_recomputed=False)
    progress(report);return report
