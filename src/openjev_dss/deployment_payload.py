"""Small localization deployment archives; historical evidence stays on the host."""
import json
from pathlib import Path
import zipfile


MAX_SMALL_UPLOAD_BYTES = 4 * 1024 ** 2


def localization_payload_paths(root, policy_path, continuation_path=None):
    root = Path(root)
    paths = {p for folder in ['src', 'research'] for p in (root / folder).rglob('*.py')
             if '__pycache__' not in p.parts and 'runs' not in p.parts
             and not any(part in ['history', 'before_plateau'] or part.startswith('before_') for part in p.relative_to(root).parts)}
    paths.update((root / 'research').glob('*.json'))
    policy_file = root / policy_path
    policy = json.loads(policy_file.read_text())
    paths.add(policy_file)
    annotation = policy.get('validation', {}).get('annotations')
    if annotation:
        paths.add(root / annotation)
    source_file = policy_file.with_name('teacher_cache_source.json')
    if source_file.exists():
        paths.add(source_file)
        paths.add(root / json.loads(source_file.read_text())['policy'])
    # These are small execution qualification inputs, not prediction histories.
    for name in ['cpu_preflight.json', 'publication_live_test.json',
                 'optimization_qualification.json', 'optimization_hold.json']:
        path = policy_file.with_name(name)
        if path.exists():
            paths.add(path)
    if continuation_path:
        paths.add(root / continuation_path)
    assert all(p.is_file() for p in paths), 'Missing localization deployment dependency'
    return sorted(paths)


def build_localization_payload(root, target, policy_path, continuation_path=None):
    root, target = Path(root), Path(target)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in localization_payload_paths(root, policy_path, continuation_path):
            archive.write(path, path.relative_to(root).as_posix())
    assert target.stat().st_size <= MAX_SMALL_UPLOAD_BYTES, 'Deployment archive exceeds small-upload limit; do not allocate GPU'
    return target.stat().st_size
