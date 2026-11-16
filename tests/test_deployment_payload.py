import json
from zipfile import ZipFile
from openjev_dss.deployment_payload import build_localization_payload


def test_resumed_training_archive_keeps_dependencies_but_omits_prediction_and_snapshot_history(tmp_path):
    def write(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) if not isinstance(value, str) else value)
    write('src/openjev_dss/train.py', 'pass')
    write('research/run_localization_training.py', 'pass')
    write('research/bootstrap_assets.json', {})
    write('research/localization_v4/policy.json', {'validation': {'annotations': 'research/dev/labels.json'}})
    write('research/localization_v4/teacher_cache_source.json', {'policy': 'research/localization_v2/policy.json'})
    write('research/localization_v2/policy.json', {})
    write('research/dev/labels.json', {})
    write('research/plateau/control.json', {'resume_steps': 806})
    write('research/localization_v4/epoch_2_review/validation/normal.json', {'scores': [1] * 10000})
    write('research/plateau/before_plateau/research/old.json', {'evidence': True})
    write('research/runs/old/receipt.json', {})
    target = tmp_path / 'source.zip'
    assert build_localization_payload(tmp_path, target, 'research/localization_v4/policy.json', 'research/plateau/control.json') < 4096
    with ZipFile(target) as archive:
        names = set(archive.namelist())
    assert 'research/dev/labels.json' in names and 'research/plateau/control.json' in names
    assert 'research/localization_v2/policy.json' in names
    assert not any('epoch_2_review' in n or 'before_plateau' in n or 'runs/old' in n for n in names)
