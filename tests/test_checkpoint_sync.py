import json
from pathlib import Path
import pytest
from openjev_dss.checkpoint_sync import publish_training_bundle


def test_interrupted_bulk_publish_keeps_previous_resume_pointer(tmp_path):
    local=tmp_path/'local';local.mkdir()
    (local/'resume_0.pt').write_bytes(b'old-weights')
    (local/'resume_1.pt').write_bytes(b'new-weights')
    (local/'status.json').write_text(json.dumps({'generation':1,'steps':20}))
    cloud={'resume_0.pt':b'old-weights','status.json':json.dumps({'generation':0,'steps':10}).encode()}
    fail=True

    def transfer(command):
        nonlocal fail
        if command[0]=='copy':
            # Simulate a transfer failing after the new checkpoint arrived.
            cloud['resume_1.pt']=(local/'resume_1.pt').read_bytes()
            if fail:
                fail=False
                raise TimeoutError('interrupted before all bulk data were durable')
        else:
            assert command[0]=='copyto' and Path(command[1]).name=='status.json'
            cloud['status.json']=Path(command[1]).read_bytes()

    with pytest.raises(TimeoutError):
        publish_training_bundle(local,'gdrive:training',transfer)
    status=json.loads(cloud['status.json'])
    assert status=={'generation':0,'steps':10}
    assert cloud[f"resume_{status['generation']%2}.pt"]==b'old-weights'
    publish_training_bundle(local,'gdrive:training',transfer)
    status=json.loads(cloud['status.json'])
    assert status=={'generation':1,'steps':20}
    assert cloud[f"resume_{status['generation']%2}.pt"]==b'new-weights'


def test_initial_bundle_without_status_does_not_publish_a_resume_pointer(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    commands=[]
    publish_training_bundle(tmp_path,'gdrive:training',commands.append)
    assert len(commands)==1 and commands[0][0]=='copy'
    assert '--exclude' in commands[0]
