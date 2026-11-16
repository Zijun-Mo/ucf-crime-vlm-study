"""Publish two-generation training bundles with the status pointer committed last."""
from pathlib import Path


def publish_training_bundle(directory, remote, transfer):
    directory = Path(directory)
    # A failed bulk upload may leave an orphan new generation. The previous
    # status still references the other slot, so it remains safely resumable.
    transfer(['copy', str(directory), remote, '--exclude', '/status.json', '--transfers', '4', '--checkers', '4'])
    status = directory / 'status.json'
    if status.exists():
        transfer(['copyto', str(status), remote.rstrip('/') + '/status.json'])
