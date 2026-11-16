from pathlib import Path

import pytest

from openjev_dss.colab_control import marked_cell


def test_retried_control_cell_preserves_globals_and_executes_once(tmp_path):
    cell, marker, acknowledgement = marked_cell('updates = globals().get("updates", 0) + 1', 'setup.py')
    marker_path = tmp_path / Path(marker).name
    cell = cell.replace(marker, str(marker_path))
    kernel = {}
    exec(cell, kernel)
    exec(cell, kernel)
    assert kernel['updates'] == 1
    assert marker_path.read_text() == 'complete'
    assert acknowledgement in cell


def test_failed_cell_never_commits_completion(tmp_path):
    cell, marker, _ = marked_cell('raise RuntimeError("setup failed")', 'setup.py')
    marker_path = tmp_path / Path(marker).name
    with pytest.raises(RuntimeError, match='setup failed'):
        exec(cell.replace(marker, str(marker_path)), {})
    assert not marker_path.exists()
