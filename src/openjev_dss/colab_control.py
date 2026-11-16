"""Completion markers for retryable Colab control cells, separate from training."""
import uuid


def marked_cell(source, filename):
    token = uuid.uuid4().hex
    marker = '/content/ucf_control_' + token + '.done'
    acknowledgement = 'UCF_CONTROL_COMPLETE ' + token
    # Execute in the persistent kernel namespace: later setup cells depend on it.
    cell = (
        "import pathlib as _ucf_control_pathlib\n"
        f"if not _ucf_control_pathlib.Path({marker!r}).exists():\n"
        f"    exec(compile({source!r}, {str(filename)!r}, 'exec'), globals(), globals())\n"
        f"    _ucf_control_pathlib.Path({marker!r}).write_text('complete')\n"
        f"print({acknowledgement!r})\n"
    )
    return cell, marker, acknowledgement
