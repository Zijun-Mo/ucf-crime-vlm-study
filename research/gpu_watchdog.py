"""Emergency release for the preflight runtime, independent of local controller."""
import os
import threading
import time

UCF_PREFLIGHT_HARD_DEADLINE = time.time() + 900

def _ucf_hard_stop():
    time.sleep(max(0, UCF_PREFLIGHT_HARD_DEADLINE - time.time()))
    try:
        from google.colab import runtime
        runtime.unassign()
    finally:
        os._exit(0)

threading.Thread(target=_ucf_hard_stop, daemon=True).start()
print('UCF_PREFLIGHT_WATCHDOG_SECONDS', 900)
