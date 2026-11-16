from pathlib import Path
import cv2
import numpy as np
from openjev_dss.study_visuals import duration_windows
from openjev_dss.runtime import iter_canonical_windows


def test_duration_streams_preserve_4s_and_correctly_pad_2s_8s(tmp_path):
    path = tmp_path / 'canonical.mp4'
    valid = 7
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 2, (224, 224))
    assert writer.isOpened()
    for t in range(valid + 1):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[:, :, 0] = 20 * t
        image[10:100, 20:90, 2] = 200 - 10 * t
        writer.write(image)
    writer.release()
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    starts = [0, 1, 2, 3]
    for duration in [2, 4, 8]:
        actual = [window for _, batch in duration_windows(path, valid, starts, duration, 3) for window in batch]
        for start, window in zip(starts, actual):
            expected = np.stack([frames[min(2 * start + j, valid)] for j in range(duration * 2)])
            np.testing.assert_array_equal(window, expected)
        if duration == 4:
            historical = [window for _, batch in iter_canonical_windows(path, valid, starts, 3) for window in batch]
            for a, b in zip(actual, historical):
                np.testing.assert_array_equal(a, b)
