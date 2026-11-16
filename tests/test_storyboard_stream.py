import cv2
import numpy as np
import pytest
from openjev_dss.storyboard_stream import StreamingWindows


def test_stream_reproduces_full_decode_overlap_tail_and_restart(tmp_path):
    path=tmp_path/'canonical.mp4';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2.,(224,224))
    assert w.isOpened()
    for i in range(21):
        frame=np.zeros((224,224,3),np.uint8);frame[:,:,0]=i*10;frame[:,:,1]=np.arange(224,dtype=np.uint8)
        w.write(frame)
    w.release();cap=cv2.VideoCapture(str(path));frames=[]
    while True:
        ok,x=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))
    cap.release();view=StreamingWindows(path,dict(window_count=10,sampled_frames=20))
    for repeat in range(2):
        actual=[]
        for i in range(0,len(view),4):actual.extend(view[i:i+4])
        expected=[np.stack([frames[min(2*i+j,20)] for j in range(8)]) for i in range(10)]
        assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
    with pytest.raises(ValueError,match='Non-sequential'):view[4:8]
    view.close()
