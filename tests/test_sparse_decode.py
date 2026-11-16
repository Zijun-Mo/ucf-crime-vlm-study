import unittest,tempfile,importlib.util
from pathlib import Path
import numpy as np
from openjev_dss.runtime import decode_canonical_frames,iter_canonical_windows

@unittest.skipUnless(importlib.util.find_spec('cv2'),'opencv needed')
class SparseDecodeTests(unittest.TestCase):
 def test_seek_pixels_equal_sequential_and_tail(self):
  import cv2
  for valid,starts in [(3,[0,1]),(61,[0,6,23,30])]:
   with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'clip.mp4';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2,(224,224));self.assertTrue(w.isOpened())
    for i in range(valid+1):
     image=np.zeros((224,224,3),dtype=np.uint8);image[:,:,0]=i*7%256;image[:,:,1]=np.arange(224,dtype=np.uint8);image[10+i%50:40+i%50,20:90,2]=200;w.write(image)
    w.release()
    full=decode_canonical_frames(path,valid,starts,True);sparse=decode_canonical_frames(path,valid,starts,False)
    for k,value in sparse.items():np.testing.assert_array_equal(value,full[k])
    all_starts=list(range((valid+1)//2))
    streamed=[w for _,ws in iter_canonical_windows(path,valid,all_starts,4) for w in ws]
    self.assertEqual(len(streamed),len(all_starts))
    for s,w in zip(all_starts,streamed):np.testing.assert_array_equal(w,np.stack([full[min(2*s+j,valid)] for j in range(8)]))
    for s in starts:
     a=np.stack([full[min(2*s+j,valid)] for j in range(8)])
     b=np.stack([sparse[min(2*s+j,valid)] for j in range(8)])
     np.testing.assert_array_equal(a,b)
