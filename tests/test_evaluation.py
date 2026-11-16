import unittest,tempfile
import numpy as np
from openjev_dss.evaluation import aggregate,paired_report
class Evaluation(unittest.TestCase):
 def test_raw_overlap_and_stride(self):
  r={'n_frames':3,'fps':1};p={'start_s':[0,1,2],'raw_logits':[0.,2.,4.]}
  np.testing.assert_array_equal(aggregate(r,p,1),[0,1,2]);np.testing.assert_array_equal(aggregate(r,p,2),[0,0,2]);np.testing.assert_array_equal(aggregate(r,p,4),[0,0,0])
 def test_paired_video_bootstrap_identical_arms(self):
  records=[{'name':str(i),'n_frames':3,'fps':1,'intervals':[(2,2)] if i%2 else []} for i in range(290)]
  scores={r['name']:{'start_s':[0,1,2],'raw_logits':[-2.,0.,2.] if i%2 else [-4.,-3.,-2.]} for i,r in enumerate(records)}
  with tempfile.NamedTemporaryFile() as f:
   report=paired_report(records,{'A':scores,'B':scores},f.name,bootstrap=5)
  for stride in report['strides'].values():
   for scope in stride['paired_B_minus_A_95pct_CI'].values():
    for ci in scope.values():self.assertEqual(ci,[0.,0.])
if __name__=='__main__':unittest.main()
