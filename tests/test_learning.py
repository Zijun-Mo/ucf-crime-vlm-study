import unittest,copy
from openjev_dss.learning_audit import candidate_pairs
from openjev_dss.learning import schedule

class LearningTests(unittest.TestCase):
 def test_topk_reordering_is_not_instance_change(self):
  import torch
  from openjev_dss.losses import bag_loss
  probabilities=torch.tensor([.86,.99]+[.01]*14)
  _,_,info=bag_loss(torch.logit(probabilities),1,True)
  self.assertFalse(info['top_changed']);self.assertEqual(set(info['mil_top']),set(info['ss_top']))
 def test_hash_radius_index_has_no_false_negative(self):
  import random
  rng=random.Random(42)
  for changed in range(5):
   a=rng.getrandbits(64);b=a
   for bit in rng.sample(range(64),changed):b ^=1<<bit
   rows=[{'name':'a','phashes':[format(a,'064b')]*8},{'name':'b','phashes':[format(b,'064b')]*8}]
   self.assertEqual(len(candidate_pairs(rows)),1)
 def test_contained_clip_is_isolated(self):
  rows=[{'name':'long','phashes':['0'*64]*3+['1'*64]*9},{'name':'short','phashes':['0'*64]*8}]
  self.assertEqual(len(candidate_pairs(rows)),1)
 def test_no_single_frame_duplicate_acceptance(self):
  rows=[{'name':'a','phashes':['0'*64]*8},{'name':'b','phashes':['0'*64]+['1'*64]*7}]
  self.assertEqual(candidate_pairs(rows),[])
 def test_equal_epoch_video_coverage_and_balanced_micros(self):
  rs=[{'name':str(i),'split':'train','label':int(i>=7)} for i in range(16)]
  a=schedule(rs,3);self.assertEqual(a,schedule(copy.deepcopy(rs),3))
  for ep in range(3):
   names=[]
   for epoch,micros in a:
    if epoch!=ep:continue
    self.assertLessEqual(len(micros),4)
    for normal,abnormal in micros:
     self.assertEqual((normal['label'],abnormal['label']),(0,1));names.extend([normal['name'],abnormal['name']])
   self.assertEqual(set(names),set(str(i) for i in range(16)))

class ResumeTests(unittest.TestCase):
 def test_budget_stop_then_resume_preserves_paired_training(self):
  import torch,tempfile,time,json
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from openjev_dss.learning import run
  class Model(torch.nn.Linear):
   def save_pretrained(self,path):Path(path).mkdir(exist_ok=True)
  model=Model(2,2);head=torch.nn.Linear(2,1);frozen=torch.nn.Linear(2,2);frozen.requires_grad_(False)
  rt=SimpleNamespace(model=model,head=head,full=frozen,forward=lambda x:head(model(x)).flatten())
  rs=[{'name':str(i),'split':'train' if i<10 else 'val','label':i%2,'group':str(i)} for i in range(14)]
  manifest={'records':rs,'audit':{'near_duplicate_visual_audit':'passed'}}
  def provider(r,epoch,full):return torch.randn(16,2,generator=torch.Generator().manual_seed(42+int(r['name'])+epoch))
  before={k:v.clone() for k,v in frozen.state_dict().items()}
  with tempfile.TemporaryDirectory() as d,patch('torch.cuda.get_rng_state',side_effect=torch.get_rng_state),patch('torch.cuda.set_rng_state'),patch('torch.cuda.manual_seed_all'):
   out=Path(d)/'training'
   stopped=run(rt,manifest,provider,out,deadline=0,epochs=1)
   self.assertEqual(stopped['status'],'budget_stopped')
   complete=run(rt,manifest,provider,out,deadline=time.time()+3600,epochs=1)
   self.assertEqual(complete['status'],'complete')
   self.assertEqual([len(complete['arms'][a]['rows']) for a in ['A','B']],[2,2])
   self.assertTrue(complete['paired_steps_equal']);self.assertTrue(complete['frozen_versions_unchanged'])
   for k,v in before.items():torch.testing.assert_close(v,frozen.state_dict()[k],rtol=0,atol=0)
   for a in ['A','B']:self.assertTrue((out/a/'selection.json').exists())
