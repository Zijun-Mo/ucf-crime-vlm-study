import unittest,torch,math
from openjev_dss.losses import loss,bag_loss
from openjev_dss.training import two_pass
class DSS(unittest.TestCase):
 def test_equation(self):
  z=torch.tensor([-2.,-.4,.5,1.,2.,3.,4.,0.,-.8,1.4,1.9,0.6,-.3,-1.,-.9,-3.],requires_grad=True)
  p=z.sigmoid();q=p.topk(2).values.mean();s=torch.where(p>.85,.55*p.max(),p)
  a=s.topk(2).values.mean();b=s.topk(2,largest=False).values.mean()
  mil,ss,_=bag_loss(z,1,True)
  torch.testing.assert_close(mil,-q.log());torch.testing.assert_close(ss,-(1-a)*a.log()-b*torch.log1p(-b))
  torch.testing.assert_close(torch.autograd.grad(mil+ss,z,retain_graph=True)[0],torch.autograd.grad(-q.log()-(1-a)*a.log()-b*torch.log1p(-b),z)[0])
 def test_extreme_short_ties(self):
  for vals in [[-1000.],[1000.],[-1000.,1000.],[0.]*16,[-2.]*3,[3.]*16]:
   for y in [0,1]:
    z=torch.tensor(vals,requires_grad=True);l,_=loss([z],[y],True);g=torch.autograd.grad(l,z)[0]
    self.assertTrue(torch.isfinite(l));self.assertTrue(torch.isfinite(g).all())
  for zval,y,direction in [(-1000.,1,-1),(1000.,0,1),(-2.,1,-1),(2.,0,1)]:
   z=torch.tensor([zval],requires_grad=True);v,_=loss([z],[y]);self.assertEqual(torch.autograd.grad(v,z)[0].sign().item(),direction)
 def test_chain_rule(self):
  torch.manual_seed(42);net=torch.nn.Linear(5,1);x=[torch.randn(16,5),torch.randn(9,5)]
  for dss in [False,True]:
   net.zero_grad();v,_=loss([net(w).flatten() for w in x],[0,1],dss);v.backward();expected=[p.grad.clone() for p in net.parameters()]
   net.zero_grad();two_pass(net,x,[0,1],dss,chunk=3)
   for p,g in zip(net.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=1e-6)

class DataAndScores(unittest.TestCase):
 def test_score_sensitive_directions(self):
  z=torch.tensor([1.,-2.,.2,.3,.4,.1,0.,-.2],requires_grad=True)
  _,ss,info=bag_loss(z,1,True);g=torch.autograd.grad(ss,z)[0]
  self.assertLess(g[0],0);self.assertGreater(g[1],0);self.assertEqual(info['suppressed'],0)
  z=torch.tensor([4.,1.,-.5,-1.,0.,.2,.3,.4],requires_grad=True)
  _,ss,info=bag_loss(z,1,True);self.assertEqual(info['suppressed'],1);self.assertTrue(info['top_changed'])
 def test_stratified_unique(self):
  from openjev_dss.data import sample_starts
  for n in [1,3,16,17,30,999]:
   a=sample_starts(n,0,'video');self.assertEqual(len(a),min(n,16));self.assertEqual(len(set(a)),len(a));self.assertTrue(all(0<=i<n for i in a));self.assertEqual(a,sample_starts(n,0,'video'))
 def test_mean_probability_not_mean_logit(self):
  z=torch.tensor([-10.,1.,0.,0.,0.,0.,0.,0.,0.],requires_grad=True)
  q=z.sigmoid().topk(2).values.mean();mil,_,_=bag_loss(z,1)
  torch.testing.assert_close(mil,-q.log());self.assertGreater(abs(float((mil+z.topk(2).values.mean().sigmoid().log()).detach())),.0001)

if __name__=='__main__':unittest.main()
