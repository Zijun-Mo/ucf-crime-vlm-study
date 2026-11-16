import tempfile,unittest
from pathlib import Path
import torch
from unittest.mock import patch
from openjev_dss.losses import loss
from openjev_dss.training import two_pass
from openjev_dss.cache import FeatureCache
from openjev_dss.experiment import score_windows

class Efficiency(unittest.TestCase):
    def test_compiler_cache_only_restores_matching_environment(self):
        from openjev_dss.kernels import save_kernel_cache,restore_kernel_cache
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scratch=root/'scratch';scratch.mkdir();(scratch/'kernel.bin').write_bytes(b'compiled')
            with patch('openjev_dss.kernels.environment',return_value={'driver':'one'}):
                save_kernel_cache(root/'drive',scratch)
                self.assertTrue(restore_kernel_cache(root/'drive',root/'restored'))
                self.assertEqual((root/'restored/kernel.bin').read_bytes(),b'compiled')
            with patch('openjev_dss.kernels.environment',return_value={'driver':'two'}):
                self.assertFalse(restore_kernel_cache(root/'drive',root/'changed'))
                self.assertFalse((root/'changed/kernel.bin').exists())
    def test_inference_padding_never_counts_dummy_windows(self):
        shapes=[]
        def forward(w):shapes.append(len(w));return w.sum(-1)
        x=torch.arange(30.).reshape(6,5)
        torch.testing.assert_close(score_windows(forward,x,4),x.sum(-1))
        self.assertEqual(shapes,[4,4])
    def test_packed_accumulation_matches_four_microbatches(self):
        torch.manual_seed(7);net=torch.nn.Sequential(torch.nn.Linear(5,9),torch.nn.Tanh(),torch.nn.Linear(9,1))
        windows=[torch.randn(n,5) for n in [16,16,3,9,1,8,16,7]];labels=[0,1]*4
        for dss in [False,True]:
            net.zero_grad()
            for i in range(0,8,2):
                v,_=loss([net(w).flatten() for w in windows[i:i+2]],labels[i:i+2],dss)
                (v/4).backward()
            expected=[p.grad.clone() for p in net.parameters()]
            for batch in [1,4,8]:
                net.zero_grad();_,info=two_pass(net,windows,labels,dss,chunk=batch,score_batch=batch,pad_batches=True)
                for p,g in zip(net.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=1e-6)
                self.assertEqual(info['score_calls'],(sum(map(len,windows))+batch-1)//batch)
    def test_drive_authority_and_bounded_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'drive_a.pt';b=root/'drive_b.pt';signature={'revision':'frozen'}
            cache=FeatureCache(root/'scratch',memory_bytes=16,disk_bytes=10**6)
            cache.put(a,signature,torch.ones(4));self.assertTrue(a.exists())
            torch.testing.assert_close(cache.get(a,signature),torch.ones(4))
            with self.assertRaises(AssertionError):cache.get(a,{'revision':'wrong'})
            cache.put(b,signature,torch.zeros(4));self.assertLessEqual(cache.memory_used,16)
            cache.get(a,signature);self.assertEqual(cache.stats['disk_hits'],1)
            cache=FeatureCache(root/'fresh_scratch',memory_bytes=0,disk_bytes=10**6)
            cache.get(a,signature);cache.get(a,signature)
            self.assertEqual(cache.stats['drive_reads'],1);self.assertEqual(cache.stats['disk_hits'],1)

if __name__=='__main__':unittest.main()
