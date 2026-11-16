import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from openjev_dss.experiment import calibrate_and_test

class SelectionBoundary(unittest.TestCase):
    def test_frozen_choices_precede_test_read_and_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);train=root/'train';test=root/'test';test.mkdir()
            for arm in ['A','B']:
                folder=train/arm;folder.mkdir(parents=True)
                (folder/'training.json').write_text(json.dumps([{'val_video_ap':.5}]))
                (folder/'selection.json').write_text(json.dumps({'test_used':False}))
                (folder/'best_state.pt').write_bytes(arm.encode())
            manifest={'records':[], 'test_identity':[{'name':'v','sha256':'s'}]}
            (test/'manifest.json').write_text(json.dumps({'videos':manifest['test_identity']}))
            rt=SimpleNamespace(model=SimpleNamespace(eval=lambda:None),head=SimpleNamespace(eval=lambda:None),forward=None)
            original_read=Path.read_text
            def guarded_read(path,*args,**kwargs):
                if path==test/'manifest.json':
                    self.assertTrue((root/'out/selection_lock.json').exists())
                return original_read(path,*args,**kwargs)
            with patch('openjev_dss.experiment.restore'),patch('openjev_dss.experiment.torch.load',return_value={}),patch('openjev_dss.experiment.calibrate_videos',return_value={'temperature':1.}),patch('openjev_dss.evaluation.run_frozen_paired',return_value='done'),patch.object(Path,'read_text',guarded_read):
                self.assertEqual(calibrate_and_test(rt,manifest,None,train,test,root/'out'),'done')
                (train/'B/best_state.pt').write_bytes(b'changed')
                with self.assertRaises(AssertionError):
                    calibrate_and_test(rt,manifest,None,train,test,root/'out')

if __name__=='__main__':unittest.main()
