import copy
import numpy as np
import pytest
import torch
from contextlib import nullcontext
from types import SimpleNamespace
from openjev_dss.localization_objective import normal_reference, selection, objective
from openjev_dss.localization_backward import backward_video, padded_forward
from openjev_dss.localization_training import run_localization,validate_annotations
from openjev_dss.localization_fast_backward import backward_fixed_selection


def test_teacher_empty_positive_bag_is_not_forced_positive():
    reference=normal_reference([[-2,-1,0,1],[-2,-1,0,1]])
    chosen=selection([-3.]*20,reference)
    z=torch.full((20,),-3.,requires_grad=True)
    loss,stats=objective(z,1,chosen);loss.backward()
    assert stats==dict(normal=0.,positive=0.,rank=0.)
    assert chosen['summary']['positive_windows']==0
    assert not z.grad.any()


def test_normal_calibration_weights_videos_equally_and_rejects_invalid():
    a=normal_reference([[0.]*100,[10.]],.75)
    assert a['threshold']==10.
    with pytest.raises(ValueError):normal_reference([[float('nan')]])


def test_adaptive_segments_budget_and_teacher_order_protection():
    reference=dict(threshold=0.,scale=1.)
    t=[-3.]*20+[4.]*20+[-3.]*20+[5.]*20
    chosen=selection(t,reference)
    assert chosen['summary']['positive_windows']==40
    assert chosen['summary']['positive_segments']==2
    assert chosen['summary']['soft_coverage_s'] <= .25*len(t)+1e-8
    teacher=torch.tensor(t,requires_grad=True)
    _,baseline=objective(teacher,1,chosen)
    assert baseline['rank']==0.
    inverted=-teacher.detach().clone().requires_grad_(True)
    _,bad=objective(inverted,1,chosen)
    assert bad['rank']>0
    # Relative low anchors are not assigned a zero-label positive-video BCE.
    for segment in chosen['segments']:
        assert all(t[i]>0 for i in segment)


def test_normal_bce_reaches_every_window_and_respects_video_normalization():
    z=torch.tensor([-10.,-3.,0.,3.],requires_grad=True)
    loss,_=objective(z,0);loss.backward()
    torch.testing.assert_close(z.grad,z.detach().sigmoid()/4)


@pytest.mark.parametrize('label',[0,1])
@pytest.mark.parametrize('empty',[False,True])
def test_fast_algebra_preserves_dense_loss_and_gradients_across_batches(label,empty):
    torch.manual_seed(22);model=torch.nn.Linear(2,1);fast=copy.deepcopy(model)
    x=torch.randn(73,2)
    t=[-4.]*25+[5.]*20+[-3.]*20+[6.]*8
    chosen=selection([-3.]*73 if empty else t,dict(threshold=0.,scale=1.))
    dense=torch.cat([padded_forward(model,x[i:i+8],8) for i in range(0,len(x),8)])
    expected,_=objective(dense,label,chosen,positive_weight=.7,rank_weight=1.3)
    (expected*.4).backward()
    value,info=backward_fixed_selection(fast,x,label,chosen,batch=8,scale=.4,positive_weight=.7,rank_weight=1.3)
    assert value==pytest.approx(float(expected.detach()),rel=1e-6)
    assert info['loss_equivalent']
    if label==0:assert info['score_calls']==0
    for a,b in zip(model.parameters(),fast.parameters()):
        # A completely inactive positive video performs no backward, just as
        # the original score-leaf path skips all zero-upstream batches.
        actual=b.grad if b.grad is not None else torch.zeros_like(b)
        torch.testing.assert_close(a.grad,actual,rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('label',[0,1])
def test_streaming_full_video_matches_dense_gradients_including_tail(label):
    torch.manual_seed(4)
    model=torch.nn.Linear(2,1);stream=copy.deepcopy(model)
    x=torch.randn(19,2)
    selected=selection([-3.]*9+[4.]*10,dict(threshold=0.,scale=1.))
    dense=torch.cat([padded_forward(model,x[i:i+4]) for i in range(0,len(x),4)])
    expected,_=objective(dense,label,selected);(.5*expected).backward()
    class Lazy:
        def __init__(self): self.reads=[]
        def __len__(self):return len(x)
        def __getitem__(self,key):
            assert isinstance(key,slice) and key.stop-key.start<=4
            self.reads.append(key.start);return x[key]
    lazy=Lazy()
    actual,info=backward_video(stream,lazy,lambda z:objective(z,label,selected),scale=.5)
    assert actual==pytest.approx(float(expected.detach()))
    assert info['windows']==19 and info['all_windows_scored']
    assert set(range(0,19,4)) <= set(lazy.reads)
    for a,b in zip(model.parameters(),stream.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)


def test_real_annotation_gate_rejects_train_video_and_unreviewed_intervals():
    row=dict(name='v',source_sha256='sha',n_frames=12,fps=1.,intervals=[[3,6]],reviewed=True)
    manifest=dict(records=[dict(name='v',sha256='sha',label=1,split='val')])
    meta={'v':dict(n_frames=12,fps=1.)}
    annotations=dict(source='visually reviewed fixture',minimum_videos=1,records=[row])
    assert validate_annotations(annotations,manifest,meta)['v']['intervals']==[[3,6]]
    with pytest.raises(ValueError):validate_annotations({**annotations,'records':[{**row,'reviewed':False}]},manifest,meta)
    manifest['records'][0]['split']='train'
    with pytest.raises(ValueError):validate_annotations(annotations,manifest,meta)


@pytest.mark.parametrize('fast',[False,True,'pointwise','soft'])
def test_full_pipeline_cpu_smoke_and_resume_after_partial_accumulation(tmp_path,fast):
    import json,time
    model=torch.nn.Linear(1,1);model.disable_adapter=lambda:nullcontext()
    with torch.no_grad():model.weight.fill_(1.);model.bias.zero_()
    modes=[]
    def forward(x):modes.append(model.training);return model(x).flatten()
    rt=SimpleNamespace(model=model,full=model,head=torch.nn.Identity(),device='cpu',forward=forward)
    # Checkpoint exporter used only when an epoch completes; replace with a
    # small, real filesystem artifact in this CPU integration fixture.
    def save_pretrained(path):path.mkdir(exist_ok=True);(path/'fixture.txt').write_text('adapter')
    model.save_pretrained=save_pretrained
    records=[dict(name=name,label=label,split=split,sha256=name) for name,label,split in
        [('normal',0,'train'),('abnormal',1,'train'),('vn',0,'val'),('va',1,'val')]]
    metadata={r['name']:dict(n_frames=12,fps=1.,window_count=12) for r in records}
    class Provider:
        def __init__(self):self.metadata=metadata;self.fail=False;self.calls=0
        def __call__(self,record,epoch,exhaustive=False):
            assert exhaustive
            self.calls+=1
            if self.fail and record['split']=='train' and self.calls%2==0:raise TimeoutError('injected mid-step interruption')
            return torch.tensor(([-2.]*6+[3.]*6) if record['label'] else [-2.]*12)[:,None]
    provider=Provider();manifest=dict(records=records,audit=dict(near_duplicate_visual_audit='passed'))
    annotations=dict(source='real reviewed toy fixture',minimum_videos=1,records=[dict(name='va',source_sha256='va',
        n_frames=12,fps=1.,intervals=[[7,12]],reviewed=True)])
    policy=dict(selection={},optimizer=dict(lr=1e-3),loss=dict(positive_weight=1.,rank_weight=1.),
        training=dict(max_epochs=2,min_epochs=2,patience=2,min_delta=.002,videos_per_step=2,seed=42))
    policy['selection']['normal_quantile']=.99
    if fast in ['pointwise','soft']:
        policy['loss'].update(kind='pointwise_teacher_v2',anchor_weight=1.,huber_delta=1.)
        policy['sampling']=dict(fraction=.25,normal_cap=4,positive_cap=3,anchor_cap=3)
        policy['input']=dict(batch=4);policy['resources']=dict(epoch_max_hours=3.)
    if fast=='soft':
        policy['loss']=dict(kind='pointwise_minmax_soft_v4',minimum_probability_range=1e-12)
        policy['sampling']=dict(fraction=.25,normal_cap=4,anomaly_cap=4)
        policy['validation']=dict(reviewed_anomaly_videos=1)
    (tmp_path/'gradient_preflight.json').write_text(json.dumps(dict(passed=True)))
    # Preparation is a genuine teacher stage with no optimizer state mutation.
    initial=copy.deepcopy(model.state_dict())
    prepared=run_localization(rt,manifest,provider,tmp_path,time.time()+3600,policy,'prepare')
    assert prepared['status']=='prepared' and prepared['steps']==0
    for key,value in initial.items():torch.testing.assert_close(model.state_dict()[key],value)
    provider.fail=True;provider.calls=0
    interrupted=run_localization(rt,manifest,provider,tmp_path,time.time()+3600,policy,'train',annotations)
    assert interrupted['status']=='budget_stopped' and interrupted['steps']==0 and interrupted['video_cursor']==0
    provider.fail=False
    final=run_localization(rt,manifest,provider,tmp_path,time.time()+3600,policy,'train',annotations,
        fast_execution=dict(batch=4,backend='math',qualification_sha256='toy') if fast is True else None)
    assert final['status']=='complete' and final['steps']==2 and len(final['validation'])==2
    assert all(v['genuine_temporal_labels'] for v in final['validation'])
    assert final['test_used'] is False
    if fast=='soft':
        update=json.loads((tmp_path/'first_optimizer_update.json').read_text())
        assert update['actual_parameters_changed'] and update['initial_tensor_sha256']!=update['updated_tensor_sha256']
        assert final['validation'][-1]['dev_frame_videos']==2
        # The extension keeps the exact scientific-policy fingerprint and
        # optimizer, and must cross the old epoch cap before stopping on plateau.
        final.update(bad_epochs=0,status='budget_complete',stage='finished')
        (tmp_path/'status.json').write_text(json.dumps(final))
        control=dict(id='toy-extension',base_fingerprint=final['fingerprint'],
            resume_steps=2,completed_epochs=2,resume_video_cursor=0)
        continued=run_localization(rt,manifest,provider,tmp_path,time.time()+3600,policy,'train',annotations,
            continuation=control)
        assert continued['epoch_cursor']==4>policy['training']['max_epochs']
        assert continued['stop_reason']=='localization_validation_plateau'
        update=json.loads((tmp_path/'continuation_first_optimizer_update.json').read_text())
        assert update['previous_steps']==2 and update['steps']==3
        assert update['actual_parameters_changed'] and update['durable_checkpoint_verified']
        assert all(s==3 for s in update['optimizer_steps'])
    assert any(modes),'Teacher eval mode must not leak into optimizer training'
