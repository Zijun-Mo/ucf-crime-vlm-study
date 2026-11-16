"""Resumable Qwen localization adaptation with immutable frozen-teacher scores."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from .controlled_training import tensor_digest
from .learning import atomic_json
from .localization_backward import stream_scores, backward_video, padded_forward
from .localization_objective import normal_reference, selection, objective
from .localization_fast_backward import backward_fixed_selection
from .localization_pointwise import plan as pointwise_plan, backward_pointwise
from .study_evaluation import frame_labels, frame_scores
from .training import trainable_state, restore


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate_annotations(data,manifest,metadata):
    """Fail closed on unreviewed, training/test, identity or timeline mismatches."""
    records={r['name']:r for r in manifest['records'] if r['split']=='val' and r['label']==1}
    if not data.get('source') or not data.get('records'):raise ValueError('Missing real dev annotations')
    result={}
    for row in data['records']:
        name=row['name']
        if name in result or name not in records or row.get('reviewed') is not True:
            raise ValueError('Dev annotation must be unique, reviewed and validation-only')
        record=records[name];meta=metadata[name]
        if row['source_sha256'] != record['sha256']:
            raise ValueError('Dev source identity mismatch')
        if row['n_frames'] != meta['n_frames'] or row['fps'] != meta['fps']:
            raise ValueError('Dev annotation timeline differs from original video')
        intervals=row['intervals']
        if not intervals or any(not isinstance(a,int) or not isinstance(b,int) or not 1<=a<=b<=meta['n_frames'] for a,b in intervals):
            raise ValueError('Expected one-based inclusive, real temporal intervals')
        item={**meta,'intervals':intervals};y=frame_labels(item)
        if not 0 < y.sum() < len(y):raise ValueError('Within-video AUROC needs both frame classes')
        result[name]=item
    if len(result)<data.get('minimum_videos',20):
        raise ValueError('Too few reviewed localization development videos')
    return result


def metrics(records,scores,annotations):
    y=[r['label'] for r in records];z=[max(scores[r['name']]) for r in records]
    result={'video_max_ap':float(average_precision_score(y,z)),
            'video_max_auroc':float(roc_auc_score(y,z)), 'videos':len(records)}
    values=[]
    for name,row in annotations.items():
        p=scores[name]
        values.append(roc_auc_score(frame_labels(row),frame_scores(row,list(range(len(p))),p)))
    result.update(within_macro_auroc=float(np.mean(values)) if values else None,
                  localization_videos=len(values), genuine_temporal_labels=bool(values))
    return result


def gradient_preflight(runtime,features,selected,out):
    """Check actual LoRA gradients against dense AD using the new loss."""
    params=[p for p in runtime.model.parameters() if p.requires_grad]
    runtime.model.train();runtime.head.train()
    def clear():runtime.model.zero_grad(set_to_none=True)
    def vector():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    reports=[]
    for label in [0,1]:
        clear();z=torch.cat([padded_forward(runtime.forward,features[i:i+4]) for i in range(0,len(features),4)])
        dense,_=objective(z,label,selected);dense.backward();expected=vector().clone()
        clear();value,info=backward_video(runtime.forward,features,lambda z:objective(z,label,selected))
        actual=vector();norm=float(expected.norm())
        relative=float((expected-actual).norm()/max(norm,1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(expected.double(),actual.double(),dim=0)) if norm else 1.
        reports.append(dict(label=label,dense_loss=float(dense.detach()),stream_loss=value,
                            relative_l2=relative,cosine=cosine,execution=info))
        del actual,expected,dense,z
    clear()
    report=dict(passed=all(r['relative_l2']<.01 and r['cosine']>.999 for r in reports),
                train_only=True,physical_batch=4,checks=reports)
    atomic_json(out,report)
    if not report['passed']:raise ValueError('Localization dense/stream gradient gate failed')
    return report


def run_localization(runtime,manifest,provider,out,deadline,policy,phase='prepare',
                     annotations=None,publish=lambda:None,annotation_loader=lambda:None,fast_execution=None,
                     continuation=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert manifest['audit']['near_duplicate_visual_audit']=='passed'
    records=sorted([r for r in manifest['records'] if r['split'] in ['train','val']],key=lambda r:r['name'])
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    dev=validate_annotations(annotations,manifest,provider.metadata) if annotations else {}
    if phase=='train' and not dev:raise ValueError('Training requires real reviewed temporal development labels')
    fingerprint=digest(dict(manifest=manifest,policy=policy))
    soft=policy['loss'].get('kind')=='pointwise_minmax_soft_v4'
    frame_selection=soft or policy['loss'].get('kind')=='pointwise_focal_v3'
    focal=policy['loss'].get('kind')=='pointwise_focal_v3'
    pointwise=policy['loss'].get('kind') in ['pointwise_teacher_v2','pointwise_focal_v3','pointwise_minmax_soft_v4']
    if focal:
        from .localization_focal import select as focal_select,plan as focal_plan,backward as focal_backward,calibration,eligibility
        from .localization_focal_audit import development_metrics,pseudo_label_audit
    if soft:
        from .localization_soft import targets as soft_targets,plan as soft_plan,backward as soft_backward
        from .localization_focal import eligibility
        from .localization_focal_audit import development_metrics
    status_path=out/'status.json'
    state=dict(status='running',stage='preflight',fingerprint=fingerprint,started=time.time(),
               steps=0,epoch_cursor=0,video_cursor=0,generation=-1,teacher_videos=0,
               test_used=False,validation=[],best_epoch=0,bad_epochs=0,rows=[])
    if status_path.exists():
        state=json.loads(status_path.read_text())
        assert state['fingerprint']==fingerprint,'Localization protocol changed'
        state.update(status='running');state.pop('error',None)
    state['phase']=phase
    params=[p for p in runtime.model.parameters() if p.requires_grad]
    assert params and not any(p.requires_grad for p in runtime.head.parameters())
    frozen=[(p,p._version) for p in runtime.full.parameters() if not p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=policy['optimizer']['lr'],weight_decay=.01)
    durable_steps=-1
    active_clock=time.time()
    if state['generation']>=0:
        saved=torch.load(out/f"resume_{state['generation']%2}.pt",map_location='cpu',weights_only=True)
        assert saved['steps']==state['steps'] and saved['fingerprint']==fingerprint
        restore(runtime.model,runtime.head,saved['trainable']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        if str(runtime.device).startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'])
        durable_steps=state['steps']
    if continuation:
        from .continuation_control import verify_resume
        assert soft and state['generation'] >= 0, 'Continuation requires the existing V4 optimizer checkpoint'
        counters = verify_resume(continuation, state, saved)
        resume_steps = state['steps']
        resume_tensor = tensor_digest(trainable_state(runtime.model, runtime.head))
        if resume_steps == continuation['resume_steps'] and continuation.get('resume_tensor_sha256'):
            assert resume_tensor == continuation['resume_tensor_sha256']
        state['continuation'] = dict(id=continuation['id'], until_validation_plateau=True,
            entry_steps=resume_steps, optimizer_steps=counters, tensor_sha256=resume_tensor)
        state.pop('stop_reason', None)
        atomic_json(out/'continuation_restore_audit.json',dict(passed=True,steps=resume_steps,
            optimizer_steps=counters,tensor_sha256=resume_tensor,optimizer_restored=True,
            rng_restored=True,continuation_id=continuation['id']))
    if frame_selection and state['steps']==0:
        state.setdefault('initial_tensor_sha256',tensor_digest(trainable_state(runtime.model,runtime.head)))
    teacher_dir=out/'teacher';teacher_dir.mkdir(exist_ok=True)
    selections_dir=out/'selection';selections_dir.mkdir(exist_ok=True)
    def guard(reserve=300):
        if time.time()>deadline-reserve:raise TimeoutError('Localization durable-checkpoint reserve')
    def persist(weights=False):
        nonlocal durable_steps,active_clock
        if weights and durable_steps!=state['steps']:
            generation=state['generation']+1;path=out/f'resume_{generation%2}.pt'
            temporary=path.with_suffix('.tmp')
            torch.save(dict(trainable=trainable_state(runtime.model,runtime.head),optimizer=optimizer.state_dict(),
                steps=state['steps'],fingerprint=fingerprint,rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if str(runtime.device).startswith('cuda') else None),temporary)
            temporary.replace(path);state['generation']=generation;durable_steps=state['steps']
        if pointwise and state.get('stage') in ['training','validation']:
            state['epoch_active_seconds']=state.get('epoch_active_seconds',0.)+time.time()-active_clock
        active_clock=time.time()
        state['elapsed_s']=time.time()-state['started'];atomic_json(status_path,state);publish()
    def read_teacher(record):
        data=json.loads((teacher_dir/(record['name']+'.json')).read_text())
        assert data['source_sha256']==record['sha256'] and data['fingerprint']==fingerprint
        assert data['windows']==provider.metadata[record['name']]['window_count']
        assert len(data['logits'])==data['windows'] and np.isfinite(data['logits']).all()
        return data['logits']
    try:
        if not (out/'gradient_preflight.json').exists():
            if pointwise:raise ValueError('Pointwise training requires its own measured GPU preflight before teacher/training')
            fixture=min((r for r in train if r['label']),key=lambda r:(r['size'],r['name']))
            view=provider(fixture,0,True);features=view[:min(4,len(view))]
            # Synthetic teacher values are used ONLY for a derivative check,
            # never for training labels or scientific localization validation.
            t=[-3.]*(len(features)//2)+[4.]*(len(features)-len(features)//2)
            selected=selection(t,dict(threshold=0.,scale=1.),policy['selection'])
            gradient_preflight(runtime,features,selected,out/'gradient_preflight.json')
        state['stage']='teacher_scoring';persist()
        # Disable LoRA even on resume: teacher identity cannot drift with student.
        runtime.model.eval();runtime.head.eval()
        with runtime.model.disable_adapter():
            for index,record in enumerate(records):
                path=teacher_dir/(record['name']+'.json')
                if path.exists():read_teacher(record)
                else:
                    guard();started=time.time();view=provider(record,0,True)
                    z=stream_scores(runtime.forward,view,4,guard).tolist()
                    atomic_json(path,dict(name=record['name'],source_sha256=record['sha256'],
                        fingerprint=fingerprint,windows=len(z),logits=z,seconds=time.time()-started,
                        teacher='pinned_zero_shot_adapter_disabled'))
                state.update(teacher_videos=index+1,active_video=record['name'])
                if (index+1)%5==0 and not continuation:persist()
        if continuation:persist()
        reference=normal_reference([read_teacher(r) for r in train if not r['label']],policy['selection']['normal_quantile'])
        atomic_json(out/'normal_reference.json',reference)
        counts=[]
        for record in train:
            if not record['label']:continue
            chosen=(soft_targets(read_teacher(record),1,policy['loss']['minimum_probability_range']) if soft
                else (focal_select if focal else selection)(read_teacher(record),reference,policy['selection']))
            atomic_json(selections_dir/(record['name']+'.json'),chosen)
            counts.append(dict(name=record['name'],**chosen['summary']))
        atomic_json(out/'selection_audit.json',dict(videos=counts,
            positive_videos=len(counts),abstained_videos=sum(not r['positive_windows'] for r in counts),
            total_positive_windows=sum(r['positive_windows'] for r in counts),
            kind=policy['loss'].get('kind'),
            teacher_fixed=True,quantity_budget_is_outer_selection=True,test_used=False))
        if phase=='auto':
            annotations=annotation_loader()
            dev=validate_annotations(annotations,manifest,provider.metadata) if annotations else {}
            phase='train' if dev else 'prepare';state['phase']=phase
        baseline=metrics(val,{r['name']:read_teacher(r) for r in val},dev)
        if focal and phase!='prepare':
            assert len(dev)==34,'V3 is held until all 12 additional annotations are reviewed'
            teacher_scores={r['name']:read_teacher(r) for r in records}
            audit=pseudo_label_audit(dev,teacher_scores,reference,policy['selection'])
            atomic_json(out/'pseudo_label_quality_gate.json',audit)
            if not audit['passed']:
                state.update(status='quality_gate_stopped',stage='held',reason='Frozen pseudo labels fail preregistered .80/.90 development overlap thresholds',quality_gate=audit)
                persist();return state
            chosen_all={r['name']:json.loads((selections_dir/(r['name']+'.json')).read_text()) for r in train if r['label']}
            calibrated=calibration(train,teacher_scores,chosen_all,policy['loss']['gamma'],policy['loss']['background_weight'])
            atomic_json(out/'positive_calibration.json',calibrated)
            positive_weight=calibrated['positive_weight']
            baseline.update(development_metrics(val,teacher_scores,dev,provider.metadata))
        if soft and phase!='prepare':
            assert len(dev)==policy['validation']['reviewed_anomaly_videos'],'Reviewed dev labels missing'
            baseline.update(development_metrics(val,{r['name']:read_teacher(r) for r in val},dev,provider.metadata))
        atomic_json(out/'baseline_validation.json',baseline);state['baseline']=baseline
        if phase=='prepare':
            state.update(status='prepared',stage='awaiting_reviewed_dev_annotations',
                         selection_status='pending_real_temporal_development_labels')
            persist();return state
        if state.get('dev_sha256'):
            assert state['dev_sha256']==digest(annotations),'Development labels changed during training'
        state['dev_sha256']=digest(annotations)
        if fast_execution:
            if state.get('fast_execution'):assert state['fast_execution']==fast_execution,'Qualified execution changed on resume'
            state['fast_execution']=fast_execution
        state.setdefault('best_localization',baseline['within_macro_auroc'])
        if frame_selection:state.setdefault('best_frame_auroc',baseline['dev_pooled_auroc'])
        state.setdefault('selection_status','zero_shot_baseline')
        def validate(epoch):
            runtime.model.eval();runtime.head.eval()
            folder=out/'validation'/f'epoch_{epoch}';folder.mkdir(parents=True,exist_ok=True)
            state_hash=tensor_digest(trainable_state(runtime.model,runtime.head));scores={}
            for record in val:
                path=folder/(record['name']+'.json')
                if path.exists():
                    row=json.loads(path.read_text());assert row['state_sha256']==state_hash
                else:
                    guard();z=stream_scores(runtime.forward,provider(record,0,True),4,guard).tolist()
                    row=dict(name=record['name'],source_sha256=record['sha256'],state_sha256=state_hash,logits=z)
                    atomic_json(path,row)
                scores[record['name']]=row['logits']
            result=metrics(val,scores,dev);result.update(epoch=epoch,state_sha256=state_hash)
            if frame_selection:result.update(development_metrics(val,scores,dev,provider.metadata))
            atomic_json(folder/'metrics.json',result);return result
        while continuation or state['epoch_cursor']<policy['training']['max_epochs']:
            epoch=state['epoch_cursor']
            order=np.random.default_rng(policy['training']['seed']+epoch).permutation(len(train))
            class_counts={y:sum(r['label']==y for r in train) for y in [0,1]}
            accumulation=policy['training']['videos_per_step']
            if pointwise:active_clock=time.time()
            state['stage']='training';persist(weights=True)
            while state['video_cursor']<len(order):
                runtime.model.train();runtime.head.train()
                guard();optimizer.zero_grad(set_to_none=True);start=time.time();rows=[]
                indices=order[state['video_cursor']:state['video_cursor']+accumulation]
                for index in indices:
                    record=train[int(index)];view=provider(record,epoch,True)
                    chosen=json.loads((selections_dir/(record['name']+'.json')).read_text()) if record['label'] else None
                    # Class-balanced, video-normalized empirical objective;
                    # each unique video appears exactly once per full epoch.
                    weight=len(train)/(2*class_counts[record['label']]*(accumulation if pointwise else len(indices)))
                    if soft:
                        sampled=soft_plan(record['name'],record['label'],len(view),chosen,epoch,policy['sampling'],policy['training']['seed'])
                        value,info=soft_backward(runtime.forward,view,sampled,batch=policy['input']['batch'],scale=weight,guard=guard)
                    elif focal:
                        sampled=focal_plan(record['name'],record['label'],len(view),chosen,
                            epoch,policy['sampling'],policy['training']['seed'],policy['loss']['background_weight'])
                        value,info=focal_backward(runtime.forward,view,sampled,batch=policy['input']['batch'],
                            scale=weight,guard=guard,positive_weight=positive_weight,gamma=policy['loss']['gamma'])
                    elif pointwise:
                        sampled=pointwise_plan(record['name'],record['label'],len(view),chosen,
                            epoch,policy['sampling'],policy['training']['seed'])
                        value,info=backward_pointwise(runtime.forward,view,sampled,
                            batch=policy['input']['batch'],scale=weight,guard=guard,
                            anchor_weight=policy['loss']['anchor_weight'],huber_delta=policy['loss']['huber_delta'])
                    elif fast_execution:
                        value,info=backward_fixed_selection(runtime.forward,view,record['label'],chosen,
                            batch=fast_execution['batch'],scale=weight,guard=guard,
                            positive_weight=policy['loss']['positive_weight'],rank_weight=policy['loss']['rank_weight'])
                    else:
                        value,info=backward_video(runtime.forward,view,
                            lambda z,r=record,c=chosen:objective(z,r['label'],c,
                                policy['loss']['positive_weight'],policy['loss']['rank_weight']),
                            scale=weight,guard=guard)
                    rows.append(dict(name=record['name'],label=record['label'],loss=value,**info))
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite LoRA gradient')
                optimizer.step();state['steps']+=1;state['video_cursor']+=len(indices)
                if frame_selection and state['steps']==1:
                    updated=tensor_digest(trainable_state(runtime.model,runtime.head))
                    assert float(norm)>0 and updated!=state['initial_tensor_sha256'],'No actual first optimizer parameter change'
                    atomic_json(out/'first_optimizer_update.json',dict(steps=1,grad_norm=float(norm),
                        initial_tensor_sha256=state['initial_tensor_sha256'],updated_tensor_sha256=updated,
                        actual_parameters_changed=True,test_used=False))
                state['rows'].append(dict(step=state['steps'],epoch=epoch+1,seconds=time.time()-start,
                    grad_norm=float(norm),videos=rows))
                # Two generations persist every step: long videos must not
                # cause hours of optimizer work to be lost at allocation end.
                persist(weights=True)
                if continuation and state['steps']==resume_steps+1:
                    updated = tensor_digest(trainable_state(runtime.model, runtime.head))
                    assert float(norm)>0 and updated!=resume_tensor, 'No actual continuation parameter change'
                    checkpoint=out/f"resume_{state['generation']%2}.pt"
                    durable=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    from .continuation_control import verify_resume
                    counters=verify_resume(continuation,state,durable)
                    assert tensor_digest(durable['trainable'])==updated
                    atomic_json(out/'continuation_first_optimizer_update.json',dict(
                        previous_steps=resume_steps,steps=state['steps'],grad_norm=float(norm),
                        initial_tensor_sha256=resume_tensor,updated_tensor_sha256=updated,
                        actual_parameters_changed=True,durable_checkpoint_verified=True,
                        optimizer_steps=counters,continuation_id=continuation['id'],test_used=False))
                    publish()
            state['stage']='validation';persist(weights=True)
            result=validate(epoch+1)
            if pointwise:
                result['epoch_active_hours']=(state.get('epoch_active_seconds',0.)+time.time()-active_clock)/3600
                result['sampling_policy']=policy['sampling']
            folder=out/'epochs'/f'epoch_{epoch+1}';folder.mkdir(parents=True,exist_ok=True)
            torch.save(trainable_state(runtime.model,runtime.head),folder/'state.pt')
            runtime.model.save_pretrained(folder/'adapter');atomic_json(folder/'metrics.json',result)
            eligible=eligibility(result,baseline) if frame_selection else result['video_max_ap']>=baseline['video_max_ap']
            key='dev_pooled_auroc' if frame_selection else 'within_macro_auroc'
            previous=state['best_frame_auroc'] if frame_selection else state['best_localization']
            improved=eligible and result[key]>previous
            significant=improved and result[key]>previous+policy['training']['min_delta']
            if improved:
                state.update(best_epoch=epoch+1,best_localization=result['within_macro_auroc'],
                    selected_artifact=str(folder.relative_to(out)),selection_status='selected_on_real_localization_dev')
                if frame_selection:state.update(best_frame_auroc=result[key],selection_status='selected_on_frame_dev_with_three_guards')
            state['bad_epochs']=0 if significant else state['bad_epochs']+1
            state['validation'].append(result);state.update(epoch_cursor=epoch+1,video_cursor=0)
            atomic_json(out/'selection.json',dict(best_epoch=state['best_epoch'],
                within_macro_auroc=state['best_localization'],status=state['selection_status'],
                dev_sha256=state['dev_sha256'],test_used=False))
            if frame_selection:
                atomic_json(out/'selection.json',dict(best_epoch=state['best_epoch'],dev_pooled_auroc=state['best_frame_auroc'],
                    status=state['selection_status'],dev_sha256=state['dev_sha256'],test_used=False,
                    eligible_for_full_test=state['best_epoch']>0,guards=['within_macro_auroc','dev_pooled_ap','video_max_ap']))
            persist(weights=True)
            if pointwise:
                measured=state.get('epoch_active_seconds',0.)/3600
                state['last_epoch_active_hours']=measured;state['epoch_active_seconds']=0.
                if measured>policy['resources']['epoch_max_hours']:
                    state.update(status='speed_gate_stopped',stage='held',reason='Measured epoch exceeded user three-hour limit')
                    break
            if epoch+1>=policy['training']['min_epochs'] and state['bad_epochs']>=policy['training']['patience']:
                state.update(status='complete',stage='finished',stop_reason='localization_validation_plateau');break
        else:state.update(status='budget_complete',stage='finished',stop_reason='fixed_epoch_limit_not_convergence')
        state['frozen_parameters_unchanged']=all(p._version==version for p,version in frozen)
        assert state['frozen_parameters_unchanged']
        persist(weights=True)
    except TimeoutError as error:
        # Discard a partially accumulated step; reload the last committed
        # optimizer state on resume. The cursor advances only after opt.step.
        optimizer.zero_grad(set_to_none=True)
        state.update(status='budget_stopped',reason=str(error));persist()
    except Exception:
        import traceback
        optimizer.zero_grad(set_to_none=True)
        state.update(status='error',error=traceback.format_exc());persist();raise
    return state
