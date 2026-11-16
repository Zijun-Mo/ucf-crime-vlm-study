"""Bounded actual-model A/B smoke. Invoked in a fresh Colab subprocess."""
import os,sys,time,json,math,hashlib,shutil,traceback,copy,gc,subprocess
from pathlib import Path
ROOT=Path('/content/drive/MyDrive/OpenJev_UCFCrime');OUT=ROOT/'weak_dss_smoke_v1'
DEADLINE=float(os.environ['DSS_DEADLINE']);STATE={'status':'running','stage':'packages','started':time.time(),'arms':{'A':[],'B':[]}}
def persist():
    STATE['elapsed_s']=time.time()-STATE['started'];Path('/content/dss_status.json').write_text(json.dumps(STATE,indent=2))
    try:(OUT/'gpu_status.json').write_text(json.dumps(STATE,indent=2))
    except OSError:pass

def guard(reserve=120):
    if time.time()>DEADLINE-reserve:raise TimeoutError('Smoke budget reserve reached')

def main():
    try:
        persist()
        dep=Path('/content/dss_deps');dep.mkdir(exist_ok=True)
        if not (dep/'transformers').exists():
            subprocess.run([sys.executable,'-m','pip','install','-q','--target',str(dep),'--upgrade','--no-deps','transformers==5.10.2','peft==0.19.1','av==19.0.0','einops==0.8.2','fla-core==0.5.2','tokenizers==0.22.2'],check=True,timeout=240)
        sys.path.insert(0,str(dep))
        import transformers.utils.import_utils as iu,transformers.utils as tu
        iu.is_torchao_available=lambda *a,**kw:False;tu.is_torchao_available=iu.is_torchao_available
        import torch,numpy as np,cv2
        from transformers import AutoModelForImageTextToText,AutoProcessor
        from peft import PeftModel
        import peft.import_utils as piu,peft.tuners.lora.torchao as pta
        piu.is_torchao_available=lambda *a,**kw:False;pta.is_torchao_available=piu.is_torchao_available
        import transformers.models.qwen3_5.modeling_qwen3_5 as qm
        from .losses import loss
        from .training import two_pass,trainable_state,restore
        from .data import sample_starts
        torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        STATE.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,precision='bf16',prefix_cache=False,use_cache=False,checkpoint='nonreentrant');persist()
        work=Path('/content/dss_gpu');work.mkdir(exist_ok=True)
        def cp(src,dst):
            src=Path(src);dst=Path(dst)
            for attempt in range(4):
                guard()
                try:
                    if dst.exists() and dst.stat().st_size==src.stat().st_size:return str(dst)
                    with src.open('rb') as r,dst.open('wb') as w:shutil.copyfileobj(r,w,4*1024*1024)
                    assert dst.stat().st_size==src.stat().st_size;return str(dst)
                except OSError:
                    if attempt==3:raise
                    time.sleep(3)
        STATE['stage']='load_base';persist()
        base=ROOT/'models/base/15852e8c16360a2fea060d615a32b45270f8a8fc';original=ROOT/'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint'
        shutil.copytree(base,work/'base',dirs_exist_ok=True,copy_function=cp)
        shutil.copytree(original,work/'checkpoint',dirs_exist_ok=True,copy_function=cp)
        proc=AutoProcessor.from_pretrained(work/'base')
        full=AutoModelForImageTextToText.from_pretrained(work/'base',dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
        manifest=json.loads((OUT/'manifest.json').read_text());records=manifest['smoke']
        assert json.loads((OUT/'cpu_status.json').read_text())['status']=='complete'
        for p in full.parameters():p.requires_grad_(False)
        template=torch.load(ROOT/'formal_stride1/template.pt',map_location='cpu',weights_only=True)
        template={k:v[:1].cuda() for k,v in template.items() if isinstance(v,torch.Tensor)}
        assert template['input_ids'].shape==(1,330)
        emb=full.model.get_input_embeddings()(template['input_ids']).detach()
        placeholder=(template['input_ids']==full.config.video_token_id)[0];assert int(placeholder.sum())==196
        pos=full.model.compute_3d_position_ids(input_ids=template['input_ids'],inputs_embeds=emb,video_grid_thw=template['video_grid_thw'],attention_mask=template['attention_mask'],mm_token_type_ids=template['mm_token_type_ids'])
        full.eval();STATE['stage']='features';persist();features={};cache_checks=[];metadata={}
        (OUT/'features').mkdir(exist_ok=True)
        with torch.no_grad():
            for rec in records:
                guard();name=rec['name'];m=json.loads((OUT/'metadata'/(name+'.json')).read_text());metadata[name]=m
                local=work/name;cp(OUT/'canonical'/name,local)
                assert hashlib.sha256(local.read_bytes()).hexdigest()==m['canonical_sha256']
                cap=cv2.VideoCapture(str(local));frames=[]
                while True:
                    ok,f=cap.read()
                    if not ok:break
                    frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
                cap.release();local.unlink();frames=np.stack(frames);valid=m['sampled_frames'];assert len(frames)==valid+1
                starts=sample_starts(m['window_count'],0,name);vf=[]
                cached_path=OUT/'features'/(name+'.pt')
                if cached_path.exists():
                    cached=torch.load(cached_path,map_location='cpu',weights_only=True)
                    assert cached['source_sha256']==rec['sha256'] and cached['canonical_sha256']==m['canonical_sha256'] and cached['starts_s']==starts and cached['precision']=='bf16'
                    features[name]=cached['features'];STATE['features_videos']=len(features);persist();continue
                for start in starts:
                    window=np.stack([frames[2*start+j] if 2*start+j<valid else frames[valid] for j in range(8)])
                    data=proc.video_processor(videos=[window],do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
                    assert data['video_grid_thw'].tolist()==[[4,14,14]]
                    out=full.model.get_video_features(data['pixel_values_videos'].cuda(),data['video_grid_thw'].cuda(),return_dict=True).pooler_output
                    out=torch.cat(out) if isinstance(out,(list,tuple)) else out;vf.append(out.detach().cpu())
                    if not cache_checks:
                        direct=out.clone();cached=vf[-1].cuda();cache_checks.append({'video':name,'feature_max_abs_diff':float((direct-cached).abs().max()),'starts_s':starts})
                features[name]=torch.stack(vf)
                torch.save({'features':features[name],'starts_s':starts,'source_sha256':rec['sha256'],'canonical_sha256':m['canonical_sha256'],'precision':'bf16','epoch':0,'training_prefix_cache':False},OUT/'features'/(name+'.pt'))
                STATE['features_videos']=len(features);persist()
        # Independently recompute a frozen-vision window from canonical pixels.
        first=records[0];name=first['name'];meta=metadata[name]
        saved=torch.load(OUT/'features'/(name+'.pt'),map_location='cpu',weights_only=True)
        local=work/name;cp(OUT/'canonical'/name,local);cap=cv2.VideoCapture(str(local));decoded=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release();local.unlink();start=saved['starts_s'][0];valid=meta['sampled_frames']
        window=np.stack([decoded[2*start+j] if 2*start+j<valid else decoded[valid] for j in range(8)])
        data=proc.video_processor(videos=[window],do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
        with torch.no_grad():
            direct=full.model.get_video_features(data['pixel_values_videos'].cuda(),data['video_grid_thw'].cuda(),return_dict=True).pooler_output
            direct=torch.cat(direct) if isinstance(direct,(list,tuple)) else direct
            torch.testing.assert_close(direct.cpu(),saved['features'][0],rtol=0,atol=0)
        STATE['cache_gate']={'passed':True,'scope':'independent full-window frozen vision vs Drive-restored cache','max_abs_diff':float((direct.cpu()-saved['features'][0]).abs().max()),'video':name}
        (OUT/'cache_gate.json').write_text(json.dumps(STATE['cache_gate'],indent=2))
        STATE['stage']='adapter';persist()
        model=PeftModel.from_pretrained(full.model.language_model,work/'checkpoint/adapter',is_trainable=True)
        full.model.language_model=model
        from safetensors.torch import load_file
        published=load_file(str(work/'checkpoint/adapter/adapter_model.safetensors'))
        for n,p in model.named_parameters():
            if 'lora_' in n:
                key=n.replace('.default.','.')
                assert key in published,key
                torch.testing.assert_close(p.detach().cpu().float(),published[key].float(),rtol=0,atol=0)
        conf=model.peft_config['default'];assert (conf.r,conf.lora_alpha,conf.lora_dropout)==(8,16,0)
        # Original adapter may include visual projections: freeze them explicitly.
        for n,p in model.named_parameters():
            if '.visual.' in n:p.requires_grad_(False)
            if p.requires_grad:assert 'lora_' in n,n
        head=torch.nn.Linear(2048,1,dtype=torch.float32).cuda();head.load_state_dict(torch.load(work/'checkpoint/head.pt',map_location='cuda',weights_only=True))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();head.train()
        lm=model
        counts={}
        for module in lm.modules():
            if isinstance(module,qm.Qwen3_5GatedDeltaNet):
                fn=module.chunk_gated_delta_rule;label=fn.__module__+'.'+fn.__name__
                assert label.startswith('fla.'),label
                def counted(*a,_fn=fn,_label=label,**kw):
                    counts[_label]=counts.get(_label,0)+1;return _fn(*a,**kw)
                module.chunk_gated_delta_rule=counted
        fp32_gate=False
        def forward(v):
            guard(90);b=len(v);dtype=next(lm.parameters()).dtype;e=emb.to(dtype).expand(b,-1,-1).clone();e[:,placeholder]=v.to('cuda',dtype=dtype);e.requires_grad_(True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32_gate):
                h=lm(inputs_embeds=e,position_ids=pos.expand(-1,b,-1),attention_mask=template['attention_mask'].expand(b,-1),use_cache=False,return_dict=True).last_hidden_state[:,-1]
            return head(h.float()).flatten()
        adapters=[p for p in model.parameters() if p.requires_grad];params=adapters+list(head.parameters())
        init=trainable_state(model,head)
        def frozen_hash():
            h=hashlib.sha256()
            for n,p in full.named_parameters():
                if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        frozen=frozen_hash();STATE['frozen_initial_sha256']=frozen;STATE['trainable_numel']=sum(p.numel() for p in params);persist()
        # Actual model graph vs score-leaf chain-rule gate, 4 windows.
        gate=[features[records[0]['name']][:2],features[next(r['name'] for r in records if r['label'])][:2]]
        STATE['stage']='gradient_gate';persist();guard(150)
        # Gate the chain rule in FP32; BF16 no-grad/grad kernels can differ numerically.
        fp32_gate=True
        original_dtypes={n:p.dtype for n,p in model.named_parameters() if not p.requires_grad}
        for p in model.parameters():
            if not p.requires_grad:p.data=p.data.float()
        model.zero_grad();head.zero_grad();values=[torch.cat([forward(w[j:j+1]) for j in range(len(w))]) for w in gate]
        value,_=loss(values,[0,1],True);value.backward()
        expected=[None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
        model.zero_grad();head.zero_grad();two_pass(forward,gate,[0,1],True,chunk=1)
        torch.cuda.synchronize()
        num=den=maximum=0.
        for p,g in zip(params,expected):
            if g is None:assert p.grad is None;continue
            assert p.grad is not None
            diff=p.grad.detach().cpu()-g;num+=float(diff.square().sum());den+=float(g.square().sum());maximum=max(maximum,float(diff.abs().max()))
            torch.testing.assert_close(p.grad.detach().cpu(),g,rtol=.03,atol=.002)
        STATE['gradient_gate']={'passed':True,'relative_l2':math.sqrt(num/max(den,1e-30)),'max_abs':maximum,'actual_model':True,'precision':'FP32; BF16 gradients are numerically sensitive'}
        for n,p in model.named_parameters():
            if not p.requires_grad:p.data=p.data.to(original_dtypes[n])
        fp32_gate=False;persist()
        # Compile BF16 forward/backward before recording a small warm trace.
        model.zero_grad();head.zero_grad();forward(gate[0][:1]).sum().backward();torch.cuda.synchronize()
        model.zero_grad();head.zero_grad()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            forward(gate[0][:1]).sum().backward();torch.cuda.synchronize()
        STATE['actual_operators']=[e.key for e in profile.key_averages()]
        model.zero_grad();head.zero_grad();persist()
        # Restore independent A/B weights, optimizer and RNG for every switch.
        def optimizer():return torch.optim.AdamW([{'params':adapters,'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
        armstate={a:{'weights':copy.deepcopy(init),'optimizer':None,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'steps':0} for a in ['A','B']}
        normals=[r for r in records if r['split']=='train' and not r['label']];abnormals=[r for r in records if r['split']=='train' and r['label']];assert len(normals)==len(abnormals)==8
        order=['A','A','B','B']+[a for _ in range(8) for a in ['A','B']]
        STATE['stage']='training';persist()
        STATE['target_steps_reached']=True
        for arm in order:
            if time.time()>DEADLINE-150:
                STATE['target_steps_reached']=False;break
            a=armstate[arm];restore(model,head,a['weights']);opt=optimizer()
            if a['optimizer'] is not None:opt.load_state_dict(a['optimizer'])
            torch.set_rng_state(a['rng']);torch.cuda.set_rng_state(a['cuda_rng']);opt.zero_grad(set_to_none=True)
            start=time.time();stats=[];step=a['steps'];torch.cuda.reset_peak_memory_stats()
            for micro in range(4):
                idx=(step*4+micro)%8;bags=[features[normals[idx]['name']],features[abnormals[idx]['name']]]
                obj,info=two_pass(forward,bags,[0,1],arm=='B',scale=.25,chunk=1);stats.append(info)
            for p in params:assert p.grad is None or torch.isfinite(p.grad).all()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapters)
            assert head.weight.grad is not None and head.weight.grad.abs().sum()>0
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();torch.cuda.synchronize()
            a.update(weights=trainable_state(model,head),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),steps=step+1)
            row={'step':step+1,'elapsed_s':time.time()-start,'loss_mil':sum(x['mil'] for x in stats)/4,'loss_ss':sum(x['ss'] for x in stats)/4,'gradient_norm':norm,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'stats':stats,'scored_windows':sum(sum(b['n'] for b in x['bags']) for x in stats),'active_windows':sum(x['active_windows'] for x in stats)}
            row['windows_per_s']=row['scored_windows']/row['elapsed_s'];STATE['arms'][arm].append(row);STATE['active_arm']=arm;persist()
            # Durable trainable-only resume with both optimizer and random states.
            torch.save(a,OUT/(arm+'_resume.pt'))
        STATE['stage']='verification';persist();guard(100)
        STATE['frozen_unchanged']=frozen_hash()==frozen;assert STATE['frozen_unchanged']
        for arm,a in armstate.items():
            restore(model,head,a['weights']);folder=OUT/arm;folder.mkdir(exist_ok=True);model.save_pretrained(folder/'adapter');torch.save(head.state_dict(),folder/'head.pt')
            changed_lora=sum(float((a['weights'][k]-init[k]).abs().sum()) for k in init if k.startswith('model.'))
            changed_head=sum(float((a['weights'][k]-init[k]).abs().sum()) for k in init if k.startswith('head.'))
            assert changed_lora>0 and changed_head>0;STATE[arm+'_updates']={'lora_abs_sum':changed_lora,'head_abs_sum':changed_head}
            # Restore saved trainable weights and compare independent reload outputs.
            with torch.no_grad():before=forward(gate[0][:1]).cpu()
            saved=torch.load(OUT/(arm+'_resume.pt'),map_location='cpu',weights_only=False);restore(model,head,init);restore(model,head,saved['weights'])
            with torch.no_grad():after=forward(gate[0][:1]).cpu()
            torch.testing.assert_close(before,after,rtol=0,atol=0);STATE[arm+'_restore_verified']=True
        STATE.update(status='complete' if min(len(v) for v in STATE['arms'].values())>=2 else 'insufficient_steps',stage='done',kernel_counts=counts,transformers_fast_path_flag=bool(qm.is_fast_path_available),max_epochs_formal=3,near_duplicate_full_audit='pending; formal training gated')
    except TimeoutError as e:STATE.update(status='budget_stopped',error=str(e))
    except BaseException:STATE.update(status='error',error=traceback.format_exc())
    finally:persist();print(json.dumps(STATE),flush=True)
if __name__=='__main__':main()
