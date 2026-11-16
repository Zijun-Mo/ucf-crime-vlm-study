"""Reuse certified local-video features while images retain native execution."""
import hashlib,json
from pathlib import Path
import torch


def cache_signature(runtime,record,metadata,revision):
    from .study_visuals import StudyVisualStore
    store=StudyVisualStore(runtime,'/content/storyboard_canonical',{},'/content/storyboard_features',revision)
    signature={**store.signature,'representation':'temporal_patch_pairs','pair_batch':16,'duration_gate':4,
               'canonical_sha256':metadata['canonical_sha256'],'source_sha256':record['sha256']}
    return signature,hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()+'.pt'


class PairFeatureView:
    def __init__(self,path,signature,count):
        entry=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert entry['signature']==signature
        self.blocks=entry['features'];self.count=count
        assert self.blocks.shape==(count+1,49,2048) and self.blocks.dtype==torch.bfloat16

    def __len__(self):return self.count

    def __getitem__(self,key):
        indices=torch.as_tensor(list(range(*key.indices(self.count))))[:,None]+torch.arange(4)[None,:]
        result=self.blocks[indices.clamp(max=self.count)].reshape(len(indices),196,2048)
        assert torch.isfinite(result).all()
        return result


def score_cached_video(scorer,features,batch=4,guard=lambda:None):
    """Only get_video_features is substituted; native image/M-RoPE/LM stay intact."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    runtime=scorer.runtime;model=runtime.full.model;original=model.get_video_features;output=[]
    with torch.inference_mode():
        try:
            for start in range(0,len(features),batch):
                guard();visual=features[start:start+batch];n=len(visual)
                if n<batch:visual=torch.cat([visual,visual[-1:].expand(batch-n,-1,-1)])
                visual=visual.to(runtime.device)
                def cached(*args,**kwargs):
                    return BaseModelOutputWithPooling(pooler_output=tuple(visual.unbind(0)))
                model.get_video_features=cached
                inputs={k:v.repeat(batch,*([1]*(v.ndim-1))) for k,v in scorer.template.items()}
                # The native forward still performs its video placeholder and
                # length checks. No video pixels are encoded in this branch.
                inputs['pixel_values_videos']=torch.empty(0,device=runtime.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
                z=runtime.head(hidden.float()).flatten()[:n];assert torch.isfinite(z).all()
                output.extend(z.cpu().tolist())
        finally:model.get_video_features=original
    return output
