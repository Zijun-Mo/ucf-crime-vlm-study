"""Uniform, resolution-aware video overview sheets and native multimodal inputs."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CONTEXT_INSTRUCTION = (
    'The contact sheets, when provided, summarize this same full video in chronological order, '
    'left to right and top to bottom. They provide scene and event context only. '
    'Judge ONLY the following local video clip. An abnormal event elsewhere in the overview '
    'does not make this local clip abnormal. Do not infer actions not visible in the local clip.\n\n'
)


def sheet_plan(width, height, n_frames, fps, token_budget=768):
    if min(width, height, n_frames, fps) <= 0:
        raise ValueError('Invalid source video dimensions or timeline')
    wanted = min(8, max(4, math.ceil(n_frames / fps / 8)))
    indices = np.unique(np.rint(np.linspace(0, n_frames - 1, min(wanted, n_frames))).astype(int)).tolist()
    pages = math.ceil(len(indices) / 4)
    # A 32x32 image region becomes one merged visual token for the pinned model.
    for longest in range(320, 95, -32):
        factor = min(1., longest / max(width, height))
        image_w, image_h = max(1, round(width * factor)), max(1, round(height * factor))
        cell_w = math.ceil(image_w / 32) * 32
        cell_h = math.ceil((image_h + 24) / 32) * 32
        canvas_w, canvas_h = 2 * cell_w, 2 * cell_h
        tokens = max(64, canvas_w * canvas_h // 1024) * pages
        if tokens <= token_budget:
            return dict(frame_indices=indices, pages=pages, source_width=width, source_height=height,
                        image_width=image_w, image_height=image_h, cell_width=cell_w, cell_height=cell_h,
                        canvas_width=canvas_w, canvas_height=canvas_h, estimated_image_tokens=tokens,
                        image_token_budget=token_budget, aspect_ratio_preserved=True, upscaled=False)
    raise ValueError('Overview cannot fit image-token budget')


def make_sheets(path, output):
    import cv2
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot decode original video')
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        plan=sheet_plan(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),n,fps)
        images=[];files=[]
        for page in range(plan['pages']):
            sheet=Image.new('RGB',(plan['canvas_width'],plan['canvas_height']),(32,32,32));draw=ImageDraw.Draw(sheet)
            for j,index in enumerate(plan['frame_indices'][page*4:page*4+4]):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise ValueError('Overview frame decode failed')
                im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize(
                    (plan['image_width'],plan['image_height']),Image.Resampling.LANCZOS)
                x=(j%2)*plan['cell_width'];y=(j//2)*plan['cell_height']
                sheet.paste(im,(x+(plan['cell_width']-im.width)//2,y+24))
                draw.text((x+4,y+4),f'{index/fps:.1f}s',fill='white')
            target=output/f'overview_{page}.jpg';sheet.save(target,quality=92)
            images.append(sheet);files.append(str(target))
        return images,dict(**plan,fps=fps,n_frames=n,files=files,
                           timestamps_s=[i/fps for i in plan['frame_indices']])
    finally:cap.release()


def set_context(runtime, images, prompt, template_video):
    """Encode overview once per video; preserve native video M-RoPE and readout."""
    import torch
    content=[{'type':'text','text':CONTEXT_INSTRUCTION}]
    content.extend({'type':'image','image':im} for im in images)
    content.extend([{'type':'text','text':prompt},{'type':'video','path':str(template_video)}])
    batch=runtime.proc.apply_chat_template([[{'role':'user','content':content}]],tokenize=True,
        add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False,
        processor_kwargs={'fps':2.,'padding':True})
    t={k:v.to(runtime.device) for k,v in batch.items() if isinstance(v,torch.Tensor)
       and k not in ['pixel_values','pixel_values_videos']}
    with torch.no_grad():
        emb=runtime.full.model.get_input_embeddings()(t['input_ids']).detach().clone()
        mask=(t['input_ids']==runtime.full.config.image_token_id)[0]
        if images:
            features=runtime.full.model.get_image_features(batch['pixel_values'].to(runtime.device),
                t['image_grid_thw'],return_dict=True).pooler_output
            features=torch.cat(features) if isinstance(features,(list,tuple)) else features
            assert mask.sum().item()==len(features)
            emb[:,mask]=features.to(emb.dtype)
        else:assert not mask.any()
        runtime.pos=runtime.full.model.compute_3d_position_ids(input_ids=t['input_ids'],inputs_embeds=emb,
            image_grid_thw=t.get('image_grid_thw'),video_grid_thw=t['video_grid_thw'],
            attention_mask=t['attention_mask'],mm_token_type_ids=t['mm_token_type_ids'])
    runtime.template=t;runtime.emb=emb;runtime.mask=(t['input_ids']==runtime.full.config.video_token_id)[0]
    assert runtime.mask.sum().item()==196
    assert mask.sum().item()<=768,'Processor unexpectedly exceeded overview budget'
    return dict(image_tokens=int(mask.sum()),local_video_tokens=int(runtime.mask.sum()),
                sequence_tokens=t['input_ids'].shape[1],image_grid=t.get('image_grid_thw',torch.empty(0)).cpu().tolist()),batch


class NativeContextScorer:
    """Official multimodal forward for every arm; cache pixels, not image embeddings."""
    def __init__(self,runtime,images,prompt,template_video,context=True):
        import torch
        self.runtime=runtime
        content=([{'type':'text','text':CONTEXT_INSTRUCTION}] if context else [])
        content.extend({'type':'image','image':im} for im in images)
        content.extend([{'type':'text','text':prompt},{'type':'video','path':str(template_video)}])
        batch=runtime.proc.apply_chat_template([[{'role':'user','content':content}]],tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False,
            processor_kwargs={'fps':2.,'padding':True})
        self.template={k:v.to(runtime.device) for k,v in batch.items() if isinstance(v,torch.Tensor)
                       and k!='pixel_values_videos'}
        ids=self.template['input_ids']
        self.info=dict(image_tokens=int((ids==runtime.full.config.image_token_id).sum()),
                       local_video_tokens=int((ids==runtime.full.config.video_token_id).sum()),
                       sequence_tokens=ids.shape[1],execution='native_full_multimodal_forward',
                       image_grid=self.template.get('image_grid_thw',torch.empty(0)).cpu().tolist())
        assert self.info['local_video_tokens']==196 and self.info['image_tokens']<=768
        assert bool(images)==bool(self.info['image_tokens'])

    def score(self,windows,batch=4,guard=lambda:None):
        import torch
        output=[]
        with torch.inference_mode():
            for start in range(0,len(windows),batch):
                guard();selected=windows[start:start+batch];n=len(selected)
                selected=selected+[selected[-1]]*(batch-n)
                inputs={k:v.repeat(batch,*([1]*(v.ndim-1))) for k,v in self.template.items()}
                video=self.runtime.proc.video_processor(videos=selected,do_sample_frames=False,
                    return_tensors='pt',input_data_format='channels_last')
                assert torch.equal(video['video_grid_thw'].to(self.runtime.device),inputs['video_grid_thw'])
                inputs['pixel_values_videos']=video['pixel_values_videos'].to(self.runtime.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=self.runtime.full.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
                z=self.runtime.head(hidden.float()).flatten()[:n]
                assert torch.isfinite(z).all()
                output.extend(z.cpu().tolist())
        return output
