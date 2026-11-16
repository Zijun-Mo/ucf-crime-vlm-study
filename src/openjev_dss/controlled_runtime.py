"""BF16, no-prefix-cache runtime for the frozen remaining-experiment protocol.

The original historical runtime is left unchanged. Template, model and video
feature numerical gates must pass on train/development fixtures before full jobs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def load_trainable_snapshot(model, head, state):
    """Restore fine-tuned weights also when inference has frozen their gradients."""
    used = set()
    with torch.no_grad():
        for prefix, module in [('model.', model), ('head.', head)]:
            for name, parameter in module.named_parameters():
                key = prefix + name
                if key in state:
                    parameter.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))
                    used.add(key)
    if used != set(state):
        raise ValueError('Snapshot parameter names do not match the inference model: ' + str(sorted(set(state) - used)))


class CandidateReadout(torch.nn.Module):
    """Frozen native Yes-minus-No readout without allocating vocabulary logits."""
    def __init__(self, lm_head, yes, no):
        super().__init__()
        self.register_buffer('weight', lm_head.weight[[yes, no]].detach().float().clone())
        bias = lm_head.bias[[yes, no]].detach().float().clone() if lm_head.bias is not None else None
        self.register_buffer('bias', bias)

    def forward(self, hidden):
        with torch.autocast('cuda', enabled=False):
            logits = torch.nn.functional.linear(hidden.float(), self.weight, self.bias)
        return (logits[:, 0] - logits[:, 1]).unsqueeze(-1)


def build_template(processor, prompt, duration, work):
    import cv2
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    filename = work / f'template_{duration}s.mp4'
    writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*'mp4v'), 2, (224, 224))
    if not writer.isOpened():
        raise RuntimeError('Cannot create template video')
    for _ in range(duration * 2):
        writer.write(np.zeros((224, 224, 3), np.uint8))
    writer.release()
    messages = [[{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                            {'type': 'video', 'path': str(filename)}]}]]
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors='pt', enable_thinking=False,
                                         processor_kwargs={'fps': 2.0, 'padding': True})
    return {k: v for k, v in batch.items() if k != 'pixel_values_videos'}


class ControlledRuntime:
    def __init__(self, base, prompt, duration=4, published_checkpoint=None,
                 train=False, adapter=None, state=None, work='/content/ucf_templates', device='cuda'):
        import transformers.utils.import_utils as iu
        import transformers.utils as tu
        iu.is_torchao_available = lambda *a, **k: False
        tu.is_torchao_available = iu.is_torchao_available
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, PeftModel, get_peft_model
        import peft.import_utils as piu
        import peft.tuners.lora.torchao as pta
        piu.is_torchao_available = lambda *a, **k: False
        pta.is_torchao_available = piu.is_torchao_available
        torch.manual_seed(42)
        self.device, self.duration = device, duration
        self.proc = AutoProcessor.from_pretrained(base)
        self.proc.tokenizer.padding_side = 'right'
        self.full = AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
        self.full.requires_grad_(False)
        self.full.eval()
        self.model = self.full.model.language_model
        if published_checkpoint:
            self.model = PeftModel.from_pretrained(self.model, str(Path(published_checkpoint) / 'adapter'), is_trainable=train)
        elif adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=train)
        elif train:
            self.model = get_peft_model(self.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
                target_modules=['q_proj', 'out_proj', 'o_proj', 'v_proj', 'k_proj', 'in_proj_qkv']))
        self.full.model.language_model = self.model
        if published_checkpoint:
            self.head = torch.nn.Linear(self.full.config.text_config.hidden_size, 1, dtype=torch.float32).to(device)
            self.head.load_state_dict(torch.load(Path(published_checkpoint) / 'head.pt', map_location=device, weights_only=True))
            self.head.requires_grad_(train)
        else:
            candidates = [self.proc.tokenizer.encode(word, add_special_tokens=False) for word in ['Yes', 'No']]
            if not all(len(ids) == 1 for ids in candidates):
                raise RuntimeError('Yes/No candidate must be one token')
            self.head = CandidateReadout(self.full.lm_head, candidates[0][0], candidates[1][0]).to(device)
        if train:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if state:
            load_trainable_snapshot(self.model, self.head, torch.load(state, map_location='cpu', weights_only=True))
        self.set_prompt(prompt, duration, work)
        self.trainable_parameters = sum(p.numel() for p in self.full.parameters() if p.requires_grad)
        self.trainable_parameters += sum(p.numel() for p in self.head.parameters() if p.requires_grad)

    def set_prompt(self, prompt, duration=4, work='/content/ucf_templates'):
        self.duration = duration
        template = build_template(self.proc, prompt, duration, work)
        self.template = {k: v.to(self.device) for k, v in template.items() if isinstance(v, torch.Tensor)}
        t = self.template
        self.emb = self.full.model.get_input_embeddings()(t['input_ids']).detach()
        self.mask = (t['input_ids'] == self.full.config.video_token_id)[0]
        if self.mask.sum().item() != duration * 49:
            raise RuntimeError('Unexpected visual token count')
        self.pos = self.full.model.compute_3d_position_ids(
            input_ids=t['input_ids'], inputs_embeds=self.emb, video_grid_thw=t['video_grid_thw'],
            attention_mask=t['attention_mask'], mm_token_type_ids=t['mm_token_type_ids'])
        descriptor = {k: v.cpu().tolist() for k, v in self.template.items()}
        self.template_sha256 = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()

    def forward(self, visual):
        b = len(visual)
        e = self.emb.expand(b, -1, -1).clone()
        e[:, self.mask] = visual.to(self.device, dtype=e.dtype, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = self.model(inputs_embeds=e, position_ids=self.pos.expand(-1, b, -1),
                                attention_mask=self.template['attention_mask'].expand(b, -1),
                                use_cache=False, return_dict=True).last_hidden_state[:, -1]
        return self.head(hidden.float()).flatten()

    @torch.no_grad()
    def vision(self, windows):
        inputs = self.proc.video_processor(videos=list(windows), do_sample_frames=False,
                                          return_tensors='pt', input_data_format='channels_last')
        output = self.full.model.get_video_features(inputs['pixel_values_videos'].to(self.device),
                    inputs['video_grid_thw'].to(self.device), return_dict=True).pooler_output
        output = torch.cat(output) if isinstance(output, (tuple, list)) else output
        frame_count = len(windows[0])
        assert frame_count % 2 == 0 and all(len(w) == frame_count for w in windows)
        return output.reshape(len(windows), (frame_count // 2) * 49, -1).cpu()
