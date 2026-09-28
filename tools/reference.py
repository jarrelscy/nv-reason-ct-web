"""PyTorch reference run of NV-Reason-CT, saving the intermediates the browser port is checked against.

  /data/huggingface/nv-reason-ct/venv/bin/python tools/reference.py CT.nii.gz OUT_DIR [chest|abdomen]

Saves: crop.npy (normalised 192^3 input), input_ids.npy, position_ids.npy, vision.npy (13824x2560 tokens
fed to the LLM), logits0.npy (first-step logits) and ref.json (prompt, greedy outputs, timings).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = '/data/huggingface/nv-reason-ct/hf'
ct, out = sys.argv[1], Path(sys.argv[2]); region = sys.argv[3] if len(sys.argv) > 3 else 'abdomen'
out.mkdir(parents=True, exist_ok=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

def inputs_for(text, thinking):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return prompt, proc(text=prompt, images3d=[ct], anatomy_region=region, return_tensors='pt').to('cuda')

res = {'region': region, 'runs': []}
prompt, inp = inputs_for(f'Write a structured {region} CT report.', False)
print({k: (tuple(v.shape), v.dtype) for k, v in inp.items() if hasattr(v, 'shape')})
np.save(out / 'crop.npy', inp['pixel_values'].float().cpu().numpy())
np.save(out / 'input_ids.npy', inp['input_ids'].cpu().numpy())
with torch.inference_mode():
    m = model.model
    feats = m.get_image_features(inp['pixel_values'], inp.get('image_grid_thw')).pooler_output
    np.save(out / 'vision.npy', torch.cat(feats).float().cpu().numpy())
    pos, _ = m.get_rope_index(inp['input_ids'], inp['mm_token_type_ids'], inp.get('image_grid_thw'), None, inp['attention_mask'])
    np.save(out / 'position_ids.npy', pos.cpu().numpy())
    logits = model(**inp).logits[0, -1].float().cpu().numpy()
    np.save(out / 'logits0.npy', logits)
for text, thinking, n in [(f'Write a structured {region} CT report.', False, 1024),
                          ('Is there any abnormality in the liver? Answer only Yes or No.', False, 8),
                          (f'Provide a full reasoning analysis of this {region} CT.', True, 2048)]:
    prompt, inp = inputs_for(text, thinking)
    t0 = time.time()
    with torch.inference_mode():
        ids = model.generate(**inp, max_new_tokens=n, do_sample=False, use_cache=True)
    new = ids[0, inp['input_ids'].shape[1]:].tolist()
    txt = proc.batch_decode([new], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    res['runs'].append({'prompt': prompt, 'thinking': thinking, 'tokens': new, 'text': txt, 'seconds': time.time() - t0})
    print(f'--- {text} ({len(new)} tokens, {time.time() - t0:.1f}s)\n{txt[:1500]}\n')
(out / 'ref.json').write_text(json.dumps(res, indent=1))
