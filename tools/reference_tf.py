"""Teacher-forced PyTorch logits over the reference report tokens (bf16 and fp32), for tools/check_decoder.py --tf.

  venv/bin/python tools/reference_tf.py REF_DIR
Saves tf_bf16.npy and tf_fp32.npy: [n_generated, vocab] float16 logits predicting each generated token.
"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForImageTextToText

ref = sys.argv[1]; MODEL = '/data/huggingface/nv-reason-ct/hf'; IMAGE = 248056
ids = torch.from_numpy(np.load(f'{ref}/input_ids.npy')).cuda(); pos = torch.from_numpy(np.load(f'{ref}/position_ids.npy')).cuda()
vis = torch.from_numpy(np.load(f'{ref}/vision.npy')).cuda()
gen = json.load(open(f'{ref}/ref.json'))['runs'][0]['tokens']
full = torch.cat([ids, torch.tensor([gen[:-1]], device='cuda')], 1)
p0 = int(pos.max()) + 1
fpos = torch.cat([pos, (torch.arange(len(gen) - 1, device='cuda') + p0).view(1, 1, -1).expand(3, 1, -1)], 2)
for dt, name in [(torch.bfloat16, 'bf16'), (torch.float32, 'fp32')]:
    m = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=dt, attn_implementation='sdpa').eval().cuda()
    with torch.inference_mode():
        e = m.model.get_input_embeddings()(full); e[full == IMAGE] = vis.to(dt)
        h = m.model.language_model(inputs_embeds=e, position_ids=fpos, attention_mask=torch.ones_like(full)).last_hidden_state
        lg = m.lm_head(h[0, ids.shape[1] - 1:]).float()
    np.save(f'{ref}/tf_{name}.npy', lg.cpu().numpy().astype(np.float16))
    print(name, tuple(lg.shape), 'greedy agreement with generated tokens', float((lg.argmax(-1).cpu() == torch.tensor(gen)).float().mean()))
    del m; torch.cuda.empty_cache()
