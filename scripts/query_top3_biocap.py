import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5TokenizerFast




def patch_tokenizers():
    for cls in (T5Tokenizer, T5TokenizerFast):
        if cls is not None and not hasattr(cls, 'batch_encode_plus'):
            def _batch_encode_plus(self, *args, **kwargs):
                return self(*args, **kwargs)
            setattr(cls, 'batch_encode_plus', _batch_encode_plus)


def extract(out):
    if torch.is_tensor(out):
        return out
    for attr in ('text_embeds', 'pooler_output', 'last_hidden_state'):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            if attr == 'last_hidden_state' and v.ndim >= 3:
                return v[:, 0, :]
            return v
    if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
        v = out[0]
        return v[:, 0, :] if v.ndim >= 3 else v
    raise TypeError(type(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--model', default='biocap')
    p.add_argument('--caption-model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--topk', type=int, default=3)
    args = p.parse_args()

    print('step=patch_tokenizers', flush=True)
    patch_tokenizers()

    print(f'step=load_cache path={args.cache}', flush=True)
    with Path(args.cache).open('r', encoding='utf-8') as f:
        caption_cache = json.load(f)
    print(f'cache_size={len(caption_cache)}', flush=True)

    split_name = 'validation' if args.split == 'val' else 'test'
    print(f'step=load_dataset split={split_name}', flush=True)
    dataset = load_dataset('evendrow/INQUIRE-Rerank', split=split_name)
    print(f'dataset_size={len(dataset)}', flush=True)

    queries = np.asarray(dataset['query'])
    idxs = np.argwhere(queries == args.query).reshape(-1).tolist()
    print(f'query_candidates={len(idxs)}', flush=True)
    if not idxs:
        raise SystemExit(f'query not found: {args.query}')

    image_ids = [dataset['inat24_image_id'][i] for i in idxs]
    file_names = [dataset['inat24_file_name'][i] for i in idxs]
    relevant = [int(dataset['relevant'][i]) for i in idxs]
    cap_keys = [f"{args.caption_model}::inat24:{iid}" for iid in image_ids]
    captions = [caption_cache.get(k, 'unlabeled image') for k in cap_keys]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('step=import_load_clip', flush=True)
    from src.inquire.utils import load_clip
    print(f'step=load_clip model={args.model} device={device}', flush=True)
    model, _, tokenizer = load_clip(args.model, use_jit=False, device=device)
    model.eval()
    print('step=encode_text', flush=True)

    def to_tokens(texts):
        tokens = tokenizer(texts)
        if hasattr(tokens, 'to'):
            return tokens.to(device)
        if isinstance(tokens, dict):
            return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in tokens.items()}
        raise TypeError(type(tokens))

    with torch.no_grad():
        q_tok = to_tokens([args.query])
        q_out = model.encode_text(**q_tok) if isinstance(q_tok, dict) else model.encode_text(q_tok)
        q_emb = extract(q_out).float()
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        c_tok = to_tokens(captions)
        c_out = model.encode_text(**c_tok) if isinstance(c_tok, dict) else model.encode_text(c_tok)
        c_emb = extract(c_out).float()
        c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    print('step=rank', flush=True)
    scores = (c_emb @ q_emb.T).squeeze(-1).detach().cpu().numpy()
    order = np.argsort(-scores)

    print(json.dumps({'query': args.query, 'model': args.model, 'n_candidates': len(idxs), 'device': device}, ensure_ascii=True), flush=True)
    for rank, j in enumerate(order[:args.topk], start=1):
        print(json.dumps({
            'rank': rank,
            'score': float(scores[j]),
            'inat24_image_id': int(image_ids[j]) if isinstance(image_ids[j], (int, np.integer)) else image_ids[j],
            'inat24_file_name': file_names[j],
            'relevant': int(relevant[j]),
            'caption': captions[j],
        }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
