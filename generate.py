#!/usr/bin/env python3
"""
generate.py - write text with a trained checkpoint (works on a laptop CPU).

    python generate.py --ckpt out/best.pt --prompt "Attention is all you need"
    python generate.py --ckpt out/best.pt --prompt "def train(model):" --temperature 0.6 --max-new-tokens 200

Note: this is a BASE model. It continues text in the style of its training data
(ML abstracts and Python code). It does not follow instructions yet; that comes after
instruction tuning (SFT).
"""
import argparse
import glob
import json
import os
import sys

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(ckpt_path="out/best.pt", device=None):
    """Load model weights and config from checkpoint, auto-falling back to ckpt.pt if needed."""
    device = resolve_device(device or "auto")
    resolved_path = ckpt_path

    if not os.path.exists(resolved_path):
        # Fall back if out/best.pt requested but out/ckpt.pt exists
        if os.path.basename(resolved_path) == "best.pt":
            alt = os.path.join(os.path.dirname(resolved_path), "ckpt.pt")
            if os.path.exists(alt):
                resolved_path = alt

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. Run train.py first to train a model."
        )

    ck = torch.load(resolved_path, map_location=device)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck, resolved_path


def resolve_tokenizer(tokenizer_path=None, vocab_size=None, ck=None):
    """Resolve and load the matching tokenizer for the model."""
    candidates = []

    # 1. Explicit path if provided and exists
    if tokenizer_path and os.path.exists(tokenizer_path):
        candidates.append(tokenizer_path)

    # 2. Path saved in checkpoint dictionary
    if ck and ck.get("tokenizer") and os.path.exists(ck["tokenizer"]):
        candidates.append(ck["tokenizer"])

    # 3. Path recorded in data/meta.json
    if os.path.exists("data/meta.json"):
        try:
            with open("data/meta.json") as f:
                meta = json.load(f)
            if meta.get("tokenizer") and os.path.exists(meta["tokenizer"]):
                candidates.append(meta["tokenizer"])
        except Exception:
            pass

    # 4. Canonical paths based on vocab_size
    if vocab_size:
        v_cand = os.path.join("tokenizer_out", f"tokenizer_{vocab_size}.json")
        if os.path.exists(v_cand):
            candidates.append(v_cand)

    # 5. Look for any tokenizer file in tokenizer_out
    for f in sorted(glob.glob(os.path.join("tokenizer_out", "*.json"))):
        if f not in candidates:
            candidates.append(f)

    # Pick the first one whose vocab size matches (or the first available if vocab_size is None)
    for path in candidates:
        try:
            tok = Tokenizer.from_file(path)
            if vocab_size is None or tok.get_vocab_size() == vocab_size:
                return tok, path
        except Exception:
            continue

    # Fallback attempt with requested path
    target = tokenizer_path or (candidates[0] if candidates else "tokenizer_out/tokenizer_32768.json")
    if os.path.exists(target):
        tok = Tokenizer.from_file(target)
        if vocab_size is not None and tok.get_vocab_size() != vocab_size:
            sys.exit(
                f"Tokenizer vocab ({tok.get_vocab_size()}) does not match the model ({vocab_size}). "
                f"Use the tokenizer the data was prepared with."
            )
        return tok, target

    raise FileNotFoundError(
        f"No suitable tokenizer found in 'tokenizer_out/'. Run train_tokenizer.py first."
    )


def generate_text(
    model,
    tok,
    prompt="",
    max_new_tokens=150,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_samples=1,
    device=None,
):
    """Generate completions from prompt using model and tokenizer."""
    device = device or next(model.parameters()).device
    eot = tok.token_to_id("<|endoftext|>")

    # An empty prompt starts a fresh document (the model saw <|endoftext|> before every document)
    ids = tok.encode(prompt, add_special_tokens=False).ids if prompt else [eot]

    results = []
    for _ in range(num_samples):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens, temperature, top_k, top_p, eos_id=eot)
        text = tok.decode(y[0].tolist(), skip_special_tokens=True)
        results.append(text)
    return results


def main(cli_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/best.pt")
    ap.add_argument("--tokenizer", default=None, help="path to tokenizer json (auto-detected if omitted)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the most likely token")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(cli_args)

    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, ck, ckpt_path = load_model(args.ckpt, device=device)
    tok, tok_path = resolve_tokenizer(args.tokenizer, vocab_size=model.cfg.vocab_size, ck=ck)

    print(f"Loaded checkpoint: {ckpt_path} ({model.num_params()/1e6:.1f}M params, "
          f"step {ck.get('step', '?')}, val loss {ck.get('val_loss', '?')})")
    print(f"Using tokenizer : {tok_path} (vocab {tok.get_vocab_size():,})")
    print(f"Device          : {device}\n")

    results = generate_text(
        model=model,
        tok=tok,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples,
        device=device,
    )

    for n, text in enumerate(results, 1):
        print(f"--- sample {n} " + "-" * 50)
        print(text)
        print()

    return results


generate = main

if __name__ == "__main__":
    main()
