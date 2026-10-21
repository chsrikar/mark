#!/usr/bin/env python3
"""
prepare_data.py - turn raw text + code into train.bin / val.bin of token IDs.

Pipeline
  load ML abstracts (+ optional code files) -> clean -> shuffle documents
  -> tokenize with your byte-level BPE -> append <|endoftext|> after every document
  -> write flat binary arrays (uint16 for vocab <= 65536) -> verify

Usage (from the repo folder)
  python prepare_data.py --code-dir code_corpus
  python prepare_data.py --code-dir code_corpus --limit 400000        # smaller / faster
  python prepare_data.py --tokenizer tokenizer_out/tokenizer_16384.json

Outputs (default folder: data/)
  train.bin   about 99% of documents, flat array of token IDs
  val.bin     about 1% of documents, never used for training
  meta.json   vocab size, dtype, token counts, tokenizer path

Load later with:  np.memmap("data/train.bin", dtype=np.uint16, mode="r")
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
from tokenizers import Tokenizer

try:
    from train_tokenizer import ML_CATS, SPECIAL_TOKENS, resolve_path
    SPECIALS = list(SPECIAL_TOKENS)
except ImportError:
    ML_CATS = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "stat.ML"}
    SPECIALS = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>"]

    def resolve_path(path):
        if not path:
            return path
        if os.path.exists(path):
            return path
        candidates = [
            os.path.join("data", path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return path


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def clean(s):
    return " ".join(str(s).split())


def record_to_text(rec):
    title = rec.get("title") or ""
    abstract = rec.get("abstract") or rec.get("summary") or ""
    if title or abstract:
        return (clean(title) + "\n\n" + clean(abstract)).strip()
    if rec.get("text"):
        return str(rec["text"]).strip()
    parts = [str(v) for v in rec.values() if isinstance(v, str) and len(v) > 20]
    return "\n\n".join(parts).strip()


def has_ml_category(cats):
    if cats is None:
        return True
    if isinstance(cats, (list, tuple)):
        cats = " ".join(cats)
    return any(c in ML_CATS for c in str(cats).split())


def load_text_docs(path, limit=None):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No such file or directory: '{path}'. Checked current directory and 'data/' folder."
        )
    docs = []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not has_ml_category(rec.get("categories")):
                    continue
                t = record_to_text(rec)
                if len(t) > 50:
                    docs.append(t)
                if limit and len(docs) >= limit:
                    break
                if n % 500_000 == 0:
                    print(f"  scanned {n:,} lines, kept {len(docs):,} ML docs...")
    elif ext in (".csv", ".parquet"):
        import pandas as pd

        df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
        if "categories" in df.columns:
            df = df[df["categories"].map(has_ml_category)]
        for rec in df.to_dict("records"):
            t = record_to_text(rec)
            if len(t) > 50:
                docs.append(t)
            if limit and len(docs) >= limit:
                break
    else:
        sys.exit(f"Unsupported input type: {ext} (use json, jsonl, csv, parquet)")
    return docs


def load_code_docs(code_dir, max_files=50_000, max_bytes=200_000):
    files = glob.glob(os.path.join(code_dir, "**", "*.py"), recursive=True)
    random.shuffle(files)
    docs = []
    for fp in files[:max_files]:
        try:
            if os.path.getsize(fp) > max_bytes:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                t = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if len(t) > 100:
            docs.append(t)
    return docs


def sanitize(text):
    """Break up literal special-token strings (e.g. '<|endoftext|>' inside a code file) so the
    tokenizer cannot turn them into real special tokens. Only we insert <|endoftext|>."""
    for s in SPECIALS:
        if s in text:
            text = text.replace(s, s[0] + " " + s[1:])
    return text


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def encode_split(tok, docs, path, eot_id, dtype, batch_size, label):
    """docs: list of (kind, text). Streams token IDs to `path`. Returns stats."""
    total = 0
    by_kind = {"text": 0, "code": 0}
    t0 = time.time()
    with open(path, "wb") as out:
        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]
            encs = tok.encode_batch([sanitize(t) for _, t in chunk], add_special_tokens=False)
            buf = []
            for (kind, _), enc in zip(chunk, encs):
                ids = enc.ids
                buf.extend(ids)
                buf.append(eot_id)
                by_kind[kind] += len(ids) + 1
            arr = np.asarray(buf, dtype=dtype)
            out.write(arr.tobytes())
            total += len(arr)
            done = min(i + batch_size, len(docs))
            if (i // batch_size) % 10 == 0 or done == len(docs):
                rate = total / max(time.time() - t0, 1e-9) / 1e6
                print(f"  [{label}] {done:,}/{len(docs):,} docs  {total/1e6:.1f}M tokens  ({rate:.2f}M tok/s)")
    return total, by_kind


def verify(path, dtype, vocab_size, eot_id, n_docs, tok, label):
    arr = np.memmap(path, dtype=dtype, mode="r")
    ok = True
    if len(arr) == 0:
        print(f"  [{label}] FAIL: file is empty")
        return False
    mx = int(arr.max())
    if mx >= vocab_size:
        print(f"  [{label}] FAIL: max id {mx} >= vocab {vocab_size}")
        ok = False
    n_eot = int((arr == eot_id).sum())
    if n_eot != n_docs:
        print(f"  [{label}] FAIL: {n_eot:,} end-of-text tokens but {n_docs:,} documents")
        ok = False
    sample = tok.decode(arr[:120].tolist(), skip_special_tokens=False)
    print(f"  [{label}] {'OK' if ok else 'PROBLEMS'}: {len(arr):,} tokens, max id {mx}, {n_eot:,} docs")
    print(f"  [{label}] first 120 tokens decode to: {sample[:240]!r}")
    return ok


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(cli_args=None):
    default_code_dir = "code_corpus" if os.path.isdir("code_corpus") else None

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join("tokenizer_out", "tokenizer_32768.json"))
    ap.add_argument("--input", default=os.path.join("data", "arxiv-metadata-oai-snapshot.json"))
    ap.add_argument("--code-dir", default=default_code_dir, help="folder of .py files to include (defaults to code_corpus if present)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None, help="max text docs to load")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(cli_args)

    random.seed(args.seed)
    tok_path = resolve_path(args.tokenizer)
    if not os.path.exists(tok_path):
        # try inside tokenizer_out
        alt = os.path.join("tokenizer_out", os.path.basename(args.tokenizer))
        if os.path.exists(alt):
            tok_path = alt
        else:
            candidates = sorted(glob.glob(os.path.join("tokenizer_out", "*.json")))
            if candidates:
                tok_path = candidates[-1]
                print(f"Specified tokenizer not found; using '{tok_path}'")
            else:
                sys.exit(f"Tokenizer not found: {args.tokenizer}. Run train_tokenizer.py first.")

    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size()
    eot_id = tok.token_to_id("<|endoftext|>")
    if eot_id is None:
        sys.exit("Tokenizer has no <|endoftext|> token.")
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    print(f"Tokenizer: {tok_path}  (vocab {vocab_size:,}, storing as {np.dtype(dtype).name})")

    print("Loading text...")
    docs = [("text", t) for t in load_text_docs(args.input, args.limit)]
    n_text = len(docs)
    print(f"  {n_text:,} text docs")
    if args.code_dir and os.path.isdir(args.code_dir):
        print(f"Loading code from {args.code_dir}...")
        code = load_code_docs(args.code_dir)
        docs += [("code", t) for t in code]
        print(f"  {len(code):,} code files")
    elif args.code_dir:
        print(f"Warning: code-dir '{args.code_dir}' does not exist, skipping code.")

    if len(docs) < 10:
        sys.exit("Too few documents. Check --input / --code-dir.")

    random.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_fraction))
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    print(f"Split: {len(train_docs):,} train docs, {len(val_docs):,} val docs\n")

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")

    print("Tokenizing train...")
    n_train, kind_train = encode_split(tok, train_docs, train_path, eot_id, dtype, args.batch_size, "train")
    print("Tokenizing val...")
    n_val_tok, kind_val = encode_split(tok, val_docs, val_path, eot_id, dtype, args.batch_size, "val")

    print("\nVerifying...")
    ok = verify(train_path, dtype, vocab_size, eot_id, len(train_docs), tok, "train")
    ok = verify(val_path, dtype, vocab_size, eot_id, len(val_docs), tok, "val") and ok

    meta = {
        "tokenizer": tok_path,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "eot_id": eot_id,
        "train_tokens": n_train,
        "val_tokens": n_val_tok,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_tokens_text": kind_train["text"],
        "train_tokens_code": kind_train["code"],
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tot = n_train + n_val_tok
    code_share = (kind_train["code"] + kind_val["code"]) / max(tot, 1)
    print("\n" + "=" * 60)
    print(f"Total tokens : {tot/1e6:.1f}M  ({n_train/1e6:.1f}M train, {n_val_tok/1e6:.2f}M val)")
    print(f"Code share   : {code_share*100:.1f}% of tokens")
    print(f"Files        : {train_path}  ({os.path.getsize(train_path)/1e6:.0f} MB),  {val_path}")
    print(f"Rule of thumb: ~20 tokens per parameter -> about {n_train/20/1e6:.0f}M params in one pass,")
    print(f"               up to ~{n_train*4/20/1e6:.0f}M params if you train ~4 passes over the data.")
    print("Result       :", "ALL CHECKS PASSED" if ok else "CHECKS FAILED - do not train on these files")
    if not ok:
        sys.exit(1)
    return meta


prepare_data = main

if __name__ == "__main__":
    main()
