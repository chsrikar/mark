#!/usr/bin/env python3
"""
train_tokenizer.py
Byte-level BPE tokenizer for the personal LLM (Srikar).

Spec
  - Byte-level BPE (256 byte base tokens + learned merges + specials)
  - Total vocab size INCLUDES special tokens (default 32768 = 2^15)
  - GPT-4 style pre-tokenization regex, digits split individually
  - Whitespace runs preserved (matters for Python indentation)
  - No normalization

Usage examples
  # quick start with a small ML-only dataset (csv/parquet/json/jsonl), plus local code
  python train_tokenizer.py --input ML-Arxiv-Papers.csv --code-dir ./code_samples

  # Kaggle arXiv snapshot (jsonl), auto-filtered to ML categories
  python train_tokenizer.py --input arxiv-metadata-oai-snapshot.json

  # compare 16K vs 32K
  python train_tokenizer.py --input data.parquet --vocab-sizes 16384 32768

  # free code corpus: your installed Python packages (torch, numpy, ...)
  python train_tokenizer.py --input data.csv --code-dir "$(python -c 'import site;print(site.getsitepackages()[0])')"
"""
import argparse
import glob
import json
import os
import random
import sys

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

ML_CATS = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "stat.ML"}

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

# GPT-4 (cl100k) style split, with single-digit numbers and whitespace runs kept.
PRE_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def clean(s):
    return " ".join(str(s).split())


def record_to_text(rec):
    """Turn a dict-like record into one training document."""
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


def load_text_docs(path, limit=None):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No such file or directory: '{path}'. Checked current directory and 'data/' folder."
        )

    docs = []
    ext = os.path.splitext(path)[1].lower()

    if ext in (".json", ".jsonl"):
        print(f"Reading from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
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
                if (i + 1) % 250000 == 0:
                    print(f"  scanned {i+1:,} lines, found {len(docs):,} ML docs...")
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
        sys.exit(f"Unsupported input type: {ext} (use csv, parquet, json, jsonl)")
    return docs


def load_code_docs(code_dir, max_files=20000, max_bytes=200_000):
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


def utf8_bytes(docs):
    return sum(len(d.encode("utf-8")) for d in docs)


# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------
def build_and_train(train_docs, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRE_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # includes special tokens
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    def batches(bs=1000):
        for i in range(0, len(train_docs), bs):
            yield train_docs[i : i + bs]

    tok.train_from_iterator(batches(), trainer=trainer, length=len(train_docs))
    return tok


def bytes_per_token(tok, docs):
    if not docs:
        return float("nan")
    total_b, total_t = 0, 0
    for d in docs:
        total_b += len(d.encode("utf-8"))
        total_t += len(tok.encode(d).ids)
    return total_b / max(total_t, 1)


def roundtrip_ok(tok, docs, n=200):
    bad = 0
    for d in docs[:n]:
        if tok.decode(tok.encode(d).ids) != d:
            bad += 1
    return bad


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    default_input = None
    for candidate in [
        os.path.join("data", "arxiv-metadata-oai-snapshot.json"),
        "arxiv-metadata-oai-snapshot.json",
    ]:
        if os.path.exists(candidate):
            default_input = candidate
            break

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=default_input,
        required=(default_input is None),
        help=f"csv / parquet / json / jsonl (default: {default_input})",
    )
    ap.add_argument("--code-dir", default=None, help="folder of .py files for the code mix")
    ap.add_argument("--code-fraction", type=float, default=0.2, help="share of training BYTES from code")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768])
    ap.add_argument("--out", default="tokenizer_out")
    ap.add_argument("--limit", type=int, default=None, help="max text docs to load")
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("Loading text...")
    text_docs = load_text_docs(args.input, args.limit)
    random.shuffle(text_docs)
    print(f"  {len(text_docs):,} text docs, {utf8_bytes(text_docs)/1e6:.1f} MB")
    if len(text_docs) < 100:
        sys.exit("Too few documents. Check the input file / columns.")

    n_hold = max(20, int(len(text_docs) * args.holdout))
    text_hold, text_train = text_docs[:n_hold], text_docs[n_hold:]

    code_hold, code_train = [], []
    if args.code_dir:
        print("Loading code...")
        code_docs = load_code_docs(args.code_dir)
        random.shuffle(code_docs)
        n_ch = max(10, int(len(code_docs) * args.holdout)) if code_docs else 0
        code_hold = code_docs[:n_ch]
        pool = code_docs[n_ch:]
        # take code until it reaches the requested share of bytes
        f = args.code_fraction
        target = utf8_bytes(text_train) * f / max(1e-9, (1 - f))
        got = 0
        for d in pool:
            if got >= target:
                break
            code_train.append(d)
            got += len(d.encode("utf-8"))
        print(f"  {len(code_train):,} code files, {got/1e6:.1f} MB (target {target/1e6:.1f} MB)")

    train_docs = text_train + code_train
    random.shuffle(train_docs)
    print(f"Training on {len(train_docs):,} docs, {utf8_bytes(train_docs)/1e6:.1f} MB total\n")

    results = []
    for vs in args.vocab_sizes:
        print(f"=== Training byte-level BPE, vocab={vs} ===")
        tok = build_and_train(train_docs, vs)
        path = os.path.join(args.out, f"tokenizer_{vs}.json")
        tok.save(path)

        bpt_text = bytes_per_token(tok, text_hold)
        bpt_code = bytes_per_token(tok, code_hold)
        bad = roundtrip_ok(tok, text_hold + code_hold)
        results.append((vs, tok.get_vocab_size(), bpt_text, bpt_code, bad, path))

        print(f"  saved: {path}")
        print(f"  actual vocab size: {tok.get_vocab_size()}")
        print(f"  special token ids: " + ", ".join(f"{s}={tok.token_to_id(s)}" for s in SPECIAL_TOKENS))
        sample = "def forward(self, x):\n    return x @ self.W  # shape (B, T, d)\nLoss: \\nabla_\\theta L = 0.0012"
        ids = tok.encode(sample).ids
        print(f"  sample: {len(sample)} chars -> {len(ids)} tokens")
        print()

    print("=" * 62)
    print(f"{'vocab':>7} {'actual':>7} {'B/tok text':>11} {'B/tok code':>11} {'roundtrip fails':>16}")
    for vs, actual, bt, bc, bad, _ in results:
        print(f"{vs:>7} {actual:>7} {bt:>11.3f} {bc:>11.3f} {bad:>16}")
    print("\nHigher bytes/token = better compression. Roundtrip fails must be 0.")


if __name__ == "__main__":
    main()
