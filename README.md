# Mark: Personal LLM from Scratch

**Mark** is a personal decoder-only transformer language model built and trained from scratch. It features custom byte-level BPE tokenization, modern architectural enhancements (RoPE, RMSNorm, SwiGLU, tied embeddings), an efficient memory-mapped data preparation pipeline, and a local interactive Web Studio.

---

## Architecture & Codebase Map

The codebase is organized into five tightly integrated pipeline stages:

```
                  ┌──────────────────────┐
                  │  train_tokenizer.py  │  Train Byte-Level BPE (16K / 32K)
                  └──────────┬───────────┘
                             │  tokenizer_out/tokenizer_*.json
                             ▼
                  ┌──────────────────────┐
                  │   prepare_data.py    │  Tokenize arXiv + code into uint16 arrays
                  └──────────┬───────────┘
                             │  data/train.bin, data/val.bin, data/meta.json
                             ▼
┌──────────────┐  ┌──────────────────────┐
│   model.py   │─►│       train.py       │  Pre-training loop (AdamW, cosine LR, amp)
└──────────────┘  └──────────┬───────────┘
                             │  out/best.pt, out/ckpt.pt, out/log.csv
                             ▼
                  ┌──────────────────────┐
                  │     generate.py      │  Autoregressive generation & sampling
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   tokenizer_app.py   │  Interactive Web Studio (Tokenizer Lab &
                  └──────────────────────┘  Model Generation Playground)
```

---

## File Overview

| File | Role | Connected To |
|------|------|--------------|
| [train_tokenizer.py](file:///train_tokenizer.py) | Trains byte-level BPE tokenizer on text + code | Outputs to `tokenizer_out/` for `prepare_data.py` & `tokenizer_app.py` |
| [prepare_data.py](file:///prepare_data.py) | Streams raw text + code into token IDs | Uses tokenizer from `tokenizer_out/`, creates `data/` for `train.py` |
| [model.py](file:///model.py) | Decoder-only Transformer (RoPE, RMSNorm, SwiGLU) | Imported by `train.py`, `generate.py`, `tokenizer_app.py` |
| [train.py](file:///train.py) | Pre-training loop with presets (`tiny`, `30m`) | Uses `model.py` and `data/`, produces `out/best.pt` |
| [generate.py](file:///generate.py) | Text generation with top-k, top-p, temperature | Loads `out/best.pt` and auto-matches tokenizer |
| [tokenizer_app.py](file:///tokenizer_app.py) | Web interface for token analysis & generation | Integrates `tokenizer_out/`, `model.py`, and `generate.py` |
| [main.py](file:///main.py) | Unified command-line interface | Unified entrypoint for all 5 scripts |
| [__init__.py](file:///__init__.py) | Package initialization | Allows importing `mark` as a Python module |

---

## Model Architecture Specs

- **Type**: Causal decoder-only Transformer (predict next token)
- **Positional Embeddings**: Rotary Position Embeddings (RoPE) computed dynamically
- **Normalization**: RMSNorm with learned scaling
- **Feed-Forward Network**: SwiGLU gated activation (~8/3 d_model hidden dim)
- **Parameter Sharing**: Weight tying between token embedding and LM head
- **Presets**:
  - `tiny` (~4.7M parameters): 2 layers, 4 heads, 128 embedding dim, 128 context length (laptop/CPU friendly)
  - `30m` (~30M parameters): 10 layers, 6 heads, 384 embedding dim, 512 context length (GPU training)

---

## Quickstart Guide

### 1. Unified CLI (`main.py`)

You can run any part of the pipeline using `main.py`:

```bash
# View all available commands
python main.py --help

# 1. Train a tokenizer
python main.py tokenize --input data/arxiv-metadata-oai-snapshot.json --vocab-sizes 32768

# 2. Prepare binary dataset
python main.py prepare --code-dir code_corpus

# 3. Train the model (sanity check on CPU)
python main.py train --preset tiny

# 4. Generate completions
python main.py generate --prompt "Attention is all you need"

# 5. Launch the Web Studio
python main.py app --port 8000
```

---

### 2. Standalone Script Usage

Each script can also be executed directly:

#### Tokenizer Training
```bash
python train_tokenizer.py --input arxiv-metadata-oai-snapshot.json --code-dir ./code_corpus
```

#### Dataset Preparation
```bash
python prepare_data.py --code-dir code_corpus
```

#### Training
```bash
# CPU / Laptop test:
python train.py --preset tiny

# Full GPU run (Kaggle / Colab):
python train.py --preset 30m --data-dir data --out-dir out

# Resume training from checkpoint:
python train.py --preset 30m --resume
```

#### Text Generation
```bash
python generate.py --ckpt out/best.pt --prompt "def forward(self, x):" --temperature 0.7
```

#### Interactive Web Studio
```bash
python tokenizer_app.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to inspect tokenization splits and test generation in the browser.

---

## Python API Usage

You can also use Mark directly in Python:

```python
from mark import GPT, GPTConfig, generate_text, load_model, resolve_tokenizer

# Initialize a fresh model from preset
model = GPT.from_preset("tiny")
print(f"Model parameters: {model.num_params():,}")

# Or load trained checkpoint and generate text
model, ck, path = load_model("out/best.pt")
tok, _ = resolve_tokenizer(vocab_size=model.cfg.vocab_size, ck=ck)
output = generate_text(model, tok, prompt="Attention is all you need", max_new_tokens=100)
print(output[0])
```