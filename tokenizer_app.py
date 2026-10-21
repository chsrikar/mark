#!/usr/bin/env python3
"""
tokenizer_app.py - local web studio for Mark: Tokenizer Lab & Model Generation.

Run (from the repo folder):
    python tokenizer_app.py                    # loads tokenizers from ./tokenizer_out and models from ./out
    python tokenizer_app.py --dir other_dir --port 8000

Then open http://127.0.0.1:8000

Binds to 127.0.0.1, so it is reachable from your own machine only.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer

MAX_CHARS = 200_000   # input is truncated beyond this
TOKENIZERS = {}       # name -> Tokenizer

# Cached model for generation
_CACHED_MODEL = None
_CACHED_CKPT_PATH = None
_CACHED_TOK = None


def load_all_tokenizers(directory):
    TOKENIZERS.clear()
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            TOKENIZERS[name] = Tokenizer.from_file(path)
        except Exception as exc:  # not a tokenizer file, skip it
            print(f"  skipped {path}: {exc}")


def tokenize(name, text):
    tok = TOKENIZERS[name]
    ids = tok.encode(text).ids
    pieces = [tok.decode([i], skip_special_tokens=False) for i in ids]
    n_bytes = len(text.encode("utf-8"))
    return {
        "name": name,
        "vocab": tok.get_vocab_size(),
        "tokens": len(ids),
        "chars": len(text),
        "bytes": n_bytes,
        "bytes_per_token": round(n_bytes / len(ids), 3) if ids else 0,
        "ids": ids,
        "pieces": pieces,
    }


def get_model_info():
    """Inspect the output directory for available model checkpoints."""
    candidates = ["out/best.pt", "out/ckpt.pt"]
    for path in sorted(glob.glob(os.path.join("out", "*.pt"))):
        if path not in candidates:
            candidates.append(path)

    found_path = None
    for p in candidates:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        return {
            "available": False,
            "message": "No checkpoint found in out/. Train a model first with: python train.py --preset tiny",
            "tokenizers": list(TOKENIZERS.keys()),
        }

    try:
        import torch
        ck = torch.load(found_path, map_location="cpu")
        cfg = ck.get("config", {})
        return {
            "available": True,
            "ckpt_path": found_path,
            "step": ck.get("step", 0),
            "val_loss": round(ck.get("val_loss", 0.0), 4) if "val_loss" in ck else None,
            "vocab_size": cfg.get("vocab_size", 32768),
            "n_layer": cfg.get("n_layer"),
            "n_embd": cfg.get("n_embd"),
            "n_head": cfg.get("n_head"),
            "block_size": cfg.get("block_size"),
            "tokenizers": list(TOKENIZERS.keys()),
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Checkpoint found at {found_path}, but could not load: {e}",
            "tokenizers": list(TOKENIZERS.keys()),
        }


def run_generation(prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95, tokenizer_name=None):
    """Run model generation using cached or freshly loaded checkpoint."""
    global _CACHED_MODEL, _CACHED_CKPT_PATH, _CACHED_TOK

    from generate import generate_text, load_model, resolve_tokenizer

    info = get_model_info()
    if not info.get("available"):
        raise RuntimeError(info.get("message", "Model not available"))

    ckpt_path = info["ckpt_path"]

    # Load or reuse model
    if _CACHED_MODEL is None or _CACHED_CKPT_PATH != ckpt_path:
        model, ck, resolved_path = load_model(ckpt_path)
        _CACHED_MODEL = model
        _CACHED_CKPT_PATH = resolved_path

        # Resolve tokenizer
        tok_file = None
        if tokenizer_name and tokenizer_name in TOKENIZERS:
            tok_file = os.path.join("tokenizer_out", f"{tokenizer_name}.json")
        tok, _ = resolve_tokenizer(tokenizer_path=tok_file, vocab_size=model.cfg.vocab_size, ck=ck)
        _CACHED_TOK = tok
    else:
        model = _CACHED_MODEL
        tok = _CACHED_TOK

    t0 = time.time()
    results = generate_text(
        model=model,
        tok=tok,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_samples=1,
    )
    elapsed = time.time() - t0

    generated = results[0]
    tok_ids = tok.encode(generated).ids
    tok_pieces = [tok.decode([i], skip_special_tokens=False) for i in tok_ids]

    return {
        "text": generated,
        "tokens": len(tok_ids),
        "elapsed": round(elapsed, 3),
        "tokens_per_sec": round(len(tok_ids) / max(elapsed, 1e-4), 1),
        "pieces": tok_pieces,
        "ids": tok_ids,
        "ckpt": _CACHED_CKPT_PATH,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/tokenizers":
            self._json(200, [{"name": n, "vocab": t.get_vocab_size()} for n, t in TOKENIZERS.items()])
        elif self.path == "/api/model_status":
            self._json(200, get_model_info())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/tokenize":
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                text = str(payload.get("text", ""))
                names = [n for n in payload.get("names", []) if n in TOKENIZERS]
            except (ValueError, TypeError):
                return self._json(400, {"error": "bad request"})
            truncated = len(text) > MAX_CHARS
            text = text[:MAX_CHARS]
            results = [tokenize(n, text) for n in names] if text else []
            self._json(200, {"results": results, "truncated": truncated})

        elif self.path == "/api/generate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(payload.get("prompt", ""))
                max_tokens = int(payload.get("max_new_tokens", 100))
                temp = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", 50))
                top_p = float(payload.get("top_p", 0.95))
                tok_name = payload.get("tokenizer")
                res = run_generation(prompt, max_tokens, temp, top_k, top_p, tok_name)
                self._json(200, res)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
        else:
            self._json(404, {"error": "not found"})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mark LLM Studio</title>
<style>
  :root {
    --bg:#f7f6f2; --fg:#1c1c1a; --muted:#6c6c66; --card:#ffffff; --line:#e3e1d9;
    --accent:#2d6a57; --accent-soft:#2d6a5718; --chip-l:62%; --chip-a:.30;
    --radius:12px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#111312; --fg:#ecebe5; --muted:#9a9a92; --card:#1a1c1b; --line:#2b2e2c;
      --accent:#7fc9b0; --accent-soft:#7fc9b01c; --chip-l:55%; --chip-a:.38;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:28px 16px 60px; }
  .header { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px; border-bottom:1px solid var(--line); padding-bottom:12px; }
  h1 { font-size:24px; margin:0 0 4px; letter-spacing:-.02em; display:flex; align-items:center; gap:8px; }
  .sub { color:var(--muted); margin:0; font-size:14px; }
  .nav-tabs { display:flex; gap:8px; margin-bottom:20px; }
  .tab-btn { background:transparent; border:1px solid var(--line); color:var(--muted); padding:8px 16px; border-radius:999px; font-weight:500; cursor:pointer; font-size:14px; transition:.15s; }
  .tab-btn:hover { background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
  .tab-btn.active { background:var(--accent); color:#ffffff; border-color:var(--accent); }
  @media (prefers-color-scheme: dark) { .tab-btn.active { color:#111312; font-weight:600; } }

  .tab-pane { display:none; }
  .tab-pane.active { display:block; }

  textarea { width:100%; min-height:140px; resize:vertical; padding:12px 14px;
             background:var(--card); color:var(--fg); border:1px solid var(--line);
             border-radius:10px; font:14px/1.55 ui-monospace,"Cascadia Code",Consolas,Menlo,monospace; }
  textarea:focus { outline:2px solid var(--accent); outline-offset:1px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin:12px 0 18px; }
  .bar label { display:inline-flex; gap:6px; align-items:center; cursor:pointer; }
  .muted { color:var(--muted); }
  .samples button { font:inherit; font-size:13px; padding:4px 10px; margin-right:6px; cursor:pointer;
                    background:transparent; color:var(--fg); border:1px solid var(--line); border-radius:999px; }
  .samples button:hover { background:var(--accent-soft); border-color:var(--accent); }
  .grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:16px; }
  .card h2 { font-size:15px; margin:0 0 10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .badge { font-size:11px; padding:2px 8px; border-radius:999px; background:var(--accent-soft);
           color:var(--accent); border:1px solid var(--accent); }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:12px; }
  .stat { background:var(--bg); border-radius:8px; padding:8px 10px; }
  .stat b { display:block; font-size:18px; font-variant-numeric:tabular-nums; }
  .stat span { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .tokens { line-height:2; word-break:break-all; max-height:380px; overflow:auto; }
  .t { display:inline-block; padding:0 3px; margin:1px 1px; border-radius:4px; white-space:pre;
       font:13px/1.6 ui-monospace,"Cascadia Code",Consolas,Menlo,monospace;
       background:hsl(var(--h) 70% var(--chip-l) / var(--chip-a)); }
  .t0{--h:10} .t1{--h:45} .t2{--h:140} .t3{--h:200} .t4{--h:265} .t5{--h:320}
  details { margin-top:10px; }
  summary { cursor:pointer; color:var(--muted); font-size:13px; }
  .ids { margin-top:6px; font:12px/1.6 ui-monospace,Consolas,monospace; color:var(--muted);
         max-height:140px; overflow:auto; word-break:break-all; }
  .note { color:var(--muted); font-size:13px; margin-top:8px; }
  .empty { text-align:center; padding:40px 0; color:var(--muted); }

  /* Model Gen tab */
  .control-row { display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; align-items:center; }
  .control-item { display:flex; flex-direction:column; gap:4px; min-width:140px; }
  .control-item label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
  .control-item input[type="range"] { accent-color:var(--accent); }
  .primary-btn { background:var(--accent); color:#fff; border:none; border-radius:8px; padding:10px 22px; font-weight:600; font-size:14px; cursor:pointer; transition:.15s; }
  .primary-btn:hover { opacity:0.9; }
  .primary-btn:disabled { opacity:0.5; cursor:not-allowed; }
  @media (prefers-color-scheme: dark) { .primary-btn { color:#111312; } }
  .status-box { padding:12px 16px; border-radius:10px; border:1px solid var(--line); margin-bottom:16px; font-size:13px; display:flex; justify-content:space-between; align-items:center; }
  .status-tag { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e; margin-right:6px; }
  .status-tag.off { background:#eab308; }
  .out-box { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:16px; min-height:120px; white-space:pre-wrap; font:14px/1.6 ui-monospace,Consolas,monospace; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h1>Mark Studio</h1>
      <p class="sub">Byte-level BPE Tokenizer Lab &amp; Transformer Generation</p>
    </div>
  </div>

  <div class="nav-tabs">
    <button class="tab-btn active" onclick="setTab('tok')">&#128292; Tokenizer Lab</button>
    <button class="tab-btn" onclick="setTab('gen')">&#10024; Model Generation</button>
  </div>

  <!-- Tab 1: Tokenizer Lab -->
  <div id="tab-tok" class="tab-pane active">
    <textarea id="text" spellcheck="false" placeholder="Paste anything: an abstract, Python code, LaTeX, a chat message..."></textarea>
    <div class="bar">
      <span class="muted">Compare:</span>
      <span id="picks"></span>
    </div>
    <div class="bar samples">
      <span class="muted">Try:</span>
      <button data-s="ml">ML abstract</button>
      <button data-s="code">Python code</button>
      <button data-s="latex">LaTeX</button>
      <button data-s="chat">Chat message</button>
      <button data-s="special">Special tokens</button>
    </div>
    <div id="out" class="grid"><div class="empty">Tokens will appear here.</div></div>
    <p class="note">Spaces show as &middot;, newlines as &crarr;, tabs as &rarr;. Hover a token to see its ID.</p>
  </div>

  <!-- Tab 2: Model Generation -->
  <div id="tab-gen" class="tab-pane">
    <div id="model-status" class="status-box">Checking model checkpoint status...</div>
    <textarea id="prompt" spellcheck="false" placeholder="Enter prompt to continue (e.g. 'Attention is all you need' or 'def forward(self, x):')..."></textarea>

    <div class="bar samples">
      <span class="muted">Prompts:</span>
      <button onclick="setPrompt('Attention is all you need')">Attention paper</button>
      <button onclick="setPrompt('def train(model, dataloader):')">Train function</button>
      <button onclick="setPrompt('We propose a causal transformer architecture')">Abstract intro</button>
      <button onclick="setPrompt('class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model):')">PyTorch Module</button>
    </div>

    <div class="control-row">
      <div class="control-item">
        <label>Temperature (<span id="temp-val">0.8</span>)</label>
        <input type="range" id="temp" min="0" max="1.5" step="0.05" value="0.8" oninput="$('#temp-val').textContent=this.value">
      </div>
      <div class="control-item">
        <label>Max Tokens (<span id="toks-val">100</span>)</label>
        <input type="range" id="max-toks" min="20" max="400" step="10" value="100" oninput="$('#toks-val').textContent=this.value">
      </div>
      <div class="control-item">
        <label>Top-p (<span id="topp-val">0.95</span>)</label>
        <input type="range" id="top-p" min="0.5" max="1.0" step="0.05" value="0.95" oninput="$('#topp-val').textContent=this.value">
      </div>
      <button id="gen-btn" class="primary-btn" onclick="generate()">Generate with Mark</button>
    </div>

    <div id="gen-stats" class="stats" style="display:none; margin-top:14px;">
      <div class="stat"><b id="g-toks">0</b><span>tokens</span></div>
      <div class="stat"><b id="g-time">0s</b><span>time</span></div>
      <div class="stat"><b id="g-speed">0</b><span>tok/s</span></div>
      <div class="stat"><b id="g-ckpt">-</b><span>checkpoint</span></div>
    </div>

    <div id="gen-out" class="out-box" style="margin-top:12px;">Generated output will appear here.</div>
    <div id="gen-chips" class="tokens" style="margin-top:10px;"></div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const vis = s => s.replace(/ /g,'·').replace(/\n/g,'↵').replace(/\t/g,'→');
const MAX_RENDER = 5000;
let timer = null;

const SAMPLES = {
  ml: "We propose a transformer architecture trained with stochastic gradient descent and AdamW. " +
      "Backpropagation through attention layers converges faster with layer normalization and dropout regularization.",
  code: "class Block(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n" +
        "        self.attn = nn.MultiheadAttention(d_model, n_heads)\n\n    def forward(self, x):\n        return x + self.attn(x, x, x)[0]\n",
  latex: "The loss is $\\mathcal{L}(\\theta) = -\\sum_{i=1}^{N} \\log p_\\theta(x_i \\mid x_{<i})$ and $\\nabla_\\theta \\mathcal{L} = 0$.",
  chat: "Hey Mark, what's on my schedule today? Remind me to call Ashirwad at 5 pm.",
  special: "<|system|>You are a helpful assistant.<|user|>Hello!<|assistant|>Hi there.<|endoftext|>"
};

function setTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', (name === 'tok' ? i===0 : i===1)));
  $('#tab-tok').classList.toggle('active', name === 'tok');
  $('#tab-gen').classList.toggle('active', name === 'gen');
  if (name === 'gen') checkModelStatus();
}

function setPrompt(txt) {
  $('#prompt').value = txt;
}

async function init() {
  const r = await fetch('/api/tokenizers');
  const list = await r.json();
  const box = $('#picks');
  list.forEach(x => {
    const l = document.createElement('label');
    l.innerHTML = '<input type="checkbox" value="' + esc(x.name) + '" checked> ' + esc(x.name) +
                  ' <span class="muted">(' + x.vocab.toLocaleString() + ')</span>';
    box.appendChild(l);
  });
  box.addEventListener('change', run);
  $('#text').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(run, 150); });
  document.querySelectorAll('.samples button[data-s]').forEach(b => b.addEventListener('click', () => {
    $('#text').value = SAMPLES[b.dataset.s];
    run();
  }));
}

async function run() {
  const text = $('#text').value;
  const names = [...document.querySelectorAll('#picks input:checked')].map(i => i.value);
  if (!text || !names.length) {
    $('#out').innerHTML = '<div class="empty">Tokens will appear here.</div>';
    return;
  }
  const r = await fetch('/api/tokenize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, names })
  });
  const data = await r.json();
  render(data.results, data.truncated);
}

function render(results, truncated) {
  const best = Math.min(...results.map(x => x.tokens));
  $('#out').innerHTML = results.map(x => {
    const shown = x.pieces.slice(0, MAX_RENDER);
    const chips = shown.map((p, i) =>
      '<span class="t t' + (i % 6) + '" title="id ' + x.ids[i] + '">' + esc(vis(p)) + '</span>').join('');
    const more = x.pieces.length > MAX_RENDER
      ? '<div class="note">Showing the first ' + MAX_RENDER.toLocaleString() + ' tokens.</div>' : '';
    const win = (results.length > 1 && x.tokens === best) ? '<span class="badge">fewest tokens</span>' : '';
    return '<div class="card"><h2>' + esc(x.name) + ' ' + win + '</h2>' +
      '<div class="stats">' +
      '<div class="stat"><b>' + x.tokens.toLocaleString() + '</b><span>tokens</span></div>' +
      '<div class="stat"><b>' + x.chars.toLocaleString() + '</b><span>characters</span></div>' +
      '<div class="stat"><b>' + x.bytes.toLocaleString() + '</b><span>bytes</span></div>' +
      '<div class="stat"><b>' + x.bytes_per_token + '</b><span>bytes / token</span></div></div>' +
      '<div class="tokens">' + chips + '</div>' + more +
      '<details><summary>Token IDs</summary><div class="ids">' + x.ids.slice(0, MAX_RENDER).join(' ') + '</div></details>' +
      (truncated ? '<div class="note">Input was truncated to 200,000 characters.</div>' : '') +
      '</div>';
  }).join('');
}

async function checkModelStatus() {
  const box = $('#model-status');
  try {
    const r = await fetch('/api/model_status');
    const data = await r.json();
    if (data.available) {
      box.innerHTML = '<div><span class="status-tag"></span><b>Model Ready:</b> ' + esc(data.ckpt_path) +
        ' (' + (data.n_layer || '?') + ' layers, ' + (data.n_embd || '?') + ' dims, ' +
        (data.vocab_size || '?').toLocaleString() + ' vocab) &bull; val loss: ' + (data.val_loss || 'n/a') + '</div>';
      $('#gen-btn').disabled = false;
    } else {
      box.innerHTML = '<div><span class="status-tag off"></span><b>Checkpoint:</b> ' + esc(data.message) + '</div>';
      $('#gen-btn').disabled = true;
    }
  } catch (err) {
    box.innerHTML = '<div><span class="status-tag off"></span>' + esc(err.message) + '</div>';
    $('#gen-btn').disabled = true;
  }
}

async function generate() {
  const btn = $('#gen-btn');
  const out = $('#gen-out');
  const chipsBox = $('#gen-chips');
  const prompt = $('#prompt').value;
  const temp = parseFloat($('#temp').value);
  const maxToks = parseInt($('#max-toks').value);
  const topP = parseFloat($('#top-p').value);

  btn.disabled = true;
  btn.textContent = 'Generating...';
  out.textContent = 'Generating completion...';
  chipsBox.innerHTML = '';
  $('#gen-stats').style.display = 'none';

  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, temperature: temp, max_new_tokens: maxToks, top_p: topP })
    });
    const res = await r.json();
    if (res.error) throw new Error(res.error);

    out.textContent = res.text;
    $('#g-toks').textContent = res.tokens;
    $('#g-time').textContent = res.elapsed + 's';
    $('#g-speed').textContent = res.tokens_per_sec;
    $('#g-ckpt').textContent = res.ckpt.split(/[\\/]/).pop();
    $('#gen-stats').style.display = 'grid';

    if (res.pieces) {
      chipsBox.innerHTML = res.pieces.map((p, i) =>
        '<span class="t t' + (i % 6) + '" title="id ' + res.ids[i] + '">' + esc(vis(p)) + '</span>').join('');
    }
  } catch (err) {
    out.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate with Mark';
  }
}

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tokenizer_out", help="folder with tokenizer_*.json files")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    load_all_tokenizers(args.dir)
    if not TOKENIZERS:
        sys.exit(f"No tokenizer .json files found in '{args.dir}'. Run train_tokenizer.py first, or pass --dir.")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Loaded tokenizers: {', '.join(TOKENIZERS)}")
    info = get_model_info()
    if info.get("available"):
        print(f"Loaded model     : {info['ckpt_path']} (step {info.get('step')}, val loss {info.get('val_loss')})")
    else:
        print(f"Model status     : {info.get('message')}")
    print(f"Open {url}   (Ctrl+C to stop)")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


app = main

if __name__ == "__main__":
    main()
