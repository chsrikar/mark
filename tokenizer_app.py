#!/usr/bin/env python3
"""
tokenizer_app.py - local web UI for your byte-level BPE tokenizers.

Run (from the repo folder):
    python tokenizer_app.py                    # loads every *.json in ./tokenizer_out
    python tokenizer_app.py --dir other_dir --port 8000

Then open http://127.0.0.1:8000

Only needs:  pip install tokenizers
Binds to 127.0.0.1, so it is reachable from your own machine only.
"""
import argparse
import glob
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer

MAX_CHARS = 200_000   # input is truncated beyond this
TOKENIZERS = {}       # name -> Tokenizer


def load_all(directory):
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
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/tokenize":
            return self._json(404, {"error": "not found"})
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


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tokenizer Lab</title>
<style>
  :root {
    --bg:#f7f6f2; --fg:#1c1c1a; --muted:#6c6c66; --card:#ffffff; --line:#e3e1d9;
    --accent:#2d6a57; --accent-soft:#2d6a5718; --chip-l:62%; --chip-a:.30;
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
  h1 { font-size:22px; margin:0 0 2px; letter-spacing:-.01em; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .mono { font-family:ui-monospace,"Cascadia Code",Consolas,Menlo,monospace; }
  textarea { width:100%; min-height:150px; resize:vertical; padding:12px 14px;
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
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
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
  @media (max-width:560px) { .stats { grid-template-columns:repeat(2,1fr); } }
</style>
</head>
<body>
<div class="wrap">
  <h1>Tokenizer Lab</h1>
  <p class="sub">Type or paste text and see exactly how your byte-level BPE tokenizers split it.</p>

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
  <p class="note">Spaces show as &middot;, newlines as &crarr;, tabs as &rarr;. Hover a token to see its ID.
     Higher bytes/token means better compression.</p>
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
  document.querySelectorAll('.samples button').forEach(b => b.addEventListener('click', () => {
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

    load_all(args.dir)
    if not TOKENIZERS:
        sys.exit(f"No tokenizer .json files found in '{args.dir}'. Run train_tokenizer.py first, or pass --dir.")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Loaded: {', '.join(TOKENIZERS)}")
    print(f"Open {url}   (Ctrl+C to stop)")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
