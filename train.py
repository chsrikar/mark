#!/usr/bin/env python3
"""
train.py - train the GPT in model.py on data/train.bin (made by prepare_data.py).

Quick laptop sanity check (a few minutes, CPU):
    python train.py --preset tiny
Real run on a GPU (Kaggle / Colab):
    python train.py --preset 30m --data-dir /kaggle/input/<your-dataset> --out-dir /kaggle/working/out
Resume after a disconnect:
    python train.py --preset 30m --data-dir ... --out-dir ... --resume
Stop politely before a session limit (saves a checkpoint first):
    python train.py ... --time-budget-hours 11

The first evaluation prints the starting loss. A healthy start is close to ln(vocab_size),
about 10.4 for a 32K vocabulary. If it is far from that, something is wrong.
"""
import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

from model import GPT, GPTConfig

PRESETS = {
    # tiny: just proves the pipeline runs and the loss goes down (CPU-friendly)
    "tiny": dict(n_layer=2, n_head=4, n_embd=128, block_size=128, batch_size=16, grad_accum=1,
                 max_steps=300, lr=1e-3, warmup=20, eval_interval=100, eval_iters=10),
    # 30m: the real model. 128 sequences x 512 tokens = 65,536 tokens per step, ~3.9 passes over 170M tokens
    "30m": dict(n_layer=10, n_head=6, n_embd=384, block_size=512, batch_size=16, grad_accum=8,
                max_steps=10000, lr=6e-4, warmup=300, eval_interval=500, eval_iters=50),
}


def parse_args(cli_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="30m")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    for name in ["max-steps", "batch-size", "grad-accum", "block-size", "warmup", "eval-interval", "eval-iters",
                 "n-layer", "n-head", "n-embd"]:
        ap.add_argument(f"--{name}", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--time-budget-hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile (faster on Linux GPUs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(cli_args)
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


class Data:
    def __init__(self, data_dir, dtype, block_size, device):
        self.paths = {s: os.path.join(data_dir, f"{s}.bin") for s in ("train", "val")}
        for p in self.paths.values():
            if not os.path.exists(p):
                sys.exit(f"Missing {p}. Run prepare_data.py first (or fix --data-dir).")
        self.dtype, self.T, self.device = dtype, block_size, device

    def batch(self, split, bs):
        data = np.memmap(self.paths[split], dtype=self.dtype, mode="r")   # re-opened each call on purpose
        ix = torch.randint(len(data) - self.T - 1, (bs,)).tolist()
        x = torch.stack([torch.from_numpy(data[i : i + self.T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.T].astype(np.int64)) for i in ix])
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y


def save_ckpt(path, model, opt, scaler, step, best_val, cfg, tokenizer=None):
    tmp = path + ".tmp"
    data = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "best_val": best_val, "config": cfg.__dict__}
    if tokenizer:
        data["tokenizer"] = tokenizer
    torch.save(data, tmp)
    os.replace(tmp, path)   # atomic: a crash mid-save never corrupts the last good checkpoint


def main(cli_args=None):
    args = parse_args(cli_args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    meta_path = os.path.join(args.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"Missing {meta_path}. Run prepare_data.py first to create the dataset tokens.")
    with open(meta_path) as f:
        meta = json.load(f)
    tok_path = meta.get("tokenizer", "")

    # ---- precision: bf16 on Ampere+ GPUs, fp16 (+loss scaling) on older ones such as the T4, fp32 on CPU
    use_scaler = False
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, use_scaler = torch.float16, True
        autocast = lambda: torch.autocast("cuda", dtype=amp_dtype)
    else:
        amp_dtype = None
        autocast = contextlib.nullcontext
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---- model (architecture comes from the checkpoint when resuming)
    ckpt = None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = GPTConfig(**ckpt["config"])
        print(f"Resuming from {ckpt_path} at step {ckpt['step']}")
    else:
        if args.resume:
            print("No checkpoint found, starting fresh.")
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    assert cfg.vocab_size == meta["vocab_size"], "model vocab and data vocab differ"
    model = GPT(cfg).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    n_params = model.num_params()
    print(f"Device: {device}  precision: {str(amp_dtype).replace('torch.', '') if amp_dtype else 'fp32'}")
    print(f"Model: {n_params/1e6:.1f}M parameters  ({cfg.n_layer} layers, {cfg.n_embd} dims, "
          f"{cfg.n_head} heads, context {cfg.block_size}, vocab {cfg.vocab_size:,})")

    # ---- optimizer: weight decay only on matrices (not norms)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    step, best_val = 0, float("inf")
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        del ckpt

    run_model = torch.compile(model) if args.compile else model
    data = Data(args.data_dir, np.dtype(meta["dtype"]), cfg.block_size, device)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    min_lr = args.lr * args.min_lr_ratio
    print(f"{tokens_per_step:,} tokens per step, {args.max_steps:,} steps "
          f"= {tokens_per_step * args.max_steps / 1e6:.0f}M tokens\n")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        if s >= args.max_steps:
            return min_lr
        ratio = (s - args.warmup) / max(1, args.max_steps - args.warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_iters):
                x, y = data.batch(split, args.batch_size)
                with autocast():
                    _, loss = run_model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    log_path = os.path.join(args.out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tokens_seen"])

    model.train()
    t_start = time.time()
    t_last, step_last = t_start, step
    start_step = step
    try:
        while step < args.max_steps:
            lr_now = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr_now

            if step % args.eval_interval == 0 and not (step == start_step and start_step > 0):
                losses = estimate_loss()
                print(f"== step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                      f"(ppl {math.exp(min(losses['val'], 20)):.1f})")
                log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr_now:.2e}", step * tokens_per_step])
                log_f.flush()
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step,
                                "val_loss": best_val, "tokenizer": tok_path}, best_path)
                if step > start_step or step == 0:
                    save_ckpt(ckpt_path, model, opt, scaler, step, best_val, cfg, tokenizer=tok_path)

            opt.zero_grad(set_to_none=True)
            loss_acc = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x, y = data.batch("train", args.batch_size)
                with autocast():
                    _, loss = run_model(x, y)
                loss = loss / args.grad_accum
                loss_acc += loss.detach()
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % args.log_interval == 0:
                lv = loss_acc.item()
                if not math.isfinite(lv):
                    print(f"Loss became {lv} at step {step}. Stopping WITHOUT overwriting the last good checkpoint.")
                    sys.exit(1)
                now = time.time()
                tps = (step - step_last) * tokens_per_step / max(now - t_last, 1e-9)
                eta = (args.max_steps - step) * tokens_per_step / max(tps, 1)
                print(f"step {step:6d} | loss {lv:.4f} | lr {lr_now:.2e} | gnorm {float(gnorm):.2f} | "
                      f"{tps/1e3:.1f}k tok/s | ETA {eta/3600:.2f} h")
                t_last, step_last = now, step

            if args.time_budget_hours and (time.time() - t_start) > args.time_budget_hours * 3600:
                save_ckpt(ckpt_path, model, opt, scaler, step, best_val, cfg, tokenizer=tok_path)
                print(f"\nTime budget reached at step {step}. Checkpoint saved. Continue later with --resume.")
                return
    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        save_ckpt(ckpt_path, model, opt, scaler, step, best_val, cfg, tokenizer=tok_path)
        print(f"Saved at step {step}. Continue with --resume.")
        return

    losses = estimate_loss()
    print(f"\n== FINAL step {step} | train {losses['train']:.4f} | val {losses['val']:.4f} "
          f"(ppl {math.exp(min(losses['val'], 20)):.1f})")
    if losses["val"] < best_val:
        best_val = losses["val"]
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step,
                    "val_loss": best_val, "tokenizer": tok_path}, best_path)
    save_ckpt(ckpt_path, model, opt, scaler, step, best_val, cfg, tokenizer=tok_path)
    log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr_now:.2e}", step * tokens_per_step])
    log_f.close()
    print(f"Done. Best validation loss {best_val:.4f}. Weights: {best_path}")
    return {"best_val": best_val, "best_path": best_path, "ckpt_path": ckpt_path}


train = main

if __name__ == "__main__":
    main()

