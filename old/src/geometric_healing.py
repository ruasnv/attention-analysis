"""
geometric_healing.py
=======================================================================
Geometric Healing Experiments — GPT-2
=======================================================================

Tests whether freezing each attention matrix in its pure geometric
operator class, then fine-tuning remaining parameters, improves
long-range contextual performance without degrading local fluency.

THEORETICAL MOTIVATION
  W_v = Q @ S. Q is an isometry (carries all geometric function).
  S encodes local prediction bias that interferes with long-range
  integration. If the network can adapt to W_v = Q alone, S is not
  geometrically necessary.

  W_o = Q @ S. Q_fidelity = 0.000 — S carries all function.
  Q is geometrically redundant in W_o. If S alone suffices, Q is
  a rotational artifact of training, not a functional requirement.

EXPERIMENTS
  Phase 1 — Diagnostic sweeps (zero-shot, no fine-tuning)
    1.1 Layer-wise W_v Q-surgery: replace W_v[layer] with polar Q
    1.2 Layer-wise W_o S-surgery: replace W_o[layer] with polar S

  Phase 2 — Targeted geometric healing (fine-tune after injection)
    2.1 W_v healing: freeze best layer(s) as Q, fine-tune rest
    2.2 W_o healing: freeze best layer(s) as S, fine-tune rest

  Phase 3 — Global geometric healing
    3.1 Global W_v healing: freeze ALL W_v as Q, fine-tune rest
    3.2 Global W_o healing: freeze ALL W_o as S, fine-tune rest

BENCHMARKS (three context-range tiers)
  WT2 PPL       — local fluency (next-token prediction)
  HellaSwag Acc — medium-range (sentence-level coherence)
  LAMBADA Acc   — long-range (passage-level context integration)

FINE-TUNING CORPUS
  OpenWebText (distinct from WT2 evaluation corpus)
  Falls back to wikitext-103 if OpenWebText unavailable.

USAGE
  python geometric_healing.py --phases 1            # diagnostic only
  python geometric_healing.py --phases 1 2          # diag + targeted
  python geometric_healing.py --phases 1 2 3        # all phases
  python geometric_healing.py --phases 3            # global only
  python geometric_healing.py --phases 2 --wv-layers 0 --wo-layers 0
  python geometric_healing.py --ft-steps 500 --lr 2e-5
  python geometric_healing.py --skip-hellaswag      # faster Phase 1
"""

import argparse
import gc
import json
import math
import os
import time
from datetime import datetime
import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import polar
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME   = "gpt2"
DEVICE       = "cuda"
N_LAYERS     = 12
D_MODEL      = 768
RESULTS_DIR  = "healing_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_FT_STEPS   = 1000
DEFAULT_LR         = 2e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN    = 512
WARMUP_STEPS       = 50

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    return m.eval().to(DEVICE)


def get_wv(model, layer):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    return W[:, 2*D_MODEL:].copy()


def get_wo(model, layer):
    return model.transformer.h[layer].attn.c_proj.weight.data.cpu().numpy().copy()


def set_wv(model, layer, W_new):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy().copy()
    W[:, 2*D_MODEL:] = W_new.astype(np.float32)
    model.transformer.h[layer].attn.c_attn.weight.data = (
        torch.tensor(W, dtype=torch.float32).to(DEVICE)
    )


def set_wo(model, layer, W_new):
    model.transformer.h[layer].attn.c_proj.weight.data = (
        torch.tensor(W_new.astype(np.float32), dtype=torch.float32).to(DEVICE)
    )


def preserve_norm(W_new, W_orig):
    n_o = np.linalg.norm(W_orig, "fro")
    n_n = np.linalg.norm(W_new,  "fro")
    return (W_new * (n_o / n_n) if n_n > 1e-8 else W_new).astype(np.float32)


# =============================================================================
# POLAR DECOMPOSITION
# =============================================================================

def get_Q(W):
    """Extract the orthogonal factor Q from polar decomposition W = Q @ S."""
    Q, S = polar(W.astype(np.float64))
    return preserve_norm(Q.astype(np.float32), W)


def get_S(W):
    """Extract the symmetric stretch factor S from polar decomposition W = Q @ S."""
    Q, S = polar(W.astype(np.float64))
    return preserve_norm(S.astype(np.float32), W)


def inject_wv_as_Q(model, layer):
    """Replace W_v[layer] with its polar rotation Q (pure isometry)."""
    W_v = get_wv(model, layer)
    Q   = get_Q(W_v)
    set_wv(model, layer, Q)


def inject_wo_as_S(model, layer):
    """Replace W_o[layer] with its polar stretch S (pure directional amplifier)."""
    W_o = get_wo(model, layer)
    S   = get_S(W_o)
    set_wo(model, layer, S)


# =============================================================================
# FREEZING UTILITIES
# =============================================================================

def freeze_wv_layers(model, layers):
    """
    Freeze W_v at specified layers.
    Note: GPT-2 stores W_q, W_k, W_v in a single fused c_attn weight.
    We cannot freeze W_v independently of W_q/W_k without a custom forward.
    Strategy: register a backward hook that zeroes the W_v gradient slice.
    """
    hooks = []
    for layer in layers:
        attn = model.transformer.h[layer].attn
        def make_hook(l):
            def hook(grad):
                # Zero the W_v slice (columns 2*D_MODEL onward)
                grad_copy = grad.clone()
                grad_copy[:, 2*D_MODEL:] = 0.0
                return grad_copy
            return hook
        h = attn.c_attn.weight.register_hook(make_hook(layer))
        hooks.append(h)
    return hooks


def freeze_wo_layers(model, layers):
    """Freeze W_o at specified layers via requires_grad=False."""
    for layer in layers:
        model.transformer.h[layer].attn.c_proj.weight.requires_grad_(False)
        if model.transformer.h[layer].attn.c_proj.bias is not None:
            model.transformer.h[layer].attn.c_proj.bias.requires_grad_(False)


def unfreeze_all(model, hooks):
    """Remove gradient hooks and re-enable all gradients."""
    for h in hooks:
        h.remove()
    for param in model.parameters():
        param.requires_grad_(True)


# =============================================================================
# EVALUATION
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    """WikiText-2 test set perplexity — local fluency measure."""
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids  = []
    for i in range(0, len(text), 100_000):
        chunk = tokenizer(
            text[i:i+100_000],
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        ids.append(chunk)
    enc = torch.cat(ids, dim=0)

    stride, max_len, nlls = 512, 1024, []
    model.eval()
    with torch.no_grad():
        for b in range(0, enc.size(0) - max_len, stride):
            inp = enc[b:b+max_len].unsqueeze(0).to(DEVICE)
            tgt = inp.clone(); tgt[:, :stride] = -100
            nlls.append(
                model(input_ids=inp, labels=tgt).loss.item() * (max_len - stride)
            )
    return round(math.exp(sum(nlls) / (len(nlls) * (max_len - stride))), 4)


def measure_lambada_acc(model, tokenizer):
    """
    LAMBADA accuracy — long-range contextual measure.
    Predicts the final word of a passage; requires full passage context.
    """
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except Exception:
        ds = load_dataset("lambada", split="test")

    correct, total = 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  LAMBADA", leave=False):
        text   = ex.get("text") or ex.get("passage", "")
        tokens = tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )["input_ids"].squeeze(0)
        if tokens.shape[0] < 2:
            continue
        inp = tokens.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=inp).logits
        if logits[0, -2, :].argmax().item() == tokens[-1].item():
            correct += 1
        total += 1
    if total == 0:
        return None
    return round(correct / total * 100, 2)


def measure_hellaswag_acc(model, tokenizer, n_examples=1000):
    """
    HellaSwag accuracy — medium-range coherence measure.
    Selects the most likely sentence continuation from 4 choices
    using log-likelihood scoring. Uses first n_examples for speed.
    """
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n_examples and n_examples < len(ds):
        ds = ds.select(range(n_examples))

    correct, total = 0, 0
    model.eval()

    for ex in tqdm(ds, desc="  HellaSwag", leave=False):
        ctx      = ex["ctx"]
        endings  = ex["endings"]
        label    = int(ex["label"])

        log_likelihoods = []
        for ending in endings:
            full_text = ctx + " " + ending
            tokens = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            )["input_ids"].squeeze(0)
            if tokens.shape[0] < 2:
                log_likelihoods.append(float("-inf"))
                continue

            # Score only the ending tokens
            ctx_tokens = tokenizer(
                ctx, return_tensors="pt", add_special_tokens=False
            )["input_ids"].squeeze(0)
            n_ctx = min(len(ctx_tokens), len(tokens) - 1)

            inp = tokens.unsqueeze(0).to(DEVICE)
            tgt = inp.clone()
            tgt[:, :n_ctx] = -100   # mask context, score only ending

            with torch.no_grad():
                loss = model(input_ids=inp, labels=tgt).loss
            n_ending_tokens = (tgt != -100).sum().item()
            log_likelihoods.append(
                -loss.item() * n_ending_tokens if n_ending_tokens > 0 else float("-inf")
            )

        if log_likelihoods.count(float("-inf")) == len(log_likelihoods):
            continue
        if log_likelihoods.index(max(log_likelihoods)) == label:
            correct += 1
        total += 1

    if total == 0:
        return None
    return round(correct / total * 100, 2)


def evaluate(model, tokenizer, label="", skip_hellaswag=False, hellaswag_n=500):
    """Run all three benchmarks and return a result dict."""
    print(f"\n  Evaluating: {label}")
    t0 = time.time()

    wt2 = measure_wt2_ppl(model, tokenizer)
    print(f"    WT2 PPL:       {wt2}")

    lmda = measure_lambada_acc(model, tokenizer)
    print(f"    LAMBADA Acc:   {lmda}%")

    if skip_hellaswag:
        hswag = None
        print(f"    HellaSwag Acc: skipped")
    else:
        hswag = measure_hellaswag_acc(model, tokenizer, n_examples=hellaswag_n)
        print(f"    HellaSwag Acc: {hswag}%")

    elapsed = round(time.time() - t0, 1)
    print(f"    Eval time: {elapsed}s")

    return {
        "label":         label,
        "wt2_ppl":       wt2,
        "lambada_acc":   lmda,
        "hellaswag_acc": hswag,
        "eval_seconds":  elapsed,
    }


# =============================================================================
# FINE-TUNING CORPUS
# =============================================================================

class TokenDataset(IterableDataset):
    """Streams tokenized text from a HuggingFace dataset."""
    def __init__(self, hf_dataset, tokenizer, seq_len=512, max_tokens=5_000_000):
        self.ds        = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        total = 0
        for ex in self.ds:
            text = ex.get("text", "") or ""
            if not text.strip():
                continue
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            buffer.extend(ids)
            # Fetch exact seq_len chunks (no +1 needed anymore)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                inp = torch.tensor(chunk, dtype=torch.long)
                tgt = inp.clone()  # HuggingFace handles the shift internally

                yield inp, tgt
                total += self.seq_len
                if total >= self.max_tokens:
                    return


def load_ft_corpus(tokenizer, seq_len=512):
    """Load OpenWebText for fine-tuning. Falls back to WikiText-103."""
    print("  Loading fine-tuning corpus (OpenWebText)...")
    try:
        ds = load_dataset("openwebtext", split="train", streaming=True)
        # Quick validation
        next(iter(ds))
        print("  OpenWebText loaded successfully.")
        return TokenDataset(ds, tokenizer, seq_len=seq_len)
    except Exception as e:
        print(f"  OpenWebText unavailable ({e}). Falling back to WikiText-103.")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        return TokenDataset(ds, tokenizer, seq_len=seq_len)


# =============================================================================
# FINE-TUNING
# =============================================================================

def fine_tune(
    model,
    tokenizer,
    steps=DEFAULT_FT_STEPS,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_BATCH_SIZE,
    seq_len=DEFAULT_SEQ_LEN,
    desc="fine-tuning",
    frozen_wv_layers=None,
    frozen_wo_layers=None,
    wv_snapshots=None,
    wo_snapshots=None,
):
    """
    Fine-tune model on OpenWebText for a fixed number of gradient steps.
    Only parameters with requires_grad=True are updated.
    """
    frozen_wv_layers = frozen_wv_layers or []
    frozen_wo_layers = frozen_wo_layers or []
    wv_snapshots     = wv_snapshots or {}
    wo_snapshots     = wo_snapshots or {}
    corpus = load_ft_corpus(tokenizer, seq_len=seq_len)
    loader = DataLoader(corpus, batch_size=batch_size)

    # ── Freeze Embeddings ─────────────────────────────────────────────────
    # We freeze wte (token embeddings) and wpe (position embeddings)
    # to prevent vocabulary drift and force the network to adapt structurally.
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.requires_grad = False

    # ── Parameter groups — no weight decay on fused c_attn tensors ────────
    c_attn_params  = []
    other_params   = []
    c_attn_names   = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "c_attn" in name and any(
            f"h.{l}." in name for l in frozen_wv_layers
        ):
            c_attn_params.append(param)
            c_attn_names.add(name)
        else:
            other_params.append(param)

    param_groups = [
        {"params": c_attn_params, "weight_decay": 0.0,  "lr": lr},
        {"params": other_params,  "weight_decay": 0.01, "lr": lr},
    ]

    n_c_attn   = sum(p.numel() for p in c_attn_params)
    n_other    = sum(p.numel() for p in other_params)
    n_total    = n_c_attn + n_other
    print(f"  Trainable parameters: {n_total:,}")
    print(f"    c_attn (wd=0.0):  {n_c_attn:,}  ← protects frozen W_v slices")
    print(f"    other  (wd=0.01): {n_other:,}")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=steps,
    )

    model.train()
    loss_history = []
    step = 0

    pbar = tqdm(total=steps, desc=f"  {desc}")
    for inp, tgt in loader:
        if step >= steps:
            break
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        loss = model(input_ids=inp, labels=tgt).loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        # ── Re-inject frozen slices after every optimizer step ─────────
        with torch.no_grad():
            for layer, Q in wv_snapshots.items():
                W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy().copy()
                W[:, 2*D_MODEL:] = Q.astype(np.float32)
                model.transformer.h[layer].attn.c_attn.weight.data = (
                    torch.tensor(W, dtype=torch.float32).to(DEVICE)
                )
            for layer, S in wo_snapshots.items():
                model.transformer.h[layer].attn.c_proj.weight.data = (
                    torch.tensor(S.astype(np.float32), dtype=torch.float32).to(DEVICE)
                )

        loss_val = round(loss.item(), 4)
        loss_history.append(loss_val)
        pbar.set_postfix({"loss": loss_val, "step": step})
        pbar.update(1)
        step += 1

    pbar.close()
    model.eval()
    print(f"  Fine-tuning complete. Final loss: {loss_history[-1]:.4f}")

    # ── Final re-injection to guarantee pristine geometry at eval ──────
    with torch.no_grad():
        for layer, Q in wv_snapshots.items():
            W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy().copy()
            W[:, 2*D_MODEL:] = Q.astype(np.float32)
            model.transformer.h[layer].attn.c_attn.weight.data = (
                torch.tensor(W, dtype=torch.float32).to(DEVICE)
            )
        for layer, S in wo_snapshots.items():
            model.transformer.h[layer].attn.c_proj.weight.data = (
                torch.tensor(S.astype(np.float32), dtype=torch.float32).to(DEVICE)
            )
    if wv_snapshots or wo_snapshots:
        print(f"  Frozen geometry re-injected at final step "
              f"(Wv: {list(wv_snapshots.keys())}, "
              f"Wo: {list(wo_snapshots.keys())})")

    return loss_history


# =============================================================================
# PHASE 1 — DIAGNOSTIC SWEEPS
# =============================================================================

def phase1_wv_sweep(tokenizer, baseline, skip_hellaswag, hellaswag_n):
    """
    Experiment 1.1: Layer-wise W_v Q-surgery sweep.
    For each layer, replace W_v with polar Q. Evaluate. Restore.
    """
    print("\n" + "="*65)
    print("  PHASE 1.1 — Layer-wise W_v Q-surgery sweep")
    print("  Replace W_v[layer] with polar Q. All other layers untouched.")
    print("="*65)

    results = []
    b = baseline

    print(f"\n  {'Layer':<7} {'WT2':>8} {'WT2Δ':>8} "
          f"{'LMDA':>8} {'LMDAδ':>8} {'HSwag':>8} {'HSwagδ':>8}")
    print("  " + "─"*56)

    for layer in range(N_LAYERS):
        model = fresh_model()
        inject_wv_as_Q(model, layer)
        r = evaluate(model, tokenizer,
                     label=f"wv_Q_L{layer:02d}",
                     skip_hellaswag=skip_hellaswag,
                     hellaswag_n=hellaswag_n)
        del model; gc.collect()

        d_wt2  = round((r["wt2_ppl"]       - b["wt2_ppl"])       / b["wt2_ppl"]       * 100, 1) if b["wt2_ppl"] else None
        d_lmda = round((r["lambada_acc"]    - b["lambada_acc"])    / b["lambada_acc"]    * 100, 1) if b.get("lambada_acc") else None
        d_hs   = round((r["hellaswag_acc"]  - b["hellaswag_acc"]) / b["hellaswag_acc"] * 100, 1) if (r["hellaswag_acc"] and b.get("hellaswag_acc")) else None

        hs_str   = f"{r['hellaswag_acc']:.1f}%" if r["hellaswag_acc"] else "—"
        d_hs_str = f"{d_hs:+.1f}%" if d_hs is not None else "—"

        print(f"  L{layer:02d}    {r['wt2_ppl']:>8.2f} {(str(d_wt2)+'%') if d_wt2 else '—':>8} "
              f"{str(r['lambada_acc'])+('%' if r['lambada_acc'] else ''):>8} "
              f"{(str(d_lmda)+'%') if d_lmda else '—':>8} "
              f"{hs_str:>8} {d_hs_str:>8}")

        results.append({
            "layer": layer,
            **r,
            "wt2_delta_pct":      d_wt2,
            "lambada_delta_pct":  d_lmda,
            "hellaswag_delta_pct":d_hs,
        })

    # Identify best layer(s) — most LAMBADA improvement + lowest WT2 cost
    valid = [r for r in results if r["lambada_acc"] is not None]
    if valid:
        best_lmda = max(valid, key=lambda r: r["lambada_acc"])
        best_wt2  = min(valid, key=lambda r: r["wt2_ppl"])
        print(f"\n  Best LAMBADA: L{best_lmda['layer']:02d} "
              f"({best_lmda['lambada_acc']}%, Δ{best_lmda['lambada_delta_pct']:+.1f}%)")
        print(f"  Best WT2:     L{best_wt2['layer']:02d} "
              f"(PPL={best_wt2['wt2_ppl']}, Δ{best_wt2['wt2_delta_pct']:+.1f}%)")

    return results


def phase1_wo_sweep(tokenizer, baseline, skip_hellaswag, hellaswag_n):
    """
    Experiment 1.2: Layer-wise W_o S-surgery sweep.
    For each layer, replace W_o with polar S. Evaluate. Restore.
    """
    print("\n" + "="*65)
    print("  PHASE 1.2 — Layer-wise W_o S-surgery sweep")
    print("  Replace W_o[layer] with polar S. All other layers untouched.")
    print("="*65)

    results = []
    b = baseline

    print(f"\n  {'Layer':<7} {'WT2':>8} {'WT2Δ':>8} "
          f"{'LMDA':>8} {'LMDAδ':>8} {'HSwag':>8} {'HSwagδ':>8}")
    print("  " + "─"*56)

    for layer in range(N_LAYERS):
        model = fresh_model()
        inject_wo_as_S(model, layer)
        r = evaluate(model, tokenizer,
                     label=f"wo_S_L{layer:02d}",
                     skip_hellaswag=skip_hellaswag,
                     hellaswag_n=hellaswag_n)
        del model; gc.collect()

        d_wt2  = round((r["wt2_ppl"]       - b["wt2_ppl"])       / b["wt2_ppl"]       * 100, 1) if b["wt2_ppl"] else None
        d_lmda = round((r["lambada_acc"]    - b["lambada_acc"])    / b["lambada_acc"]    * 100, 1) if b.get("lambada_acc") else None
        d_hs   = round((r["hellaswag_acc"]  - b["hellaswag_acc"]) / b["hellaswag_acc"] * 100, 1) if (r["hellaswag_acc"] and b.get("hellaswag_acc")) else None

        hs_str   = f"{r['hellaswag_acc']:.1f}%" if r["hellaswag_acc"] else "—"
        d_hs_str = f"{d_hs:+.1f}%" if d_hs is not None else "—"

        print(f"  L{layer:02d}    {r['wt2_ppl']:>8.2f} {(str(d_wt2)+'%') if d_wt2 else '—':>8} "
              f"{str(r['lambada_acc'])+('%' if r['lambada_acc'] else ''):>8} "
              f"{(str(d_lmda)+'%') if d_lmda else '—':>8} "
              f"{hs_str:>8} {d_hs_str:>8}")

        results.append({
            "layer": layer,
            **r,
            "wt2_delta_pct":      d_wt2,
            "lambada_delta_pct":  d_lmda,
            "hellaswag_delta_pct":d_hs,
        })

    valid = [r for r in results if r["wt2_ppl"] is not None]
    if valid:
        most_resilient = min(valid, key=lambda r: r["wt2_ppl"])
        print(f"\n  Most resilient layer: L{most_resilient['layer']:02d} "
              f"(WT2={most_resilient['wt2_ppl']}, "
              f"Δ{most_resilient['wt2_delta_pct']:+.1f}%)")

    return results


# =============================================================================
# PHASE 2 — TARGETED GEOMETRIC HEALING
# =============================================================================

def phase2_wv_healing(tokenizer, baseline, wv_layers, ft_steps, lr,
                       skip_hellaswag, hellaswag_n):
    """
    Experiment 2.1: W_v targeted healing.
    Inject Q into selected layers. Freeze W_v gradient there.
    Fine-tune all other parameters. Evaluate before and after.
    """
    print("\n" + "="*65)
    print(f"  PHASE 2.1 — W_v targeted healing (layers {wv_layers})")
    print(f"  Inject Q → freeze W_v at those layers → fine-tune rest")
    print("="*65)

    model = fresh_model()

    # Inject Q at selected layers
    for layer in wv_layers:
        inject_wv_as_Q(model, layer)
        print(f"  Injected Q into W_v L{layer:02d}")

    # Evaluate before fine-tuning
    pre_ft = evaluate(model, tokenizer,
                      label=f"wv_Q_pre_ft_L{'_'.join(str(l) for l in wv_layers)}",
                      skip_hellaswag=skip_hellaswag,
                      hellaswag_n=hellaswag_n)

    # Freeze W_v gradient slices at selected layers
    hooks = freeze_wv_layers(model, wv_layers)
    print(f"  W_v frozen at layers {wv_layers} via gradient hooks")

    # Capture exact Q snapshots before training for re-injection
    wv_snapshots = {layer: get_wv(model, layer).copy() for layer in wv_layers}

    # Fine-tune remaining parameters
    loss_history = fine_tune(
        model, tokenizer,
        steps=ft_steps, lr=lr,
        desc=f"W_v healing L{wv_layers}",
        frozen_wv_layers=wv_layers,
        wv_snapshots=wv_snapshots,
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Evaluate after fine-tuning
    post_ft = evaluate(model, tokenizer,
                       label=f"wv_Q_post_ft_L{'_'.join(str(l) for l in wv_layers)}",
                       skip_hellaswag=skip_hellaswag,
                       hellaswag_n=hellaswag_n)

    del model; gc.collect()

    b = baseline
    result = {
        "experiment":    "2.1_wv_targeted",
        "frozen_layers": wv_layers,
        "ft_steps":      ft_steps,
        "lr":            lr,
        "pre_ft":        pre_ft,
        "post_ft":       post_ft,
        "loss_history":  loss_history,
        "deltas_vs_baseline": {
            "wt2_pre":  round((pre_ft["wt2_ppl"]  - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "wt2_post": round((post_ft["wt2_ppl"] - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "lmda_pre":  round((pre_ft["lambada_acc"]  - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
            "lmda_post": round((post_ft["lambada_acc"] - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
        }
    }

    _print_healing_summary(result)
    return result


def phase2_wo_healing(tokenizer, baseline, wo_layers, ft_steps, lr,
                       skip_hellaswag, hellaswag_n):
    """
    Experiment 2.2: W_o targeted healing.
    Inject S into selected layers. Freeze W_o there.
    Fine-tune all other parameters. Evaluate before and after.
    """
    print("\n" + "="*65)
    print(f"  PHASE 2.2 — W_o targeted healing (layers {wo_layers})")
    print(f"  Inject S → freeze W_o at those layers → fine-tune rest")
    print("="*65)

    model = fresh_model()

    for layer in wo_layers:
        inject_wo_as_S(model, layer)
        print(f"  Injected S into W_o L{layer:02d}")

    pre_ft = evaluate(model, tokenizer,
                      label=f"wo_S_pre_ft_L{'_'.join(str(l) for l in wo_layers)}",
                      skip_hellaswag=skip_hellaswag,
                      hellaswag_n=hellaswag_n)

    freeze_wo_layers(model, wo_layers)
    print(f"  W_o frozen at layers {wo_layers}")

    wo_snapshots = {layer: get_wo(model, layer).copy() for layer in wo_layers}

    loss_history = fine_tune(
        model, tokenizer,
        steps=ft_steps, lr=lr,
        desc=f"W_o healing L{wo_layers}",
        frozen_wo_layers=wo_layers,
        wo_snapshots=wo_snapshots,
    )

    # Re-enable gradients for W_o (in case further experiments follow)
    for layer in wo_layers:
        model.transformer.h[layer].attn.c_proj.weight.requires_grad_(True)

    post_ft = evaluate(model, tokenizer,
                       label=f"wo_S_post_ft_L{'_'.join(str(l) for l in wo_layers)}",
                       skip_hellaswag=skip_hellaswag,
                       hellaswag_n=hellaswag_n)

    del model; gc.collect()

    b = baseline
    result = {
        "experiment":    "2.2_wo_targeted",
        "frozen_layers": wo_layers,
        "ft_steps":      ft_steps,
        "lr":            lr,
        "pre_ft":        pre_ft,
        "post_ft":       post_ft,
        "loss_history":  loss_history,
        "deltas_vs_baseline": {
            "wt2_pre":  round((pre_ft["wt2_ppl"]  - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "wt2_post": round((post_ft["wt2_ppl"] - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "lmda_pre":  round((pre_ft["lambada_acc"]  - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
            "lmda_post": round((post_ft["lambada_acc"] - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
        }
    }

    _print_healing_summary(result)
    return result


# =============================================================================
# PHASE 3 — GLOBAL GEOMETRIC HEALING
# =============================================================================

def phase3_wv_global(tokenizer, baseline, ft_steps, lr,
                      skip_hellaswag, hellaswag_n):
    """
    Experiment 3.1: Global W_v healing.
    Inject Q across ALL 12 W_v layers. Freeze all. Fine-tune rest.
    """
    print("\n" + "="*65)
    print("  PHASE 3.1 — Global W_v healing (all 12 layers)")
    print("  Inject Q into all W_v → freeze all → fine-tune rest")
    print("="*65)

    model = fresh_model()
    all_layers = list(range(N_LAYERS))

    for layer in all_layers:
        inject_wv_as_Q(model, layer)
    print(f"  Injected Q into all 12 W_v layers")

    pre_ft = evaluate(model, tokenizer,
                      label="wv_Q_global_pre_ft",
                      skip_hellaswag=skip_hellaswag,
                      hellaswag_n=hellaswag_n)

    hooks = freeze_wv_layers(model, all_layers)
    print(f"  All W_v layers frozen via gradient hooks")

    wv_snapshots = {l: get_wv(model, l).copy() for l in all_layers}

    loss_history = fine_tune(
        model, tokenizer,
        steps=ft_steps, lr=lr,
        desc="Global W_v healing",
        frozen_wv_layers=all_layers,
        wv_snapshots=wv_snapshots,
    )

    for h in hooks:
        h.remove()

    post_ft = evaluate(model, tokenizer,
                       label="wv_Q_global_post_ft",
                       skip_hellaswag=skip_hellaswag,
                       hellaswag_n=hellaswag_n)

    del model; gc.collect()

    b = baseline
    result = {
        "experiment":   "3.1_wv_global",
        "ft_steps":     ft_steps,
        "lr":           lr,
        "pre_ft":       pre_ft,
        "post_ft":      post_ft,
        "loss_history": loss_history,
        "deltas_vs_baseline": {
            "wt2_pre":   round((pre_ft["wt2_ppl"]  - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "wt2_post":  round((post_ft["wt2_ppl"] - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "lmda_pre":  round((pre_ft["lambada_acc"]  - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
            "lmda_post": round((post_ft["lambada_acc"] - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
        }
    }

    _print_healing_summary(result)
    return result


def phase3_wo_global(tokenizer, baseline, ft_steps, lr,
                      skip_hellaswag, hellaswag_n):
    """
    Experiment 3.2: Global W_o healing.
    Inject S across ALL 12 W_o layers. Freeze all. Fine-tune rest.
    """
    print("\n" + "="*65)
    print("  PHASE 3.2 — Global W_o healing (all 12 layers)")
    print("  Inject S into all W_o → freeze all → fine-tune rest")
    print("="*65)

    model = fresh_model()
    all_layers = list(range(N_LAYERS))

    for layer in all_layers:
        inject_wo_as_S(model, layer)
    print(f"  Injected S into all 12 W_o layers")

    pre_ft = evaluate(model, tokenizer,
                      label="wo_S_global_pre_ft",
                      skip_hellaswag=skip_hellaswag,
                      hellaswag_n=hellaswag_n)

    freeze_wo_layers(model, all_layers)
    print(f"  All W_o layers frozen")

    wo_snapshots = {l: get_wo(model, l).copy() for l in all_layers}

    loss_history = fine_tune(
        model, tokenizer,
        steps=ft_steps, lr=lr,
        desc="Global W_o healing",
        frozen_wo_layers=all_layers,
        wo_snapshots=wo_snapshots,
    )

    post_ft = evaluate(model, tokenizer,
                       label="wo_S_global_post_ft",
                       skip_hellaswag=skip_hellaswag,
                       hellaswag_n=hellaswag_n)

    del model; gc.collect()

    b = baseline
    result = {
        "experiment":   "3.2_wo_global",
        "ft_steps":     ft_steps,
        "lr":           lr,
        "pre_ft":       pre_ft,
        "post_ft":      post_ft,
        "loss_history": loss_history,
        "deltas_vs_baseline": {
            "wt2_pre":   round((pre_ft["wt2_ppl"]  - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "wt2_post":  round((post_ft["wt2_ppl"] - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "lmda_pre":  round((pre_ft["lambada_acc"]  - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
            "lmda_post": round((post_ft["lambada_acc"] - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
        }
    }

    _print_healing_summary(result)
    return result

# =============================================================================
# PHASE 4 — THE FRANKENSTEIN RUN
# =============================================================================

def phase4_frankenstein(tokenizer, baseline, wv_layers, wo_layers, ft_steps, lr, skip_hellaswag, hellaswag_n):
    """
    Simultaneous targeted healing.
    Injects Q into W_v layers AND S into W_o layers on the SAME model,
    freezes all of them, and fine-tunes the rest of the network.
    """
    print("\n" + "="*65)
    print(f"  PHASE 4 — The Frankenstein Run")
    print(f"  W_v Q-Surgery: {wv_layers} | W_o S-Surgery: {wo_layers}")
    print("="*65)

    model = fresh_model()

    # 1. Inject Geometry
    for layer in wv_layers:
        inject_wv_as_Q(model, layer)
        print(f"  Injected Q into W_v L{layer:02d}")
    for layer in wo_layers:
        inject_wo_as_S(model, layer)
        print(f"  Injected S into W_o L{layer:02d}")

    # 2. Evaluate Zero-Shot
    pre_ft = evaluate(model, tokenizer,
                      label=f"frankenstein_pre_ft",
                      skip_hellaswag=skip_hellaswag,
                      hellaswag_n=hellaswag_n)

    # 3. Freeze specific slices
    hooks = freeze_wv_layers(model, wv_layers)
    freeze_wo_layers(model, wo_layers)
    print(f"  Geometry locked via requires_grad and backward hooks.")

    # Capture snapshots for re-injection
    wv_snapshots = {l: get_wv(model, l).copy() for l in wv_layers}
    wo_snapshots = {l: get_wo(model, l).copy() for l in wo_layers}

    # 4. Fine-Tune
    loss_history = fine_tune(
        model, tokenizer,
        steps=ft_steps, lr=lr,
        desc="Frankenstein Healing",
        frozen_wv_layers=wv_layers,
        frozen_wo_layers=wo_layers,
        wv_snapshots=wv_snapshots,
        wo_snapshots=wo_snapshots,
    )

    # 5. Cleanup and Final Eval
    for h in hooks:
        h.remove()

    post_ft = evaluate(model, tokenizer,
                       label=f"frankenstein_post_ft",
                       skip_hellaswag=skip_hellaswag,
                       hellaswag_n=hellaswag_n)

    del model; gc.collect()

    b = baseline
    result = {
        "experiment":    "4.0_frankenstein",
        "wv_layers":     wv_layers,
        "wo_layers":     wo_layers,
        "ft_steps":      ft_steps,
        "pre_ft":        pre_ft,
        "post_ft":       post_ft,
        "deltas_vs_baseline": {
            "wt2_pre":  round((pre_ft["wt2_ppl"]  - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "wt2_post": round((post_ft["wt2_ppl"] - b["wt2_ppl"]) / b["wt2_ppl"] * 100, 2) if b["wt2_ppl"] else None,
            "lmda_pre":  round((pre_ft["lambada_acc"]  - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
            "lmda_post": round((post_ft["lambada_acc"] - b["lambada_acc"]) / b["lambada_acc"] * 100, 2) if b.get("lambada_acc") else None,
        }
    }

    _print_healing_summary(result)
    return result

# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def _print_healing_summary(result):
    d = result["deltas_vs_baseline"]
    print(f"\n  HEALING SUMMARY ({result['experiment']})")
    print(f"  {'Metric':<14} {'Pre-FT':>10} {'Post-FT':>10} {'Change':>10}")
    print(f"  {'─'*46}")

    def fmt(val):
        return f"{val:+.1f}%" if val is not None else "—"

    pre  = result["pre_ft"]
    post = result["post_ft"]

    wt2_change  = round(post["wt2_ppl"]     - pre["wt2_ppl"],    2)  if pre["wt2_ppl"]     else None
    lmda_change = round(post["lambada_acc"] - pre["lambada_acc"], 2) if pre.get("lambada_acc") else None

    print(f"  {'WT2 PPL':<14} {str(pre['wt2_ppl']):>10} {str(post['wt2_ppl']):>10} "
          f"{(str(wt2_change) if wt2_change is not None else '—'):>10}")
    print(f"  {'LAMBADA Acc':<14} {str(pre.get('lambada_acc','—'))+'%':>10} "
          f"{str(post.get('lambada_acc','—'))+'%':>10} "
          f"{(str(lmda_change)+'pp' if lmda_change is not None else '—'):>10}")
    print(f"\n  vs Baseline:")
    print(f"    WT2:    pre={fmt(d['wt2_pre'])}  post={fmt(d['wt2_post'])}")
    print(f"    LAMBADA: pre={fmt(d['lmda_pre'])}  post={fmt(d['lmda_post'])}")

    if d["wt2_post"] is not None and d["lmda_post"] is not None:
        if d["wt2_post"] <= 5.0 and d["lmda_post"] >= 0:
            print(f"\n  ✓ HEALING SUCCEEDED — local fluency preserved, long-range maintained/improved")
        elif d["lmda_post"] > d.get("lmda_pre", 0):
            print(f"\n  ~ PARTIAL HEALING — long-range improved but local fluency cost remains")
        else:
            print(f"\n  ✗ HEALING FAILED — performance did not recover")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Geometric healing experiments for GPT-2"
    )
    p.add_argument("--wv-layers", nargs="+", type=int, default=None,
                   help="W_v layers for Phase 2.1 (default: auto from Phase 1)")
    p.add_argument("--wo-layers", nargs="+", type=int, default=None,
                   help="W_o layers for Phase 2.2 (default: auto from Phase 1)")
    p.add_argument("--ft-steps",  type=int,   default=DEFAULT_FT_STEPS,
                   help=f"Fine-tuning steps (default: {DEFAULT_FT_STEPS})")
    p.add_argument("--lr",        type=float, default=DEFAULT_LR,
                   help=f"Learning rate (default: {DEFAULT_LR})")
    p.add_argument("--batch-size",type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len",   type=int,   default=DEFAULT_SEQ_LEN)
    p.add_argument("--hellaswag-n", type=int, default=500,
                   help="Number of HellaSwag examples (default: 500)")
    p.add_argument("--skip-hellaswag", action="store_true",
                   help="Skip HellaSwag in Phase 1 for speed")
    p.add_argument("--wv-only",   action="store_true",
                   help="Run only W_v experiments (skip W_o)")
    p.add_argument("--wo-only",   action="store_true",
                   help="Run only W_o experiments (skip W_v)")
    p.add_argument("--phases", nargs="+", type=int, default=[1],
                   choices=[1, 2, 3, 4],  # Added 4 here
                   help="Which phases to run: 1=diag, 2=targeted, 3=global, 4=frankenstein")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  Geometric Healing Experiments — {MODEL_NAME}")
    print(f"  Phases: {args.phases}  Device: {DEVICE}  {ts}")
    print(f"{'='*65}")

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {
        "meta": {
            "model": MODEL_NAME,
            "timestamp": ts,
            "device": DEVICE,
            "phases": args.phases,
            "ft_steps": args.ft_steps,
            "lr": args.lr,
        }
    }

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n  Measuring baseline...")
    bm = fresh_model()
    baseline = evaluate(bm, tokenizer, label="baseline",
                        skip_hellaswag=args.skip_hellaswag,
                        hellaswag_n=args.hellaswag_n)
    del bm; gc.collect()
    all_results["baseline"] = baseline

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if 1 in args.phases:
        p1 = {}

        if not args.wo_only:
            p1["wv_sweep"] = phase1_wv_sweep(
                tokenizer, baseline,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        if not args.wv_only:
            p1["wo_sweep"] = phase1_wo_sweep(
                tokenizer, baseline,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        all_results["phase1"] = p1

        # Auto-select layers for Phase 2 if not specified
        if args.wv_layers is None and "wv_sweep" in p1:
            valid = [r for r in p1["wv_sweep"] if r.get("lambada_acc")]
            if valid:
                best = max(valid, key=lambda r: r["lambada_acc"])
                args.wv_layers = [best["layer"]]
                print(f"\n  Auto-selected W_v layer for Phase 2: L{best['layer']:02d}")

        if args.wo_layers is None and "wo_sweep" in p1:
            valid = [r for r in p1["wo_sweep"] if r.get("wt2_ppl")]
            if valid:
                best = min(valid, key=lambda r: r["wt2_ppl"])
                args.wo_layers = [best["layer"]]
                print(f"  Auto-selected W_o layer for Phase 2: L{best['layer']:02d}")

    # Defaults if Phase 1 was skipped
    if args.wv_layers is None:
        args.wv_layers = [0]
    if args.wo_layers is None:
        args.wo_layers = [0]

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if 2 in args.phases:
        p2 = {}

        if not args.wo_only:
            p2["wv_healing"] = phase2_wv_healing(
                tokenizer, baseline,
                wv_layers=args.wv_layers,
                ft_steps=args.ft_steps,
                lr=args.lr,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        if not args.wv_only:
            p2["wo_healing"] = phase2_wo_healing(
                tokenizer, baseline,
                wo_layers=args.wo_layers,
                ft_steps=args.ft_steps,
                lr=args.lr,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        all_results["phase2"] = p2

    # ── Phase 3 ───────────────────────────────────────────────────────────
    if 3 in args.phases:
        p3 = {}

        if not args.wo_only:
            p3["wv_global"] = phase3_wv_global(
                tokenizer, baseline,
                ft_steps=args.ft_steps,
                lr=args.lr,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        if not args.wv_only:
            p3["wo_global"] = phase3_wo_global(
                tokenizer, baseline,
                ft_steps=args.ft_steps,
                lr=args.lr,
                skip_hellaswag=args.skip_hellaswag,
                hellaswag_n=args.hellaswag_n,
            )

        all_results["phase3"] = p3

    # ── Phase 4 ───────────────────────────────────────────────────────────
    if 4 in args.phases:
        all_results["phase4"] = phase4_frankenstein(
            tokenizer, baseline,
            wv_layers=args.wv_layers or [0],
            wo_layers=args.wo_layers or [8],
            ft_steps=args.ft_steps,
            lr=args.lr,
            skip_hellaswag=args.skip_hellaswag,
            hellaswag_n=args.hellaswag_n,
        )

    # ── Save results ──────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"healing_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()