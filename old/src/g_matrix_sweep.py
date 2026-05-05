"""
g_matrix_sweep.py
=======================================================================
G-Matrix Repulsion Amplification — GPT-2
=======================================================================

Actively manipulates the indefinite bilinear form (G = W_q W_k^T) by
amplifying its eigenvalues. Tests the hypothesis that negative eigenvalues
physically encode the model's "repulsion" (distraction-filtering) mechanism.

EXPERIMENTS
  Phase 1 — Zero-Shot Operator Isolation
    1.0 Diagnostic: Map PSD fraction across all layers to find lowest 3.
    1.1 Repulsion Sweep: Boost negative eigenvalues by alpha.
    1.2 Control Sweep: Boost positive eigenvalues by alpha.

  Phase 2 — Targeted Geometric Healing
    2.1 G-Repulsion Healing: Freeze W_q/W_k with optimal boosted G,
        fine-tune the OV circuit and MLPs to adapt to the new routing.

USAGE
  python g_matrix_sweep.py --phases 1
  python g_matrix_sweep.py --phases 2 --best-layer 0 --best-alpha 1.25
"""

import argparse
import gc
import math
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 12
D_MODEL = 768
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS
RESULTS_DIR = "repulsion_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_FT_STEPS = 1000
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 512
WARMUP_STEPS = 50

ALPHAS = [1.1, 1.25, 1.5, 2.0]


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    return m.eval().to(DEVICE)


# =============================================================================
# G-MATRIX MANIPULATION
# =============================================================================

def measure_psd_fraction(model):
    """
    Measures the Positive Semi-Definite (PSD) fraction for all layers.
    Calculates the percentage of eigenvalues in W_k^T W_q with real parts > 0.
    Lower PSD fraction = Higher natural repulsion.
    """
    print("\n  Measuring baseline PSD fractions (natural repulsion)...")
    psd_fractions = {}

    for layer in range(N_LAYERS):
        W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
        W_q = W[:, :D_MODEL]
        W_k = W[:, D_MODEL:2 * D_MODEL]

        total_eigenvalues = 0
        positive_eigenvalues = 0

        for h in range(N_HEADS):
            wq = W_q[:, h * D_HEAD:(h + 1) * D_HEAD]
            wk = W_k[:, h * D_HEAD:(h + 1) * D_HEAD]

            # G_small has the exact same non-zero eigenvalues as W_q @ W_k.T
            G_small = wk.T @ wq
            vals = np.linalg.eigvals(G_small)

            positive_eigenvalues += np.sum(vals.real > 0)
            total_eigenvalues += len(vals)

        psd_frac = positive_eigenvalues / total_eigenvalues
        psd_fractions[layer] = round(psd_frac * 100, 2)
        print(f"    Layer {layer:02d}: {psd_fractions[layer]:>5}% Positive (Attraction)")

    return psd_fractions


def inject_boosted_g(model, layer, alpha, mode="negative"):
    """
    Amplifies specific eigenvalues of G by alpha.
    Updates W_q to satisfy the new G, leaving W_k intact.
    """
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    W_q = W[:, :D_MODEL]
    W_k = W[:, D_MODEL:2 * D_MODEL]
    W_q_new = np.zeros_like(W_q)

    for h in range(N_HEADS):
        wq = W_q[:, h * D_HEAD:(h + 1) * D_HEAD]
        wk = W_k[:, h * D_HEAD:(h + 1) * D_HEAD]

        G_small = wk.T @ wq
        vals, vecs = np.linalg.eig(G_small)
        vals_new = vals.copy()

        if mode == "negative":
            mask = vals.real < 0
        else:  # positive control
            mask = vals.real > 0

        vals_new[mask] *= alpha

        # Reconstruct G_small
        G_small_new = vecs @ np.diag(vals_new) @ np.linalg.inv(vecs)
        G_small_new = np.real(G_small_new)  # Strip floating point imaginary dust

        # Project update onto W_q: wq_new = wk @ pinv(wk.T @ wk) @ G_small_new
        wq_new_h = wk @ np.linalg.pinv(wk.T @ wk) @ G_small_new
        W_q_new[:, h * D_HEAD:(h + 1) * D_HEAD] = wq_new_h

    W_new = W.copy()
    W_new[:, :D_MODEL] = W_q_new
    model.transformer.h[layer].attn.c_attn.weight.data = (
        torch.tensor(W_new, dtype=torch.float32).to(DEVICE)
    )
    return W_q_new


def freeze_qk_circuit(model, layer):
    """
    Freezes W_q and W_k slices in c_attn via gradient hook.
    Allows W_v and W_o to fine-tune to the new routing constraints.
    """
    attn = model.transformer.h[layer].attn

    def hook(grad):
        grad_copy = grad.clone()
        grad_copy[:, :2 * D_MODEL] = 0.0  # Zero W_q and W_k gradients
        return grad_copy

    return attn.c_attn.weight.register_hook(hook)


# =============================================================================
# EVALUATION
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = []
    for i in range(0, len(text), 100_000):
        chunk = tokenizer(
            text[i:i + 100_000], return_tensors="pt", add_special_tokens=False, truncation=False
        )["input_ids"].squeeze(0)
        ids.append(chunk)
    enc = torch.cat(ids, dim=0)

    stride, max_len, nlls = 512, 1024, []
    model.eval()
    with torch.no_grad():
        for b in range(0, enc.size(0) - max_len, stride):
            inp = enc[b:b + max_len].unsqueeze(0).to(DEVICE)
            tgt = inp.clone();
            tgt[:, :stride] = -100
            nlls.append(model(input_ids=inp, labels=tgt).loss.item() * (max_len - stride))
    return round(math.exp(sum(nlls) / (len(nlls) * (max_len - stride))), 4)


def measure_lambada_acc(model, tokenizer):
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except:
        ds = load_dataset("lambada", split="test")

    correct, total = 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  LAMBADA", leave=False):
        text = ex.get("text") or ex.get("passage", "")
        tokens = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)["input_ids"].squeeze(0)
        if tokens.shape[0] < 2: continue
        inp = tokens.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=inp).logits
        if logits[0, -2, :].argmax().item() == tokens[-1].item():
            correct += 1
        total += 1
    return round(correct / total * 100, 2) if total > 0 else None


def measure_hellaswag_acc(model, tokenizer, n_examples=500):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n_examples: ds = ds.select(range(min(n_examples, len(ds))))

    correct, total = 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  HellaSwag", leave=False):
        ctx = ex["ctx"]
        label = int(ex["label"])
        log_likelihoods = []
        for ending in ex["endings"]:
            tokens = tokenizer(ctx + " " + ending, return_tensors="pt", max_length=1024, truncation=True)[
                "input_ids"].squeeze(0)
            if tokens.shape[0] < 2:
                log_likelihoods.append(float("-inf"));
                continue
            ctx_tokens = tokenizer(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            n_ctx = min(len(ctx_tokens), len(tokens) - 1)
            inp = tokens.unsqueeze(0).to(DEVICE)
            tgt = inp.clone();
            tgt[:, :n_ctx] = -100
            with torch.no_grad():
                loss = model(input_ids=inp, labels=tgt).loss
            n_ending = (tgt != -100).sum().item()
            log_likelihoods.append(-loss.item() * n_ending if n_ending > 0 else float("-inf"))

        if log_likelihoods.index(max(log_likelihoods)) == label: correct += 1
        total += 1
    return round(correct / total * 100, 2) if total > 0 else None


def evaluate(model, tokenizer, label="", hellaswag_n=500):
    print(f"\n  Evaluating: {label}")
    t0 = time.time()
    wt2 = measure_wt2_ppl(model, tokenizer)
    lmda = measure_lambada_acc(model, tokenizer)
    hswag = measure_hellaswag_acc(model, tokenizer, n_examples=hellaswag_n)
    elapsed = round(time.time() - t0, 1)

    print(f"    WT2 PPL:       {wt2}")
    print(f"    LAMBADA Acc:   {lmda}%")
    print(f"    HellaSwag Acc: {hswag}%")
    print(f"    Eval time: {elapsed}s")

    return {"label": label, "wt2_ppl": wt2, "lambada_acc": lmda, "hellaswag_acc": hswag}


# =============================================================================
# FINE-TUNING UTILITIES
# =============================================================================

class TokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len=512, max_tokens=2_000_000):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer, total = [], 0
        for ex in self.ds:
            text = ex.get("text", "") or ""
            if not text.strip(): continue
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                inp = torch.tensor(chunk, dtype=torch.long)
                yield inp, inp.clone()
                total += self.seq_len
                if total >= self.max_tokens: return


def load_openwebtext(tokenizer, seq_len=512):
    print("  Loading fine-tuning corpus (OpenWebText)...")
    # Using streaming=True is CRITICAL for OpenWebText because it is ~40GB.
    # This prevents your RAM from exploding.
    try:
        ds = load_dataset("openwebtext", split="train", streaming=True)
        # Peek at the first item to ensure it's working
        next(iter(ds))
        return TokenDataset(ds, tokenizer, seq_len=seq_len)
    except Exception as e:
        print(f"  OpenWebText load failed ({e}). Falling back to WikiText-103...")
        # WikiText-103 is a great backup; it's much larger than WT2 and
        # distinct enough to prevent simple memorization.
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        return TokenDataset(ds, tokenizer, seq_len=seq_len)


# =============================================================================
# PHASES
# =============================================================================

def phase1_alpha_sweep(tokenizer, baseline, target_layers):
    """Experiment 1.1 & 1.2: Sweep alpha on negative (repulsion) and positive (attraction) eigenvalues."""
    print("\n" + "=" * 65)
    print("  PHASE 1 — G-Matrix Alpha Sweep")
    print("  Testing alpha multipliers on negative (Repulsion) vs positive (Attraction) eigenvalues.")
    print("=" * 65)

    results = []

    print(f"\n  {'Lyr':<3} {'Mode':<8} {'Alpha':<6} | {'WT2':>7} {'WT2Δ':>8} | {'LMDA':>6} {'LMDAΔ':>8} | {'HSwag':>6}")
    print("  " + "─" * 70)

    for layer in target_layers:
        for mode in ["negative", "positive"]:
            for alpha in ALPHAS:
                model = fresh_model()
                inject_boosted_g(model, layer, alpha, mode=mode)

                r = evaluate(model, tokenizer, label=f"L{layer}_{mode}_{alpha}", hellaswag_n=500)
                del model;
                gc.collect()

                d_wt2 = round((r["wt2_ppl"] - baseline["wt2_ppl"]) / baseline["wt2_ppl"] * 100, 1)
                d_lmda = round((r["lambada_acc"] - baseline["lambada_acc"]) / baseline["lambada_acc"] * 100, 1)

                print(
                    f"  {layer:<3} {mode:<8} {alpha:<6} | {r['wt2_ppl']:>7.2f} {d_wt2:>7.1f}% | {r['lambada_acc']:>5.2f}% {d_lmda:>7.1f}% | {r['hellaswag_acc']:>5.2f}%")

                results.append({
                    "layer": layer, "mode": mode, "alpha": alpha, **r,
                    "wt2_delta_pct": d_wt2, "lambada_delta_pct": d_lmda
                })
    return results


def phase2_targeted_healing(tokenizer, baseline, best_layer, best_alpha, steps=DEFAULT_FT_STEPS, lr=DEFAULT_LR):
    """Experiment 2.1: Targeted Geometric Healing of the Amplified QK Circuit."""
    print("\n" + "=" * 65)
    print(f"  PHASE 2.1 — G-Repulsion Healing (Layer {best_layer}, Alpha {best_alpha})")
    print("  Freeze amplified QK circuit → fine-tune OV circuit and MLPs")
    print("=" * 65)

    model = fresh_model()

    # 1. Inject and grab snapshot
    wq_snap = inject_boosted_g(model, best_layer, best_alpha, mode="negative")
    print(f"  Injected Alpha={best_alpha} Repulsion into L{best_layer}")

    pre_ft = evaluate(model, tokenizer, label="pre_ft")

    # 2. Freeze Embeddings and QK Circuit
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.requires_grad = False
    hook = freeze_qk_circuit(model, best_layer)
    print("  Embeddings and L{best_layer} QK circuit frozen.")

    # 3. Fine-tuning Setup (WT2 train set)
    corpus = load_openwebtext(tokenizer, seq_len=DEFAULT_SEQ_LEN)
    loader = DataLoader(corpus, batch_size=DEFAULT_BATCH_SIZE)

    c_attn_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if f"h.{best_layer}.attn.c_attn" in name:
            c_attn_params.append(param)
        else:
            other_params.append(param)

    optim = torch.optim.AdamW([
        {"params": c_attn_params, "weight_decay": 0.0, "lr": lr},
        {"params": other_params, "weight_decay": 0.01, "lr": lr}
    ])
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=steps)

    model.train()
    loss_history = []
    step = 0
    pbar = tqdm(total=steps, desc="  Fine-Tuning")

    for inp, tgt in loader:
        if step >= steps: break
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        optim.zero_grad()
        loss = model(input_ids=inp, labels=tgt).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        sched.step()

        # Mathematical Guarantee: Re-inject W_q to prevent optimizer drift
        with torch.no_grad():
            W = model.transformer.h[best_layer].attn.c_attn.weight.data.cpu().numpy().copy()
            W[:, :D_MODEL] = wq_snap
            model.transformer.h[best_layer].attn.c_attn.weight.data = torch.tensor(W, dtype=torch.float32).to(DEVICE)

        loss_val = round(loss.item(), 4)
        loss_history.append(loss_val)
        pbar.set_postfix({"loss": loss_val})
        pbar.update(1)
        step += 1

    pbar.close()
    hook.remove()

    post_ft = evaluate(model, tokenizer, label="post_ft")
    del model;
    gc.collect()

    print("\n  HEALING SUMMARY")
    print(f"  WT2 PPL:     {pre_ft['wt2_ppl']} -> {post_ft['wt2_ppl']} (Baseline: {baseline['wt2_ppl']})")
    print(
        f"  LAMBADA Acc: {pre_ft['lambada_acc']}% -> {post_ft['lambada_acc']}% (Baseline: {baseline['lambada_acc']}%)")
    print(
        f"  HellaSwag:   {pre_ft['hellaswag_acc']}% -> {post_ft['hellaswag_acc']}% (Baseline: {baseline['hellaswag_acc']}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phases", nargs="+", type=int, default=[1])
    p.add_argument("--best-layer", type=int, default=None)
    p.add_argument("--best-alpha", type=float, default=None)
    args = p.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n  Measuring Baseline...")
    bm = fresh_model()
    baseline = evaluate(bm, tokenizer, label="baseline")
    del bm;
    gc.collect()

    if 1 in args.phases:
        bm = fresh_model()
        psd_fracs = measure_psd_fraction(bm)
        del bm;
        gc.collect()

        # Select the 3 layers with the lowest PSD fraction (highest natural repulsion)
        target_layers = sorted(psd_fracs, key=psd_fracs.get)[:3]
        print(f"\n  Selected layers for Alpha Sweep: {target_layers}")

        phase1_alpha_sweep(tokenizer, baseline, target_layers)

    if 2 in args.phases:
        if args.best_layer is None or args.best_alpha is None:
            print("\n  ERROR: Phase 2 requires --best-layer and --best-alpha flags.")
            return
        phase2_targeted_healing(tokenizer, baseline, args.best_layer, args.best_alpha)


if __name__ == "__main__":
    main()