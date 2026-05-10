"""
surgical_interventions.py
=======================================================================
Precision Geometric Surgeries — GPT-2
=======================================================================

Executes advanced, non-destructive geometric interventions on the OV circuit
to achieve Pareto improvements in long-range reasoning (LAMBADA) without
sacrificing local fluency (WT2).

EXPERIMENTS
  1. Spectral Clipping (Soft Debiasing):
     Decomposes W_v using SVD, clips the massive Marchenko-Pastur outlier
     singular values (the aggressive local bias), and reconstructs the matrix.

  2. Head-Wise Isolation (Spatial Precision):
     Sweeps through the 12 individual attention heads of a specific layer.
     Applies Q-surgery (polar decomposition) strictly to one head at a time
     to locate the exact spatial coordinates of the local bias amplifier.

USAGE
  python surgical_interventions.py --experiments 1 2 --target-layer 0
"""

import argparse
import gc
import math
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import polar
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 768
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    return m.eval().to(DEVICE)


def get_wv(model, layer):
    """Extracts the 768x768 W_v matrix from the fused c_attn block."""
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    return W[:, 2 * D_MODEL:].copy()


def set_wv(model, layer, W_new):
    """Injects a 768x768 matrix back into the W_v slice of c_attn."""
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy().copy()
    W[:, 2 * D_MODEL:] = W_new.astype(np.float32)
    model.transformer.h[layer].attn.c_attn.weight.data = torch.tensor(W, dtype=torch.float32).to(DEVICE)


# =============================================================================
# SURGICAL OPERATIONS
# =============================================================================

def apply_spectral_clipping(model, layer, clip_count=3):
    """
    Experiment 1: Soft Debiasing via SVD.
    Clips the top `clip_count` singular values to match the next highest value.
    """
    W_v = get_wv(model, layer)

    # Decompose into U, Sigma, V^T
    U, S, Vh = np.linalg.svd(W_v, full_matrices=False)

    # Clip the outliers (the massive local bias amplifiers)
    S_clipped = S.copy()
    clip_threshold = S_clipped[clip_count]
    S_clipped[:clip_count] = clip_threshold

    # Reconstruct the matrix
    W_v_new = U @ np.diag(S_clipped) @ Vh

    # Preserve original Frobenius norm to prevent arbitrary dimming of the layer
    norm_orig = np.linalg.norm(W_v, 'fro')
    norm_new = np.linalg.norm(W_v_new, 'fro')
    W_v_new = W_v_new * (norm_orig / norm_new)

    set_wv(model, layer, W_v_new)
    return S[:5], S_clipped[:5]  # Return top 5 for logging


def apply_head_wise_q_surgery(model, layer, target_head):
    """
    Experiment 2: Spatial Precision.
    Applies pure Q-surgery (polar extraction) to ONLY a specific 768x64 head.
    """
    W_v = get_wv(model, layer)
    W_v_new = W_v.copy()

    # Extract the specific 768x64 head slice
    head_start = target_head * D_HEAD
    head_end = (target_head + 1) * D_HEAD
    W_head = W_v[:, head_start:head_end]

    # Polar decomposition on the rectangular matrix
    # Returns Q (768x64 isometry) and S (64x64 symmetric stretch)
    Q, S = polar(W_head.astype(np.float64))
    Q = Q.astype(np.float32)

    # Preserve norm locally for the head
    norm_orig = np.linalg.norm(W_head, 'fro')
    norm_new = np.linalg.norm(Q, 'fro')
    Q_scaled = Q * (norm_orig / norm_new)

    # Inject the purely rotational head back into the full matrix
    W_v_new[:, head_start:head_end] = Q_scaled
    set_wv(model, layer, W_v_new)


# =============================================================================
# EVALUATION (Zero-Shot)
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = []
    for i in range(0, len(text), 100_000):
        chunk = tokenizer(text[i:i + 100_000], return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
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

    return {"wt2_ppl": wt2, "lambada_acc": lmda, "hellaswag_acc": hswag}


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_exp1_spectral_clipping(tokenizer, baseline, layer):
    print("\n" + "=" * 65)
    print(f"  EXPERIMENT 1 — Spectral Clipping (Soft Debiasing) on L{layer}")
    print(f"  Clipping top singular values to mathematically mute the local bias.")
    print("=" * 65)

    print(f"\n  {'Clip Count':<10} | {'WT2':>7} {'WT2Δ':>8} | {'LMDA':>6} {'LMDAΔ':>8} | {'HSwag':>6}")
    print("  " + "─" * 70)

    for clip_count in [1, 2, 3, 5, 10]:
        model = fresh_model()
        orig_s, clipped_s = apply_spectral_clipping(model, layer, clip_count=clip_count)

        r = evaluate(model, tokenizer, label=f"Clip_Top_{clip_count}")
        del model;
        gc.collect()

        d_wt2 = round((r["wt2_ppl"] - baseline["wt2_ppl"]) / baseline["wt2_ppl"] * 100, 1)
        d_lmda = round((r["lambada_acc"] - baseline["lambada_acc"]) / baseline["lambada_acc"] * 100, 1)

        print(
            f"  {clip_count:<10} | {r['wt2_ppl']:>7.2f} {d_wt2:>7.1f}% | {r['lambada_acc']:>5.2f}% {d_lmda:>7.1f}% | {r['hellaswag_acc']:>5.2f}%")

        if clip_count == 3:
            print("\n  [Diagnostic] Singular Values (Top 5):")
            print(f"    Original: {[round(val, 2) for val in orig_s]}")
            print(f"    Clipped:  {[round(val, 2) for val in clipped_s]}\n")


def run_exp2_head_isolation(tokenizer, baseline, layer):
    print("\n" + "=" * 65)
    print(f"  EXPERIMENT 2 — Head-Wise Isolation on W_v L{layer}")
    print(f"  Applying Q-surgery strictly to one head at a time.")
    print("=" * 65)

    print(f"\n  {'Head':<6} | {'WT2':>7} {'WT2Δ':>8} | {'LMDA':>6} {'LMDAΔ':>8} | {'HSwag':>6}")
    print("  " + "─" * 70)

    for head_idx in range(N_HEADS):
        model = fresh_model()
        apply_head_wise_q_surgery(model, layer, head_idx)

        r = evaluate(model, tokenizer, label=f"Head_{head_idx}_Q")
        del model;
        gc.collect()

        d_wt2 = round((r["wt2_ppl"] - baseline["wt2_ppl"]) / baseline["wt2_ppl"] * 100, 1)
        d_lmda = round((r["lambada_acc"] - baseline["lambada_acc"]) / baseline["lambada_acc"] * 100, 1)

        print(
            f"  H{head_idx:02d}   | {r['wt2_ppl']:>7.2f} {d_wt2:>7.1f}% | {r['lambada_acc']:>5.2f}% {d_lmda:>7.1f}% | {r['hellaswag_acc']:>5.2f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", type=int, default=[1, 2], help="Which experiments to run (1, 2)")
    p.add_argument("--target-layer", type=int, default=0, help="Which layer to target (default: 0)")
    args = p.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n  Measuring Baseline...")
    bm = fresh_model()
    baseline = evaluate(bm, tokenizer, label="baseline")
    del bm;
    gc.collect()

    if 1 in args.experiments:
        run_exp1_spectral_clipping(tokenizer, baseline, args.target_layer)

    if 2 in args.experiments:
        run_exp2_head_isolation(tokenizer, baseline, args.target_layer)


if __name__ == "__main__":
    main()