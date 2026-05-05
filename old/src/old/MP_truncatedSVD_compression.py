"""
compression_sweep.py
=====================================================================
Systematic compression sweep for W_v across all 12 layers.

Tests two strategies at multiple rank budgets:

  NAIVE SVD  — keep top-k singular values, discard rest.
               Standard low-rank approximation.
               W_v ≈ U[:, :k] @ diag(s[:k]) @ Vt[:k, :]

  MP-GUIDED  — keep all MP outliers (rank = n_outliers) PLUS
               the top-(k - n_outliers) bulk singular values.
               Guarantees all trained signal is preserved, then
               adds the most important bulk directions up to budget k.
               W_v ≈ U[:, idx[:k]] @ diag(s[idx[:k]]) @ Vt[idx[:k], :]
               where idx sorts by: outliers first, then top bulk.

Ranks tested: MP-outliers only, 150, 200, 250, 300, 350, 512, 768 (full)

For each rank × strategy:
  - Inject compressed W_v into all 12 layers simultaneously
  - Measure WT2 PPL and LAMBADA PPL + accuracy
  - Report parameter count and reduction %

Output: JSON log + printed comparison table.

Usage:
  python compression_sweep.py
  python compression_sweep.py --skip-lambada   # faster, WT2 only
"""

import argparse
import gc
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "gpt2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL    = 768
N_LAYERS   = 12
RESULTS_DIR = "compression_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Rank budgets to test (in addition to MP-outliers-only)
RANK_BUDGETS = [150, 200, 250, 300, 350, 512, 768]


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    return m.eval().to(DEVICE)


def get_wv(model, layer):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    return W[:, 2*D_MODEL:].copy()


def inject_wv(model, layer, W_v_new):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy().copy()
    W[:, 2*D_MODEL:] = W_v_new.astype(np.float32)
    model.transformer.h[layer].attn.c_attn.weight.data = (
        torch.tensor(W, dtype=torch.float32).to(DEVICE)
    )


def preserve_norm(W_new, W_orig):
    n_o = np.linalg.norm(W_orig, "fro")
    n_n = np.linalg.norm(W_new,  "fro")
    return (W_new * (n_o / n_n) if n_n > 1e-8 else W_new).astype(np.float32)


def param_count(k):
    """Parameters for a rank-k factorisation of a [768,768] matrix."""
    return int(2 * D_MODEL * k)


def param_reduction_pct(k):
    return round((1 - param_count(k) / (D_MODEL * D_MODEL)) * 100, 1)


# =============================================================================
# SVD HELPERS
# =============================================================================

def mp_boundary(S):
    """Marchenko-Pastur upper edge for a square [768,768] matrix."""
    sigma_hat = np.sqrt(np.median(S**2) / D_MODEL)
    return float(2.0 * sigma_hat * np.sqrt(D_MODEL))


def naive_svd_approx(W_v, k):
    """Keep top-k singular values. Standard low-rank SVD."""
    U, S, Vt = np.linalg.svd(W_v.astype(np.float64), full_matrices=False)
    k = min(k, len(S))
    W_approx = ((U[:, :k] * S[:k]) @ Vt[:k, :]).astype(np.float32)
    return preserve_norm(W_approx, W_v)


def mp_guided_approx(W_v, k):
    """
    MP-guided low-rank approximation at budget k.

    Priority order for singular value selection:
      1. All MP outliers (above bulk boundary) — always included first.
      2. Top bulk singular values (by magnitude) — fill remaining budget.

    This guarantees all trained signal is preserved, then supplements
    with the most informative bulk directions up to the rank budget.
    """
    U, S, Vt = np.linalg.svd(W_v.astype(np.float64), full_matrices=False)
    upper    = mp_boundary(S)
    n        = len(S)

    # Split into outliers and bulk
    outlier_mask = S > upper
    bulk_mask    = ~outlier_mask
    n_outliers   = int(np.sum(outlier_mask))

    k = min(k, n)

    if k <= n_outliers:
        # Budget smaller than outlier count — just use top-k outliers
        idx = np.where(outlier_mask)[0][:k]
    else:
        # All outliers + top-(k - n_outliers) bulk values
        outlier_idx = np.where(outlier_mask)[0]          # already sorted desc
        bulk_idx    = np.where(bulk_mask)[0]             # also sorted desc
        n_bulk      = k - n_outliers
        idx         = np.concatenate([outlier_idx, bulk_idx[:n_bulk]])
        idx         = np.sort(idx)                       # maintain SV order

    W_approx = ((U[:, idx] * S[idx]) @ Vt[idx, :]).astype(np.float32)
    return preserve_norm(W_approx, W_v), n_outliers


# =============================================================================
# EVALUATION
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids  = []
    for i in range(0, len(text), 100_000):
        ids.append(tokenizer(
            text[i:i+100_000],
            return_tensors="pt", truncation=False, add_special_tokens=False,
        )["input_ids"].squeeze(0))
    enc = torch.cat(ids, dim=0)
    stride, max_len, nlls = 512, 1024, []
    model.eval()
    for b in range(0, enc.size(0) - max_len, stride):
        inp = enc[b:b+max_len].unsqueeze(0).to(DEVICE)
        tgt = inp.clone(); tgt[:, :stride] = -100
        with torch.no_grad():
            nlls.append(model(input_ids=inp, labels=tgt).loss.item()
                        * (max_len - stride))
    return round(math.exp(sum(nlls) / (len(nlls) * (max_len - stride))), 4)


def measure_lambada(model, tokenizer):
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except Exception:
        ds = load_dataset("lambada", split="test")
    nlls, correct, total = [], 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  LAMBADA", leave=False):
        text   = ex.get("text") or ex.get("passage", "")
        tokens = tokenizer(text, return_tensors="pt",
                           max_length=1024, truncation=True
                           )["input_ids"].squeeze(0)
        if tokens.shape[0] < 2:
            continue
        inp = tokens.unsqueeze(0).to(DEVICE)
        tgt = inp.clone(); tgt[:, :-1] = -100
        with torch.no_grad():
            loss   = model(input_ids=inp, labels=tgt).loss
            logits = model(input_ids=inp).logits
        nlls.append(loss.item())
        if logits[0, -2, :].argmax().item() == tokens[-1].item():
            correct += 1
        total += 1
    if total == 0:
        return None, None
    return round(math.exp(sum(nlls)/len(nlls)), 4), round(correct/total*100, 2)


def run_eval(model, tokenizer, skip_lambada=False):
    wt2 = measure_wt2_ppl(model, tokenizer)
    if skip_lambada:
        return wt2, None, None
    lmda, acc = measure_lambada(model, tokenizer)
    return wt2, lmda, acc


# =============================================================================
# PRE-COMPUTE ALL SVD DECOMPOSITIONS
# =============================================================================

def precompute_svds():
    """Load model once, extract and decompose all W_v matrices."""
    print("  Pre-computing SVDs for all 12 layers...")
    ref   = fresh_model()
    svds  = {}
    for layer in range(N_LAYERS):
        W_v = get_wv(ref, layer)
        U, S, Vt = np.linalg.svd(W_v.astype(np.float64), full_matrices=False)
        svds[layer] = {
            "W_v_orig": W_v,
            "U": U, "S": S, "Vt": Vt,
            "mp_upper": mp_boundary(S),
            "n_outliers": int(np.sum(S > mp_boundary(S))),
        }
        print(f"    L{layer:02d}: MP upper={svds[layer]['mp_upper']:.4f}  "
              f"outliers={svds[layer]['n_outliers']}")
    del ref; gc.collect()
    return svds


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_sweep(skip_lambada=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'='*70}")
    print(f"  W_v Compression Sweep — {MODEL_NAME}   {ts}")
    print(f"  Strategies: NAIVE SVD  |  MP-GUIDED SVD")
    print(f"  Ranks: MP-only + {RANK_BUDGETS}")
    print(f"  All 12 layers compressed simultaneously")
    print(f"{'='*70}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n  Measuring baseline...")
    bm = fresh_model()
    wt2_base, lmda_base, acc_base = run_eval(bm, tokenizer, skip_lambada)
    del bm; gc.collect()
    print(f"  Baseline: WT2={wt2_base}  LMDA={lmda_base}  Acc={acc_base}%")

    results = {
        "meta": {
            "model": MODEL_NAME, "timestamp": ts,
            "device": DEVICE, "ranks_tested": RANK_BUDGETS,
        },
        "baseline": {
            "wt2_ppl": wt2_base,
            "lambada_ppl": lmda_base,
            "lambada_acc": acc_base,
            "params_per_layer": D_MODEL * D_MODEL,
            "total_wv_params": N_LAYERS * D_MODEL * D_MODEL,
        },
        "experiments": [],
    }

    # ── Pre-compute SVDs ──────────────────────────────────────────────────
    svds = precompute_svds()

    # Layer-wise outlier counts for the table
    outlier_counts = {l: svds[l]["n_outliers"] for l in range(N_LAYERS)}
    mean_outliers  = float(np.mean(list(outlier_counts.values())))

    # ── Build experiment list ─────────────────────────────────────────────
    # Each experiment: (strategy_name, rank_per_layer_fn)
    # rank_per_layer_fn(layer) returns effective k used for that layer

    experiments = []

    # MP-only: use exactly n_outliers per layer (variable rank)
    experiments.append({
        "name":     "mp_only",
        "label":    f"MP-guided (outliers only, mean k≈{mean_outliers:.0f})",
        "strategy": "mp_guided",
        "fixed_k":  None,   # variable per layer
    })

    # Fixed ranks: both naive and MP-guided
    for k in RANK_BUDGETS:
        experiments.append({
            "name":     f"naive_k{k}",
            "label":    f"Naive SVD  k={k}",
            "strategy": "naive",
            "fixed_k":  k,
        })
        experiments.append({
            "name":     f"mp_k{k}",
            "label":    f"MP-guided  k={k}",
            "strategy": "mp_guided",
            "fixed_k":  k,
        })

    # ── Run experiments ───────────────────────────────────────────────────
    for exp in experiments:
        name     = exp["name"]
        label    = exp["label"]
        strategy = exp["strategy"]
        fixed_k  = exp["fixed_k"]

        print(f"\n{'─'*70}")
        print(f"  {label}")

        model = fresh_model()
        layer_ks      = []
        layer_outliers = []

        for layer in range(N_LAYERS):
            W_v_orig  = svds[layer]["W_v_orig"]
            n_out     = svds[layer]["n_outliers"]
            layer_outliers.append(n_out)

            k_use = fixed_k if fixed_k is not None else n_out

            if strategy == "naive":
                W_v_new = naive_svd_approx(W_v_orig, k_use)
                k_actual = k_use
            else:  # mp_guided
                W_v_new, _ = mp_guided_approx(W_v_orig, k_use)
                # Actual rank used = min(k_use, 768); effective = k_use
                k_actual = k_use

            layer_ks.append(k_actual)
            inject_wv(model, layer, W_v_new)

        # Parameter accounting (mean k across layers)
        mean_k      = float(np.mean(layer_ks))
        total_orig  = N_LAYERS * D_MODEL * D_MODEL
        total_new   = int(sum(param_count(k) for k in layer_ks))
        reduction   = round((1 - total_new / total_orig) * 100, 1)

        print(f"  Params: {total_orig:,} → {total_new:,}  "
              f"(−{reduction}%)  mean_k={mean_k:.1f}")

        wt2, lmda, acc = run_eval(model, tokenizer, skip_lambada)
        del model; gc.collect()

        d_wt2  = round((wt2  - wt2_base)  / wt2_base  * 100, 2) if wt2_base  else None
        d_lmda = round((lmda - lmda_base) / lmda_base * 100, 2) \
                 if (lmda and lmda_base) else None

        print(f"  WT2={wt2} ({d_wt2:+.1f}%)  "
              f"LMDA={lmda} ({d_lmda:+.1f}% if available)  Acc={acc}%")

        results["experiments"].append({
            "name":         name,
            "label":        label,
            "strategy":     strategy,
            "fixed_k":      fixed_k,
            "mean_k":       mean_k,
            "layer_ks":     layer_ks,
            "total_params_orig": total_orig,
            "total_params_new":  total_new,
            "reduction_pct":     reduction,
            "wt2_ppl":      wt2,  "wt2_delta_pct":      d_wt2,
            "lambada_ppl":  lmda, "lambada_delta_pct":   d_lmda,
            "lambada_acc":  acc,
        })

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  COMPRESSION SWEEP — SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"  {'Experiment':<30} {'mean_k':>7} {'Params':>10} {'Reduc%':>7} "
          f"{'WT2':>8} {'WT2Δ%':>7} {'LMDA':>9} {'LMDAδ%':>8} {'Acc':>7}")
    print(f"  {'─'*88}")

    # Baseline row
    print(f"  {'baseline':<30} {'768':>7} "
          f"{N_LAYERS*D_MODEL*D_MODEL:>10,} {'0.0%':>7} "
          f"{wt2_base:>8.2f} {'±0.0%':>7} "
          f"{str(lmda_base):>9} {'±0.0%':>8} {str(acc_base):>7}")

    for r in results["experiments"]:
        lmda_str = f"{r['lambada_ppl']:.2f}" if r["lambada_ppl"] else "—"
        acc_str  = f"{r['lambada_acc']:.1f}%" if r["lambada_acc"] else "—"
        d_lmda_s = f"{r['lambada_delta_pct']:+.1f}%" \
                   if r["lambada_delta_pct"] is not None else "—"
        print(f"  {r['label']:<30} {r['mean_k']:>7.0f} "
              f"{r['total_params_new']:>10,} {r['reduction_pct']:>6.1f}% "
              f"{r['wt2_ppl']:>8.2f} {r['wt2_delta_pct']:>+7.1f}% "
              f"{lmda_str:>9} {d_lmda_s:>8} {acc_str:>7}")

    # ── Naive vs MP-guided comparison ─────────────────────────────────────
    print(f"\n  MP-GUIDED vs NAIVE ADVANTAGE (at same rank budget):")
    print(f"  {'Rank':>6}  {'ΔLMDA (MP better by)':>22}  {'ΔWT2 (MP better by)':>22}")
    print(f"  {'─'*56}")

    exp_by_k = {}
    for r in results["experiments"]:
        k = r["fixed_k"]
        if k is not None:
            exp_by_k.setdefault(k, {})[r["strategy"]] = r

    for k in RANK_BUDGETS:
        if k not in exp_by_k:
            continue
        naive = exp_by_k[k].get("naive")
        mp    = exp_by_k[k].get("mp_guided")
        if naive and mp:
            d_lmda_adv = None
            if naive["lambada_delta_pct"] and mp["lambada_delta_pct"]:
                d_lmda_adv = naive["lambada_delta_pct"] - mp["lambada_delta_pct"]
            d_wt2_adv = (naive["wt2_delta_pct"] or 0) - (mp["wt2_delta_pct"] or 0)
            lmda_adv_s = f"{d_lmda_adv:+.2f}pp" if d_lmda_adv is not None else "—"
            print(f"  k={k:<5}  {lmda_adv_s:>22}  {d_wt2_adv:>+22.2f}pp")

    # ── Parameter efficiency table ─────────────────────────────────────────
    print(f"\n  PARAMETER EFFICIENCY (per layer, mean across strategies):")
    print(f"  {'Rank':>6} {'Params/layer':>14} {'Total Wv params':>18} {'Reduction':>10}")
    print(f"  {'─'*52}")
    all_ks = [svds[l]["n_outliers"] for l in range(N_LAYERS)]
    mean_mp_k = np.mean(all_ks)
    for k_show in [int(mean_mp_k)] + RANK_BUDGETS:
        p   = param_count(k_show)
        tot = N_LAYERS * p
        red = param_reduction_pct(k_show)
        print(f"  {k_show:>6d} {p:>14,} {tot:>18,} {red:>9.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"compression_sweep_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print("  Done.")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="W_v compression sweep: naive SVD vs MP-guided SVD"
    )
    p.add_argument("--skip-lambada", action="store_true",
                   help="Skip LAMBADA evaluation (much faster, WT2 only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(skip_lambada=args.skip_lambada)