"""
geometric_analysis_v2.py
===================================================
Complete geometric characterisation of GPT-2 attention weights.
Covers all four matrices: W_q, W_k, W_v, W_o and the OV circuit.

SECTIONS
--------
A  Random+Procrustes W_v control
     Is PCA special, or does any orthogonal matrix Procrustes-aligned
     to W_v give the same result? Expected: rand_proc == polar_Q.

B  Phase 0 on W_o (output projection)
     Apply the full Phase 0 battery to W_o across all 12 layers.
     Staats et al. predict W_o will be near-random (lazy matrix).

C  OV circuit full geometric analysis
     Per head per layer: rank, effective rank, orthogonality deviation,
     MP outliers, energy_cv of W_v_h @ W_o_h.
     Also compares OV_h geometry to W_v_h alone — does W_o correct
     or distort W_v's approximately isometric structure?

D  Low-rank compression curve for W_v L0
     rank-k SVD approximation vs WT2 PPL and LAMBADA PPL.
     Also runs MP-guided pruning: remove bulk singular values, keep
     outliers — the compression strategy motivated by Staats et al.

E  Per-head analysis W_v, W_q, W_k
     Per-head orth_dev, eff_rank, MP outliers for each matrix at
     every layer. Tests whether heads are geometrically diverse or
     redundant.

F  Principal angles W_q vs W_k column spaces
     Quantifies geometric asymmetry of the QK circuit. Small angles
     = symmetric routing. Large angles = strongly directional routing.

G  Multi-layer rotation surgery
     Replace W_v with polar Q simultaneously across multiple layer
     combinations. Tests cumulative effect beyond single-layer result.

H  Effective rank W_q, W_k individually vs G = W_q W_k.T
     If eff_rank(G) << min(eff_rank(W_q), eff_rank(W_k)), the
     bottleneck is in their interaction, not in either matrix alone.
     Identifies where QK compression is most feasible.

I  W_v @ W_o combined operator
     Effective rank and MP outliers of the combined V-O operator
     [768x768 rank<=64] across layers.

Usage:
  python geometric_analysis_v2.py                        # all sections
  python geometric_analysis_v2.py --sections B C E F H I  # analysis only
  python geometric_analysis_v2.py --sections A D G        # eval sections
  python geometric_analysis_v2.py --skip-eval            # skip PPL entirely
"""

import argparse
import gc
import json
import math
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import orthogonal_procrustes, polar, subspace_angles
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME  = "gpt2"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS    = 12
N_HEADS     = 12
D_MODEL     = 768
HEAD_DIM    = 64        # D_MODEL // N_HEADS
RESULTS_DIR = "results/extra_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    return m.eval().to(DEVICE)


def get_qkv(model, layer):
    """W_q, W_k, W_v each [768,768]. Sliced from fused c_attn weight."""
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    d = D_MODEL
    return W[:, :d].copy(), W[:, d:2*d].copy(), W[:, 2*d:].copy()


def get_wo(model, layer):
    """W_o output projection [768,768] from c_proj."""
    return model.transformer.h[layer].attn.c_proj.weight.data.cpu().numpy().copy()


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


def random_orthonormal(d, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float64))
    return Q.astype(np.float32)


def procrustes_rowspace(W_stat, W_orig):
    R, _ = orthogonal_procrustes(
        W_stat.astype(np.float64).T,
        W_orig.astype(np.float64).T
    )
    return (R.T @ W_stat.astype(np.float64)).astype(np.float32)


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


# =============================================================================
# GEOMETRIC METRICS
# =============================================================================

def orth_dev(M):
    """||M.T M - I||_F"""
    M64 = M.astype(np.float64)
    return float(np.linalg.norm(M64.T @ M64 - np.eye(M64.shape[1]), "fro"))


def shannon_eff_rank(M):
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-15))))


def var90_rank(M):
    s   = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(cum, 0.90)) + 1


def mp_outliers(M):
    """Count SV outliers above the MP upper edge (square matrix version)."""
    s         = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    n         = M.shape[0]
    sigma_hat = np.sqrt(np.median(s**2) / n)
    upper_sv  = 2.0 * sigma_hat * np.sqrt(n)
    n_out     = int(np.sum(s > upper_sv))
    mass      = float(np.sum(s[s > upper_sv]**2) / (np.sum(s**2) + 1e-10))
    return n_out, mass


def polar_q_fidelity(M):
    """Q_fidelity: residual ||M - Q@S||_F / ||M||_F — lower = more rotation-like."""
    Q, S = polar(M.astype(np.float64))
    res  = float(np.linalg.norm(M.astype(np.float64) - Q @ S, "fro"))
    return float(res / (np.linalg.norm(M.astype(np.float64), "fro") + 1e-10))


def energy_cv(M):
    """Coefficient of variation of squared singular values."""
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    return float(np.std(s**2) / (np.mean(s**2) + 1e-10))


def phase0_battery(M, label=""):
    """Full Phase 0 metric battery on matrix M."""
    s64      = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    n_out, mass = mp_outliers(M)
    return {
        "orth_dev":    orth_dev(M),
        "eff_rank":    shannon_eff_rank(M),
        "var90":       var90_rank(M),
        "mp_outliers": n_out,
        "outlier_mass": mass,
        "q_fidelity":  polar_q_fidelity(M),
        "energy_cv":   energy_cv(M),
        "sv_max":      float(s64[0]),
        "sv_min":      float(s64[-1]),
        "tightness":   float(s64[-1] / s64[0]) if s64[0] > 0 else 0.0,
    }


# =============================================================================
# SECTION A — Random+Procrustes W_v control
# =============================================================================

def section_A(tokenizer, wt2_base, lmda_base):
    print("\n" + "="*65)
    print("  SECTION A — Random+Procrustes W_v control")
    print("  Question: is PCA special, or does any orthogonal matrix")
    print("  Procrustes-aligned to W_v give the same result?")
    print("  Prediction: rand_proc converges to polar_Q exactly.")
    print("="*65)

    ref = fresh_model()
    _, _, W_v_orig = get_qkv(ref, 0)
    del ref; gc.collect()

    Q_polar, _ = polar(W_v_orig.astype(np.float64))
    Q_polar    = Q_polar.astype(np.float32)

    Q_rand        = random_orthonormal(D_MODEL, seed=42)
    W_v_rand_proc = procrustes_rowspace(Q_rand, W_v_orig)
    W_v_rand_proc = preserve_norm(W_v_rand_proc, W_v_orig)

    dist_to_Q = float(np.linalg.norm(
        W_v_rand_proc.astype(np.float64) - Q_polar.astype(np.float64), "fro"
    ))
    print(f"\n  ||rand_proc - polar_Q||_F = {dist_to_Q:.6f}")
    print(f"  (0.000000 = rand_proc converged exactly to Q)")

    model = fresh_model()
    inject_wv(model, 0, W_v_rand_proc)
    wt2          = measure_wt2_ppl(model, tokenizer)
    lmda, acc    = measure_lambada(model, tokenizer)
    del model; gc.collect()

    d_wt2  = round((wt2  - wt2_base)  / wt2_base  * 100, 2)
    d_lmda = round((lmda - lmda_base) / lmda_base * 100, 2)

    print(f"\n  wv_random_proc:  WT2={wt2} ({d_wt2:+.1f}%)  "
          f"LMDA={lmda} ({d_lmda:+.1f}%)  Acc={acc}%")
    print(f"\n  Prior results for comparison:")
    print(f"    wv_polar_q:  WT2 +25.9%  LMDA -12.8%  Acc 34.8%")
    print(f"    wv_pca_proc: WT2 +25.9%  LMDA -12.8%  Acc 34.8%")
    print(f"\n  Interpretation:")
    if dist_to_Q < 0.01:
        print(f"  Procrustes drove rand_proc to Q. PCA is NOT special.")
        print(f"  Finding: W_v is functionally equivalent to its own polar Q.")
        print(f"  The isometry is the finding, not the data-derived PCA.")
    else:
        print(f"  rand_proc differs from Q by {dist_to_Q:.4f}. PCA adds value.")

    return {
        "dist_rand_proc_to_polar_Q": dist_to_Q,
        "wt2_ppl": wt2, "wt2_delta_pct": d_wt2,
        "lambada_ppl": lmda, "lambada_delta_pct": d_lmda,
        "lambada_acc": acc,
    }


# =============================================================================
# SECTION B — Phase 0 on W_o
# =============================================================================

def section_B():
    print("\n" + "="*65)
    print("  SECTION B — Phase 0 geometric analysis on W_o")
    print("  Staats et al. predict W_o is 'lazy' — near-random,")
    print("  with minimal overlap with activation covariance.")
    print("="*65)

    model   = fresh_model()
    results = {}

    print(f"\n  {'Lyr':<4} {'orth_dev':>9} {'eff_rank':>9} {'var90':>6} "
          f"{'mp_out':>7} {'out_mass':>9} {'q_fid':>7} {'tight':>8}")
    print("  " + "─"*64)

    for layer in range(N_LAYERS):
        W_o = get_wo(model, layer)
        m   = phase0_battery(W_o)
        print(f"  L{layer:02d}  "
              f"{m['orth_dev']:>9.2f} {m['eff_rank']:>9.2f} {m['var90']:>6d} "
              f"{m['mp_outliers']:>7d} {m['outlier_mass']:>9.4f} "
              f"{m['q_fidelity']:>7.4f} {m['tightness']:>8.6f}")
        results[f"L{layer:02d}"] = m

    del model; gc.collect()

    vals = list(results.values())
    print(f"\n  W_o means across layers:")
    for k in ["orth_dev", "eff_rank", "mp_outliers", "q_fidelity", "energy_cv"]:
        print(f"    {k:<14} = {np.mean([v[k] for v in vals]):.4f}")
    print(f"\n  W_v reference (from Phase 0):")
    print(f"    orth_dev≈476  eff_rank≈329  mp_out≈3.1  q_fid≈0.647")
    print(f"  If W_o shows orth_dev≈random(30091) and mp_out≈0:")
    print(f"  → Confirms Staats et al. laziness prediction for W_o.")

    return results


# =============================================================================
# SECTION C — OV circuit full geometric analysis
# =============================================================================

def section_C():
    print("\n" + "="*65)
    print("  SECTION C — OV circuit full geometric analysis")
    print("  OV_h = W_v_h [768×64] @ W_o_h [64×768] = [768×768] rank≤64")
    print("  Measures: rank, eff_rank, orth_dev, MP outliers, energy_cv")
    print("  Compares OV_h geometry vs W_v_h alone.")
    print("  Key question: does W_o correct or distort W_v's isometry?")
    print("="*65)

    model   = fresh_model()
    results = {}

    # ── Rank table ────────────────────────────────────────────────────────
    print(f"\n  Rank per head (budget = {HEAD_DIM}):")
    print(f"  {'Lyr':<4} {'mean':>6} {'min':>5} {'max':>5} {'full%':>7}  ranks")
    print("  " + "─"*70)

    for layer in range(N_LAYERS):
        _, _, W_v = get_qkv(model, layer)
        W_o       = get_wo(model, layer)
        head_data = []

        for h in range(N_HEADS):
            sl    = slice(h * HEAD_DIM, (h+1) * HEAD_DIM)
            W_v_h = W_v[:, sl].astype(np.float64)   # [768, 64]
            W_o_h = W_o[sl, :].astype(np.float64)   # [64, 768]
            OV_h  = W_v_h @ W_o_h                   # [768, 768] rank ≤ 64

            # ── Rank ──────────────────────────────────────────────────────
            r     = int(np.linalg.matrix_rank(OV_h, tol=1e-3))

            # ── Singular values ───────────────────────────────────────────
            sv_ov    = np.linalg.svd(OV_h, compute_uv=False)
            sv_nz    = sv_ov[sv_ov > 1e-10]

            # ── Effective rank (over nonzero SVs) ─────────────────────────
            p_nz  = sv_nz / sv_nz.sum()
            er_ov = float(np.exp(-np.sum(p_nz * np.log(p_nz + 1e-15))))

            # ── Orth dev of column subspace ───────────────────────────────
            U, _, _ = np.linalg.svd(OV_h, full_matrices=False)
            U_r     = U[:, :r]
            od_ov   = float(np.linalg.norm(U_r.T @ U_r - np.eye(r), "fro"))

            # ── Energy CV ─────────────────────────────────────────────────
            ev_cv = float(np.std(sv_nz**2) / (np.mean(sv_nz**2) + 1e-10))

            # ── MP outliers (scaled for rank-r) ───────────────────────────
            sigma_h  = np.sqrt(np.median(sv_nz**2 / r)) if len(sv_nz) else 0
            upper_sv = 2.0 * sigma_h * np.sqrt(r)
            n_out    = int(np.sum(sv_nz > upper_sv))

            # ── W_v_h alone for comparison ────────────────────────────────
            sv_wv  = np.linalg.svd(W_v_h, compute_uv=False)
            p_wv   = sv_wv / sv_wv.sum()
            er_wv  = float(np.exp(-np.sum(p_wv * np.log(p_wv + 1e-15))))
            od_wv  = float(np.linalg.norm(W_v_h.T @ W_v_h
                                           - np.eye(HEAD_DIM), "fro"))

            head_data.append({
                "head": h,
                "ov_rank":     r,
                "ov_eff_rank": er_ov,
                "ov_orth_dev": od_ov,
                "ov_energy_cv":ev_cv,
                "ov_mp_out":   n_out,
                "wv_eff_rank": er_wv,
                "wv_orth_dev": od_wv,
                "wo_corrects": od_ov < od_wv,  # True = W_o makes circuit more isometric
            })

        ranks     = [x["ov_rank"] for x in head_data]
        mean_r    = float(np.mean(ranks))
        frac_full = float(np.mean([r == HEAD_DIM for r in ranks])) * 100
        print(f"  L{layer:02d}  {mean_r:>6.1f} {min(ranks):>5d} {max(ranks):>5d} "
              f"{frac_full:>6.0f}%  {ranks}")
        results[f"L{layer:02d}"] = {"per_head": head_data}

    # ── Geometric comparison table ────────────────────────────────────────
    print(f"\n  OV circuit geometry vs W_v alone (mean across heads):")
    print(f"  {'Lyr':<4} {'er_OV':>8} {'od_OV':>8} {'ev_cv_OV':>9} "
          f"{'mp_OV':>6}  {'er_Wv':>8} {'od_Wv':>8}  W_o effect")
    print("  " + "─"*78)

    for layer in range(N_LAYERS):
        hd    = results[f"L{layer:02d}"]["per_head"]
        er_ov = float(np.mean([x["ov_eff_rank"]  for x in hd]))
        od_ov = float(np.mean([x["ov_orth_dev"]  for x in hd]))
        ev_cv = float(np.mean([x["ov_energy_cv"] for x in hd]))
        mp_ov = float(np.mean([x["ov_mp_out"]    for x in hd]))
        er_wv = float(np.mean([x["wv_eff_rank"]  for x in hd]))
        od_wv = float(np.mean([x["wv_orth_dev"]  for x in hd]))
        frac  = float(np.mean([x["wo_corrects"]  for x in hd]))
        effect = f"corrects→isometric ({frac*100:.0f}% heads)" \
                 if frac > 0.5 else f"adds stretch ({(1-frac)*100:.0f}% heads)"

        print(f"  L{layer:02d}  {er_ov:>8.2f} {od_ov:>8.2f} {ev_cv:>9.4f} "
              f"{mp_ov:>6.1f}  {er_wv:>8.2f} {od_wv:>8.2f}  {effect}")

        results[f"L{layer:02d}"]["layer_summary"] = {
            "mean_ov_eff_rank": er_ov, "mean_ov_orth_dev": od_ov,
            "mean_ov_energy_cv": ev_cv, "mean_ov_mp_out": mp_ov,
            "mean_wv_eff_rank": er_wv, "mean_wv_orth_dev": od_wv,
            "frac_heads_wo_corrects": frac,
        }

    del model; gc.collect()
    return results


# =============================================================================
# SECTION D — Low-rank compression curve for W_v L0
# =============================================================================

def section_D(tokenizer, wt2_base, lmda_base):
    print("\n" + "="*65)
    print("  SECTION D — W_v L0 compression curve")
    print("  Part 1: rank-k SVD approximation (naive low-rank)")
    print("  Part 2: MP-guided pruning (keep outliers, remove bulk)")
    print("  Staats et al.: small SV outliers matter; bulk does not.")
    print("="*65)

    ref = fresh_model()
    _, _, W_v_orig = get_qkv(ref, 0)
    del ref; gc.collect()

    U, s, Vt     = np.linalg.svd(W_v_orig.astype(np.float64), full_matrices=True)
    var_total    = float(np.sum(s**2))

    # Identify MP boundary for this matrix
    n         = D_MODEL
    sigma_hat = np.sqrt(np.median(s**2) / n)
    upper_sv  = 2.0 * sigma_hat * np.sqrt(n)
    outlier_mask = s > upper_sv
    n_outliers   = int(np.sum(outlier_mask))
    print(f"\n  MP upper edge: {upper_sv:.4f}  |  "
          f"Outliers above bulk: {n_outliers} / {D_MODEL}")

    results = {}

    # ── Part 1: naive rank-k ──────────────────────────────────────────────
    print(f"\n  Part 1 — Naive rank-k SVD approximation:")
    print(f"  {'Rank':>6} {'var%':>7} {'WT2':>8} {'WT2Δ%':>7} "
          f"{'LMDA':>9} {'LMDAδ%':>8} {'Acc':>7}")
    print("  " + "─"*60)

    ranks = [16, 32, 64, 96, 128, 192, 256, 291, 384, 512, 640, 768]
    for k in ranks:
        var_exp = float(np.sum(s[:k]**2) / var_total * 100)
        W_v_k   = ((U[:, :k] * s[:k]) @ Vt[:k, :]).astype(np.float32)
        W_v_k   = preserve_norm(W_v_k, W_v_orig)

        model = fresh_model()
        inject_wv(model, 0, W_v_k)
        wt2        = measure_wt2_ppl(model, tokenizer)
        lmda, acc  = measure_lambada(model, tokenizer)
        del model; gc.collect()

        d_wt2  = round((wt2  - wt2_base)  / wt2_base  * 100, 1)
        d_lmda = round((lmda - lmda_base) / lmda_base * 100, 1)
        marker = " ← Var90" if k == 291 else ""
        print(f"  {k:>6d} {var_exp:>6.1f}% {wt2:>8.2f} {d_wt2:>+7.1f}% "
              f"{lmda:>9.2f} {d_lmda:>+8.1f}% {acc:>6.1f}%{marker}")

        results[f"rank_{k}"] = {
            "rank": k, "var_explained_pct": var_exp,
            "wt2_ppl": wt2, "wt2_delta_pct": d_wt2,
            "lambada_ppl": lmda, "lambada_delta_pct": d_lmda,
            "lambada_acc": acc,
        }

    # ── Part 2: MP-guided pruning ─────────────────────────────────────────
    print(f"\n  Part 2 — MP-guided pruning (keep outliers, zero bulk):")
    print(f"  Strategy: keep the {n_outliers} outlier SVs, zero the MP bulk.")
    print(f"  Motivated by Staats et al.: bulk SVs carry little information.")

    s_pruned = s.copy()
    s_pruned[~outlier_mask] = 0.0  # zero bulk, keep outliers
    W_v_mp = ((U * s_pruned) @ Vt).astype(np.float32)
    W_v_mp = preserve_norm(W_v_mp, W_v_orig)

    model = fresh_model()
    inject_wv(model, 0, W_v_mp)
    wt2_mp        = measure_wt2_ppl(model, tokenizer)
    lmda_mp, acc_mp = measure_lambada(model, tokenizer)
    del model; gc.collect()

    d_wt2_mp  = round((wt2_mp  - wt2_base)  / wt2_base  * 100, 1)
    d_lmda_mp = round((lmda_mp - lmda_base) / lmda_base * 100, 1)

    print(f"\n  MP-guided (keep {n_outliers} outliers):")
    print(f"    WT2={wt2_mp} ({d_wt2_mp:+.1f}%)  "
          f"LMDA={lmda_mp} ({d_lmda_mp:+.1f}%)  Acc={acc_mp}%")
    print(f"\n  If MP-guided << naive rank-{n_outliers}:")
    print(f"  → Bulk SVs carry function; outlier-only is not sufficient.")
    print(f"  If MP-guided ≈ naive rank-{n_outliers}:")
    print(f"  → Outliers contain all the trained signal; bulk is noise.")

    results["mp_guided"] = {
        "n_outliers_kept": n_outliers,
        "mp_upper_edge": upper_sv,
        "wt2_ppl": wt2_mp, "wt2_delta_pct": d_wt2_mp,
        "lambada_ppl": lmda_mp, "lambada_delta_pct": d_lmda_mp,
        "lambada_acc": acc_mp,
    }

    return results


# =============================================================================
# SECTION E — Per-head analysis W_v, W_q, W_k
# =============================================================================

def section_E():
    print("\n" + "="*65)
    print("  SECTION E — Per-head analysis W_v, W_q, W_k")
    print("  Tests whether heads are geometrically diverse or redundant.")
    print("  Each head slice: [768, 64].")
    print("="*65)

    model   = fresh_model()
    results = {}

    for layer in range(N_LAYERS):
        W_q, W_k, W_v = get_qkv(model, layer)
        layer_res = {}

        for mat_name, W in [("W_v", W_v), ("W_q", W_q), ("W_k", W_k)]:
            head_data = []
            for h in range(N_HEADS):
                sl  = slice(h * HEAD_DIM, (h+1) * HEAD_DIM)
                W_h = W[:, sl]   # [768, 64]
                od  = orth_dev(W_h)
                er  = shannon_eff_rank(W_h)
                no, mass = mp_outliers(W_h)
                head_data.append({
                    "head": h, "orth_dev": od,
                    "eff_rank": er, "mp_outliers": no, "outlier_mass": mass,
                })

            ods  = [x["orth_dev"]    for x in head_data]
            ers  = [x["eff_rank"]    for x in head_data]
            outs = [x["mp_outliers"] for x in head_data]
            print(f"  L{layer:02d} {mat_name}: "
                  f"orth_dev μ={np.mean(ods):.1f} σ={np.std(ods):.1f}  "
                  f"eff_rank μ={np.mean(ers):.2f}  "
                  f"mp_out μ={np.mean(outs):.1f} σ={np.std(outs):.1f}  "
                  f"head_range=[{min(outs)},{max(outs)}]")
            layer_res[mat_name] = head_data

        results[f"L{layer:02d}"] = layer_res

    del model; gc.collect()
    return results


# =============================================================================
# SECTION F — Principal angles W_q vs W_k
# =============================================================================

def section_F():
    print("\n" + "="*65)
    print("  SECTION F — Principal angles between W_q and W_k subspaces")
    print("  0° = identical subspaces (symmetric routing)")
    print("  90° = orthogonal subspaces (fully asymmetric routing)")
    print("  Characterises what the W_q≠W_k asymmetry means geometrically.")
    print("="*65)

    model   = fresh_model()
    results = {}

    print(f"\n  {'Lyr':<4} {'mean°':>7} {'min°':>7} {'max°':>7} "
          f"{'<30°%':>7} {'<60°%':>7}  interpretation")
    print("  " + "─"*68)

    for layer in range(N_LAYERS):
        W_q, W_k, _ = get_qkv(model, layer)
        Qq, _  = np.linalg.qr(W_q.astype(np.float64))
        Qk, _  = np.linalg.qr(W_k.astype(np.float64))

        # Principal angles between top HEAD_DIM directions
        angles_rad = subspace_angles(Qq[:, :HEAD_DIM], Qk[:, :HEAD_DIM])
        angles_deg = np.degrees(angles_rad)

        mean_a  = float(np.mean(angles_deg))
        min_a   = float(np.min(angles_deg))
        max_a   = float(np.max(angles_deg))
        frac_30 = float(np.mean(angles_deg < 30.0)) * 100
        frac_60 = float(np.mean(angles_deg < 60.0)) * 100

        interp = ("near-symmetric" if mean_a < 30
                  else "moderately asymmetric" if mean_a < 60
                  else "strongly asymmetric")

        print(f"  L{layer:02d}  {mean_a:>7.2f} {min_a:>7.2f} {max_a:>7.2f} "
              f"{frac_30:>6.0f}% {frac_60:>6.0f}%  {interp}")

        results[f"L{layer:02d}"] = {
            "mean_deg": mean_a, "min_deg": min_a, "max_deg": max_a,
            "frac_below_30": frac_30, "frac_below_60": frac_60,
            "angles_deg": angles_deg.tolist(),
        }

    print(f"\n  Interpretation: large principal angles confirm W_q and W_k")
    print(f"  span different subspaces → QK routing is genuinely asymmetric.")
    print(f"  This is the geometric content of the 1.27-1.50 asymmetry metric.")

    del model; gc.collect()
    return results


# =============================================================================
# SECTION G — Multi-layer rotation surgery
# =============================================================================

def section_G(tokenizer, wt2_base, lmda_base):
    print("\n" + "="*65)
    print("  SECTION G — Multi-layer rotation surgery")
    print("  Replaces W_v with polar Q across multiple layer sets.")
    print("  L8-L11 had near-zero individual Q-surgery cost (Phase 0).")
    print("="*65)

    ref = fresh_model()
    Qs  = {}
    for layer in range(N_LAYERS):
        _, _, W_v = get_qkv(ref, layer)
        Q, _      = polar(W_v.astype(np.float64))
        Qs[layer] = preserve_norm(Q.astype(np.float32), W_v)
    del ref; gc.collect()

    layer_sets = {
        "L8-L11 (late, near-zero cost)":     [8, 9, 10, 11],
        "L0+L8-L11":                          [0, 8, 9, 10, 11],
        "L0-L3 (early)":                      [0, 1, 2, 3],
        "L0-L7 (first half)":                 list(range(8)),
        "all L0-L11":                         list(range(12)),
    }

    results = {}
    print(f"\n  {'Layer set':<28} {'WT2':>8} {'WT2Δ%':>7} "
          f"{'LMDA':>9} {'LMDAδ%':>8} {'Acc':>7}")
    print("  " + "─"*70)

    for name, layers in layer_sets.items():
        model = fresh_model()
        for layer in layers:
            inject_wv(model, layer, Qs[layer])
        wt2        = measure_wt2_ppl(model, tokenizer)
        lmda, acc  = measure_lambada(model, tokenizer)
        del model; gc.collect()

        d_wt2  = round((wt2  - wt2_base)  / wt2_base  * 100, 1)
        d_lmda = round((lmda - lmda_base) / lmda_base * 100, 1)

        print(f"  {name:<28} {wt2:>8.2f} {d_wt2:>+7.1f}% "
              f"{lmda:>9.2f} {d_lmda:>+8.1f}% {acc:>6.1f}%")
        results[name] = {
            "layers_replaced": layers,
            "wt2_ppl": wt2, "wt2_delta_pct": d_wt2,
            "lambada_ppl": lmda, "lambada_delta_pct": d_lmda,
            "lambada_acc": acc,
        }

    return results


# =============================================================================
# SECTION H — Effective rank W_q, W_k vs G
# =============================================================================

def section_H():
    print("\n" + "="*65)
    print("  SECTION H — Effective rank W_q, W_k individually vs G")
    print("  If eff_rank(G) << min(eff_rank(W_q), eff_rank(W_k)):")
    print("  → Bottleneck is in the Q-K interaction, not individual matrices.")
    print("  → This is where QK compression is most feasible.")
    print("="*65)

    model   = fresh_model()
    results = {}

    print(f"\n  {'Lyr':<4} {'er_Wq':>8} {'er_Wk':>8} {'er_G':>8} "
          f"{'v90_Wq':>7} {'v90_Wk':>7} {'v90_G':>7} {'ratio':>7}  bottleneck?")
    print("  " + "─"*76)

    for layer in range(N_LAYERS):
        W_q, W_k, _ = get_qkv(model, layer)
        G = W_q.astype(np.float64) @ W_k.astype(np.float64).T

        er_q  = shannon_eff_rank(W_q)
        er_k  = shannon_eff_rank(W_k)
        er_g  = shannon_eff_rank(G)
        v90_q = var90_rank(W_q)
        v90_k = var90_rank(W_k)
        v90_g = var90_rank(G)

        ratio  = er_g / min(er_q, er_k)
        bottle = "interaction" if ratio < 0.5 else "matrices" if ratio > 0.9 \
                 else "mixed"

        print(f"  L{layer:02d}  {er_q:>8.1f} {er_k:>8.1f} {er_g:>8.1f} "
              f"{v90_q:>7d} {v90_k:>7d} {v90_g:>7d} {ratio:>7.3f}  {bottle}")

        results[f"L{layer:02d}"] = {
            "eff_rank_Wq": er_q, "eff_rank_Wk": er_k, "eff_rank_G": er_g,
            "var90_Wq": v90_q, "var90_Wk": v90_k, "var90_G": v90_g,
            "G_to_min_ratio": ratio,
        }

    del model; gc.collect()
    return results


# =============================================================================
# SECTION I — W_v @ W_o combined operator
# =============================================================================

def section_I():
    print("\n" + "="*65)
    print("  SECTION I — W_v @ W_o combined operator geometric analysis")
    print("  Full [768x768] product across layers.")
    print("  Effective rank tells you information write dimensionality.")
    print("  Compare to W_v alone to see what W_o adds or removes.")
    print("="*65)

    model   = fresh_model()
    results = {}

    print(f"\n  {'Lyr':<4} {'er_VO':>8} {'mp_VO':>7} {'od_VO':>8} "
          f"  {'er_Wv':>8} {'mp_Wv':>7} {'od_Wv':>8}  W_o effect")
    print("  " + "─"*72)

    for layer in range(N_LAYERS):
        _, _, W_v = get_qkv(model, layer)
        W_o       = get_wo(model, layer)
        VO        = W_v.astype(np.float64) @ W_o.astype(np.float64)  # [768,768]

        er_vo = shannon_eff_rank(VO)
        mp_vo, mass_vo = mp_outliers(VO)
        od_vo = orth_dev(VO)

        er_wv = shannon_eff_rank(W_v)
        mp_wv, _ = mp_outliers(W_v)
        od_wv = orth_dev(W_v)

        effect = ("VO more isometric than Wv" if od_vo < od_wv
                  else "VO less isometric than Wv")

        print(f"  L{layer:02d}  {er_vo:>8.2f} {mp_vo:>7d} {od_vo:>8.2f}  "
              f"  {er_wv:>8.2f} {mp_wv:>7d} {od_wv:>8.2f}  {effect}")

        results[f"L{layer:02d}"] = {
            "VO": {"eff_rank": er_vo, "mp_outliers": mp_vo,
                   "outlier_mass": mass_vo, "orth_dev": od_vo},
            "Wv": {"eff_rank": er_wv, "mp_outliers": mp_wv, "orth_dev": od_wv},
            "wo_makes_more_isometric": od_vo < od_wv,
        }

    del model; gc.collect()
    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Extended geometric analysis of GPT-2 attention weights v2"
    )
    p.add_argument("--sections", nargs="+",
                   default=list("ABCDEFGHI"),
                   metavar="S",
                   help="Sections to run: A B C D E F G H I")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip PPL evaluation (sections B C E F H I only)")
    return p.parse_args()


def main():
    args     = parse_args()
    sections = [s.upper() for s in args.sections]
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  Geometric Analysis v2 — {MODEL_NAME}")
    print(f"  Sections: {' '.join(sections)}")
    print(f"  Device: {DEVICE}   {ts}")
    print(f"{'='*65}")

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {
        "meta": {
            "model": MODEL_NAME, "timestamp": ts,
            "device": DEVICE, "sections": sections,
        }
    }

    eval_sections = {"A", "D", "G"}
    wt2_base = lmda_base = None

    if eval_sections.intersection(sections) and not args.skip_eval:
        print("\n  Measuring baseline...")
        bm = fresh_model()
        wt2_base          = measure_wt2_ppl(bm, tokenizer)
        lmda_base, lmda_a = measure_lambada(bm, tokenizer)
        del bm; gc.collect()
        print(f"  Baseline: WT2={wt2_base}  LMDA={lmda_base}  Acc={lmda_a}%")
        all_results["baseline"] = {
            "wt2_ppl": wt2_base, "lambada_ppl": lmda_base, "lambada_acc": lmda_a
        }

    dispatch = {
        "A": lambda: section_A(tokenizer, wt2_base, lmda_base),
        "B": section_B,
        "C": section_C,
        "D": lambda: section_D(tokenizer, wt2_base, lmda_base),
        "E": section_E,
        "F": section_F,
        "G": lambda: section_G(tokenizer, wt2_base, lmda_base),
        "H": section_H,
        "I": section_I,
    }

    for sec in sections:
        if sec in dispatch:
            if sec in eval_sections and args.skip_eval:
                print(f"\n  Skipping section {sec} (--skip-eval)")
                continue
            all_results[sec] = dispatch[sec]()
        else:
            print(f"\n  Unknown section: {sec}")

    out_path = os.path.join(RESULTS_DIR, f"geometric_v2_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()