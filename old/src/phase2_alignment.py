"""
Phase 2:  Alignment analysis.

This script is the Phase 2 counterpart to Surgery.
Every operator tested in surgery has a corresponding alignment
measurement here.  The alignment score (Grassmannian distance
to G_true) is a prediction of surgery performance — operators
closer to G_true on the Grassmannian should cause less LAMBADA
damage.  If that prediction holds, Phase 2 and Phase 3 are
measuring the same underlying quantity from two angles.

MEASUREMENTS
------------

Section 1 — W_v alignment with U_Σ (activation manifold isometry)
  Frobenius projection score at r = head_dim, d/4, d/2.
  Claim: W_v's column space is close to the activation covariance
  rotation U_Σ — provable because polar_Q = pca_proc in surgery.

Section 2 — G eigenvalue signature
  For each layer: psd_fraction of G_true, G*_pre, G*_post.
  Claim: G is indefinite (psd_frac < 0.5) in early layers.
  G*_pre should partially recover this; G*_post cannot.
  Pre-softmax vs post-softmax gap quantifies sign contribution.

Section 3 — PMI signature analysis
  G_pmi computed at each layer via static and layer-specific embeddings.
  Metrics: psd_delta, sign_agreement, rank_correlation, amplification.
  Claim: PMI predicts G's sign structure; training amplifies PMI
  directions by ~5x.

Section 4 — Grassmannian alignment table (surgery-to-alignment bridge)
  All G operators ranked by alignment with G_true.
  Prediction: alignment rank should match surgery LAMBADA damage rank.
  This is the key cross-validation between Phase 2 and Phase 3.

Usage
-----
  python phase2_unified.py
  python phase2_unified.py --model gpt2 --only-sections 1 4
  python phase2_unified.py --layer 5    # analyse only one layer

Output
------
  phase2_results/phase2_unified_<model>_<timestamp>.json
"""

import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers import hdr, save_json_results

warnings.filterwarnings("ignore")

# ─── configuration ────────────────────────────────────────────────────────────
N_CHUNKS       = 64
CHUNK_LEN      = 512
REG_EPS        = 1e-4
N_RAND         = 20
PMI_WINDOW     = 5
PMI_MIN_COUNT  = 2      # was 5 → covers only ~4% vocab; 2 gives ~16%
RESULTS_DIR    = "phase2_results"


# ─── model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    model     = AutoModelForCausalLM.from_pretrained(model_name,
                                                     dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().to(device)
    cfg = model.config
    d   = cfg.n_embd
    nh  = cfg.n_head
    nl  = cfg.n_layer
    print(f"  {nl}L  d={d}  {nh}h  head_dim={d//nh}  "
          f"vocab={cfg.vocab_size}")
    return model, tokenizer, d, nh, nl, cfg.vocab_size


# ─── weight extraction ────────────────────────────────────────────────────────

def get_weights(model, layer: int, d: int):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    return W[:, :d].copy(), W[:, d:2*d].copy(), W[:, 2*d:].copy()


def get_true_g(model, layer: int, d: int, n_heads: int) -> np.ndarray:
    """G = Σ_h W_q_h @ W_k_h.T / sqrt(head_dim)  — the full bilinear form."""
    W_q, W_k, _ = get_weights(model, layer, d)
    hd = d // n_heads
    G  = np.zeros((d, d), dtype=np.float64)
    for h in range(n_heads):
        sl = slice(h * hd, (h + 1) * hd)
        G += (W_q[:, sl] @ W_k[:, sl].T) / np.sqrt(hd)
    return G


def get_static_embedding(model) -> np.ndarray:
    return model.transformer.wte.weight.data.cpu().numpy().copy()


# ─── corpus loading ───────────────────────────────────────────────────────────

def load_corpus(tokenizer):
    from datasets import load_dataset
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids  = tokenizer.encode(text, return_tensors="pt")   # [1, N]
    print(f"  Corpus: {ids.shape[1]:,} tokens")
    return ids


def chunk_starts(all_tokens: torch.Tensor) -> list:
    total = all_tokens.shape[1]
    return np.linspace(0, total - CHUNK_LEN - 1, N_CHUNKS).astype(int).tolist()


# ─── activation harvesting ────────────────────────────────────────────────────

def harvest_all(model, all_tokens: torch.Tensor,
                d: int, n_heads: int, n_layers: int, device: str,
                cache_raw_layers: Optional[List[int]] = None
                ) -> Tuple[List[np.ndarray], List[np.ndarray],
                           List[np.ndarray], List[np.ndarray],
                           Dict[int, np.ndarray]]:
    """
    Single forward-pass loop collecting everything needed:
      sigma_xx[l]   — input covariance X.T @ X
      C_xs_pre[l]   — X.T @ S_pre  @ X  (pre-softmax scores, signed)
      C_xs_post[l]  — X.T @ S_post @ X  (post-softmax weights, positive)
      E_layer[l]    — mean activation per token at layer l
      raw_X         — dict {layer_idx: ndarray [N*T, d]} for cached layers

    S_pre can be negative → C_xs_pre captures repulsion.
    S_post is always ≥ 0  → C_xs_post is PSD, blind to repulsion.
    """
    hd = d // n_heads
    if cache_raw_layers is None:
        cache_raw_layers = []

    sigma_xx   = [np.zeros((d, d), dtype=np.float64) for _ in range(n_layers)]
    C_xs_pre   = [np.zeros((d, d), dtype=np.float64) for _ in range(n_layers)]
    C_xs_post  = [np.zeros((d, d), dtype=np.float64) for _ in range(n_layers)]
    sum_acts   = [np.zeros((model.config.vocab_size, d), dtype=np.float64)
                  for _ in range(n_layers)]
    tok_counts = np.zeros(model.config.vocab_size, dtype=np.float64)
    n_tokens   = [0] * n_layers
    raw_X_buf: Dict[int, List[np.ndarray]] = {l: [] for l in cache_raw_layers}

    # Use a dict (not list-of-None) so PyCharm knows values are ndarray
    hook_store: Dict[int, np.ndarray] = {}

    def make_hook(l):
        def hook(module, attn_input, output):
            x   = attn_input[0].detach().float().squeeze(0).cpu().numpy()
            T   = x.shape[0]
            W   = module.c_attn.weight.data.float().cpu().numpy()
            b   = module.c_attn.bias.data.float().cpu().numpy()
            qkv = x @ W + b
            Q_a = qkv[:, :d];   K_a = qkv[:, d:2*d]

            S_p  = np.zeros((T, T), dtype=np.float64)
            S_po = np.zeros((T, T), dtype=np.float64)
            for h in range(n_heads):
                sl    = slice(h * hd, (h + 1) * hd)
                Q_h   = Q_a[:, sl].astype(np.float64)
                K_h   = K_a[:, sl].astype(np.float64)
                raw_h = Q_h @ K_h.T / np.sqrt(hd)
                msk         = np.triu(np.ones((T, T), dtype=bool), k=1)
                raw_m       = raw_h.copy()
                raw_m[msk]  = -1e9
                exp_h       = np.exp(raw_m - raw_m.max(1, keepdims=True))
                soft_h      = exp_h / exp_h.sum(1, keepdims=True)
                S_p  += raw_h
                S_po += soft_h

            x64 = x.astype(np.float64)
            sigma_xx[l]  += x64.T @ x64
            C_xs_pre[l]  += x64.T @ S_p  @ x64
            C_xs_post[l] += x64.T @ S_po @ x64
            n_tokens[l]  += T
            hook_store[l] = x   # dict keyed by int; value is ndarray
        return hook

    hooks = [
        model.transformer.h[l].attn.register_forward_hook(make_hook(l))
        for l in range(n_layers)
    ]

    starts = chunk_starts(all_tokens)
    print(f"  Harvesting {N_CHUNKS}×{CHUNK_LEN} tokens across {n_layers} layers...")
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(starts):
            chunk   = all_tokens[:, s:s+CHUNK_LEN].to(device)
            tok_ids = chunk[0].cpu().tolist()
            model(chunk)
            for l in range(n_layers):
                acts: np.ndarray = hook_store[l]  # dict value is always ndarray
                for pos in range(len(tok_ids)):
                    tid = tok_ids[pos]
                    if 0 <= tid < model.config.vocab_size:
                        sum_acts[l][tid] += acts[pos]
                        if l == 0:
                            tok_counts[tid] += 1
                if l in raw_X_buf:
                    raw_X_buf[l].append(acts.copy())
            if (i + 1) % 16 == 0:
                print(f"    chunk {i+1}/{N_CHUNKS}")

    for hk in hooks:
        hk.remove()

    for l in range(n_layers):
        N = max(n_tokens[l], 1)
        sigma_xx[l]  /= N
        C_xs_pre[l]  /= N
        C_xs_post[l] /= N

    vocab = model.config.vocab_size
    E_layers = []
    for l in range(n_layers):
        E_l  = np.zeros((vocab, d), dtype=np.float32)
        seen = tok_counts > 0
        E_l[seen] = (sum_acts[l][seen] / tok_counts[seen, np.newaxis]).astype(np.float32)
        E_layers.append(E_l)

    # Concatenate cached raw activations into single arrays
    raw_X = {l: np.concatenate(raw_X_buf[l], axis=0).astype(np.float32)
             for l in cache_raw_layers if raw_X_buf[l]}

    n_seen = int(np.sum(tok_counts > 0))
    print(f"  Tokens seen for E_layer: {n_seen}/{vocab}")

    return sigma_xx, C_xs_pre, C_xs_post, E_layers, raw_X


# ─── PMI ─────────────────────────────────────────────────────────────────────

def compute_pmi(all_tokens: torch.Tensor, vocab_size: int) -> np.ndarray:
    """Signed PMI matrix [vocab, vocab]. Negative = repulsion."""
    print(f"  Computing PMI (window={PMI_WINDOW}, min_count={PMI_MIN_COUNT})...")
    ids_list = all_tokens[0].tolist()
    counts_i  = np.zeros(vocab_size, dtype=np.float64)
    counts_ij = defaultdict(float)
    total_i = total_ij = 0

    starts = np.linspace(0, len(ids_list) - CHUNK_LEN,
                         N_CHUNKS, dtype=int).tolist()
    for s in starts:
        seq = ids_list[s:s+CHUNK_LEN]
        T   = len(seq)
        for pos in range(T):
            tid = seq[pos]
            counts_i[tid] += 1;  total_i += 1
            lo = max(0, pos - PMI_WINDOW)
            hi = min(T, pos + PMI_WINDOW + 1)
            for k in range(lo, hi):
                if k != pos:
                    counts_ij[(tid, seq[k])] += 1.0;  total_ij += 1

    n_active = int(np.sum(counts_i >= PMI_MIN_COUNT))
    n_seen   = int(np.sum(counts_i >= 1))
    print(f"  Tokens ≥1: {n_seen}/{vocab_size}  "
          f"Active (≥{PMI_MIN_COUNT}): {n_active}/{vocab_size} "
          f"→ G_pmi rank ≤ {n_active}")

    P_i = counts_i / max(total_i, 1)
    PMI = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for (i, j), c in counts_ij.items():
        if counts_i[i] < PMI_MIN_COUNT or counts_i[j] < PMI_MIN_COUNT:
            continue
        p_ij = c / max(total_ij, 1)
        denom = P_i[i] * P_i[j]
        if denom > 0 and p_ij > 0:
            PMI[i, j] = float(np.log(p_ij / denom))

    n_nz  = int(np.count_nonzero(PMI))
    n_neg = int(np.sum(PMI < 0))
    print(f"  PMI: {n_nz:,} nonzero  {100*n_neg/max(n_nz,1):.1f}% negative")
    return PMI


# ─── operator builders ────────────────────────────────────────────────────────
# Each returns the candidate [d, d] matrix in the same space as G_true,
# for Grassmannian alignment comparison.

def build_wv_operators(X: np.ndarray, W_v_orig: np.ndarray, d: int) -> dict:
    """
    Returns W_v candidate matrices (not factorised — alignment is on W_v directly).
    Keys match surgery_all_v2.py experiment names.
    """
    ops = {}

    # PCA + Procrustes
    pca   = PCA(n_components=d, svd_solver="full")
    pca.fit(X.astype(np.float32))
    V_pca = pca.components_.T.astype(np.float32)
    from scipy.linalg import orthogonal_procrustes
    R, _  = orthogonal_procrustes(V_pca.T, W_v_orig.T)
    ops["wv_pca_proc"]   = (R.T @ V_pca).astype(np.float64)

    # Whitening + Procrustes
    X_c   = X - X.mean(0, keepdims=True)
    Cov   = X_c.T @ X_c / len(X_c)
    eigs, V = np.linalg.eigh(Cov.astype(np.float64))
    eigs  = np.maximum(eigs, REG_EPS * eigs[-1])
    W_wh  = ((V * (1.0/np.sqrt(eigs))[np.newaxis, :]) @ V.T).astype(np.float32)
    R2, _ = orthogonal_procrustes(W_wh.T, W_v_orig.T)

    # Raw PCA (no Procrustes) — orientation cost baseline
    ops["wv_pca_raw"]    = V_pca.astype(np.float64)

    return ops


def build_g_operators(PMI: np.ndarray,
                      E_layer: np.ndarray,
                      W_q_orig: np.ndarray,
                      W_k_orig: np.ndarray,
                      X_raw: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Returns candidate G matrices [d, d] keyed by surgery experiment name.
    Includes both no-Procrustes (raw) and Procrustes-aligned variants.
    """
    from scipy.linalg import orthogonal_procrustes
    ops = {}

    PMI_E_l  = PMI.astype(np.float32) @ E_layer.astype(np.float32)
    G_pmi_l  = E_layer.T.astype(np.float64) @ PMI_E_l.astype(np.float64)
    ops["g_pmi_layer"] = G_pmi_l

    # Procrustes-aligned variant: aligns G_pmi row space toward G_true
    G_true = (W_q_orig.astype(np.float64) @ W_k_orig.astype(np.float64).T)
    R, _   = orthogonal_procrustes(G_pmi_l.T.astype(np.float64), G_true.T)
    ops["g_pmi_layer_proc"] = (R.T @ G_pmi_l).astype(np.float64)

    return ops


# ─── analysis functions ───────────────────────────────────────────────────────

def eig_signature(M: np.ndarray) -> dict:
    """Symmetrise and compute eigenvalue signature."""
    Ms   = (M + M.T) / 2.0
    eigs = np.linalg.eigvalsh(Ms)
    n    = len(eigs)
    npos = int(np.sum(eigs > 0))
    nneg = int(np.sum(eigs < 0))
    emax = float(eigs[-1]);  emin = float(eigs[0])
    abs_e = np.abs(eigs)
    return {
        "psd_fraction":      float(npos / n),
        "n_positive":        npos,
        "n_negative":        nneg,
        "eig_max":           emax,
        "eig_min":           emin,
        "pos_neg_ratio":     abs(emax) / max(abs(emin), 1e-12) if nneg > 0 else float("inf"),
        "spectral_mass_pos": float(abs_e[eigs > 0].sum() / max(abs_e.sum(), 1e-12)),
        "eig_sorted_desc":   eigs[::-1].tolist(),
    }


def grassmannian_score(M_cand: np.ndarray, M_true: np.ndarray,
                       r: int, n_rand: int = N_RAND) -> dict:
    """
    Frobenius projection score between top-r subspaces of M_cand and M_true.
    score = ||U_true[:,:r].T @ U_cand[:,:r]||_F / r
    = 1.0 when identical, ≈ r/d when random.
    """
    d = M_cand.shape[0]
    U_c, _, _ = np.linalg.svd(M_cand.astype(np.float64), full_matrices=False)
    U_t, _, _ = np.linalg.svd(M_true.astype(np.float64), full_matrices=False)
    score = float(np.linalg.norm(U_t[:, :r].T @ U_c[:, :r], "fro") / r)

    rand_s = []
    for _ in range(n_rand):
        R1 = np.linalg.qr(np.random.randn(d, r))[0]
        rand_s.append(float(np.linalg.norm(U_t[:, :r].T @ R1, "fro") / r))
    rand_base = float(np.mean(rand_s))
    return {
        "frob_score":     score,
        "rand_base":      rand_base,
        "above_rand_pct": 100.0 * (score - rand_base) / rand_base,
    }


def pmi_compare(sig_G: dict, sig_pmi: dict) -> dict:
    """Sign agreement and rank correlation between G and G_pmi spectra."""
    e_G   = np.array(sig_G["eig_sorted_desc"])
    e_pmi = np.array(sig_pmi["eig_sorted_desc"])
    n     = min(len(e_G), len(e_pmi))
    i_G   = np.argsort(-np.abs(e_G[:n]))
    i_pmi = np.argsort(-np.abs(e_pmi[:n]))
    sign_agr = float(np.mean(
        np.sign(e_G[i_G]) == np.sign(e_pmi[i_pmi])
    ))

    abs_G   = np.abs(e_G[:n])
    abs_pmi = np.abs(e_pmi[:n])
    rk_G    = np.argsort(np.argsort(-abs_G)).astype(float)
    rk_pmi  = np.argsort(np.argsort(-abs_pmi)).astype(float)
    num     = np.mean((rk_G - rk_G.mean()) * (rk_pmi - rk_pmi.mean()))
    den     = np.std(rk_G) * np.std(rk_pmi)
    rank_corr = float(num / den) if den > 0 else 0.0

    return {
        "psd_delta":        sig_pmi["psd_fraction"] - sig_G["psd_fraction"],
        "sign_agreement":   sign_agr,
        "rank_correlation": rank_corr,
    }


def amplification(G_true: np.ndarray, G_pmi: np.ndarray,
                  top_k: int = 20) -> dict:
    """
    Measure how much training amplified PMI directions.
    amplification_ratio[i] = Rayleigh(G_true, v_pmi_i) / |eig_pmi_i|
    """
    d   = G_true.shape[0]
    Gt  = (G_true + G_true.T) / 2.0
    Gp  = (G_pmi  + G_pmi.T)  / 2.0

    eigs_t, vecs_t = np.linalg.eigh(Gt)
    eigs_p, vecs_p = np.linalg.eigh(Gp)
    idx_t = np.argsort(-np.abs(eigs_t));  top_t = vecs_t[:, idx_t[:top_k]]
    idx_p = np.argsort(-np.abs(eigs_p));  top_p = vecs_p[:, idx_p[:top_k]]
    top_eigs_p = eigs_p[idx_p[:top_k]]

    ratios = [float((top_p[:, i] @ Gt @ top_p[:, i]) /
                    (abs(float(top_eigs_p[i])) + 1e-10))
              for i in range(top_k)]

    frob = float(np.linalg.norm(top_t.T @ top_p, "fro") / top_k)
    rand_frobs = [float(np.linalg.norm(
        np.linalg.qr(np.random.randn(d, top_k))[0].T @ top_p, "fro"
    ) / top_k) for _ in range(20)]
    rand_f = float(np.mean(rand_frobs))

    return {
        "mean_amplification":          float(np.mean(ratios)),
        "grassmannian_above_rand_pct": 100.0 * (frob - rand_f) / rand_f,
    }

# ─── main ────────────────────────────────────────────────────────────────────

def run(model_name: str, device: str,
        only_sections: list = None, target_layer: int = None) -> str:

    print(f"\n{'#'*65}\n#  Phase 2 Unified  —  {model_name}\n{'#'*65}")

    model, tokenizer, d, nh, nl, vocab = load_model(model_name, device)
    hd    = d // nh
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers = [target_layer] if target_layer is not None else list(range(nl))

    # ── Data collection ───────────────────────────────────────────────────
    all_tokens = load_corpus(tokenizer)
    # QK_LAYER = last layer for GPT-2; cache raw activations for crosscov
    qk_layer = nl - 1
    sigma_xx, C_xs_pre, C_xs_post, E_layers, raw_X = harvest_all(
        model, all_tokens, d, nh, nl, device,
        cache_raw_layers=[qk_layer]
    )
    PMI      = compute_pmi(all_tokens, vocab)

    def should(sec):
        return only_sections is None or str(sec) in only_sections

    all_results: Dict[str, Any] = {
        "meta": {
            "model_name": model_name, "n_layers": nl,
            "d_model": d, "n_heads": nh, "head_dim": hd,
            "n_chunks": N_CHUNKS, "chunk_len": CHUNK_LEN,
            "pmi_window": PMI_WINDOW, "pmi_min_count": PMI_MIN_COUNT,
            "reg_eps": REG_EPS, "timestamp": ts,
        },
        "layers": {str(l): {} for l in layers},
    }

    # =========================================================================
    # SECTION 1 — W_v Grassmannian alignment with U_Σ
    # =========================================================================
    if should(1):
        hdr("SECTION 1 — W_v alignment with activation covariance U_Σ")
        print(
            "  Claim: W_v's principal-axis isometry = activation manifold isometry\n"
            "  Proved by surgery: polar_Q = pca_proc (zero orientation cost).\n"
            "  Here we measure HOW WELL each W_v candidate aligns with U_Σ.\n"
        )
        rank_fracs = [hd/d, 0.25, 0.50]
        print(f"  {'Layer':<6}", end="")
        for frac in rank_fracs:
            r = max(1, int(d*frac))
            print(f"  {'r='+str(r):>14}", end="")
        print()
        print("  " + "─" * (6 + 16 * len(rank_fracs)))

        for layer in layers:
            _, _, W_v = get_weights(model, layer, d)
            U_sigma   = np.linalg.svd(sigma_xx[layer], full_matrices=False)[0]
            U_wv      = np.linalg.svd(W_v.astype(np.float64), full_matrices=False)[0]

            row: dict[str, Any] = {}
            print(f"  L{layer:02d}  ", end="")
            for frac in rank_fracs:
                r         = max(1, int(d * frac))
                score     = float(np.linalg.norm(
                    U_sigma[:, :r].T @ U_wv[:, :r], "fro") / r)
                rand_s    = [float(np.linalg.norm(
                    np.linalg.qr(np.random.randn(d, r))[0].T @ U_sigma[:, :r],
                    "fro") / r) for _ in range(N_RAND)]
                rand_base = float(np.mean(rand_s))
                pct       = 100.0 * (score - rand_base) / rand_base
                label     = f"r{r}"
                row[label] = {"frob_score": score, "rand_base": rand_base,
                               "above_rand_pct": pct}
                print(f"  {score:.4f}({pct:+.0f}%)", end="")
            print()
            all_results["layers"][str(layer)]["wv_alignment"] = row

        # W_v operator candidates
        hdr("SECTION 1b — W_v operator candidates alignment with U_Σ")
        print(
            "  For each candidate operator, measure alignment with U_Σ at r=head_dim.\n"
            "  This bridges the alignment score to surgery predictions.\n"
        )
        r_head = hd
        print("  " + "─" * 50)

        for layer in layers:
            _, _, W_v_orig = get_weights(model, layer, d)
            # we need activations for W_v operators
            # Use sigma_xx SVD for X proxy (covariance eigenvectors as stand-in)
            # For full accuracy, would need raw activations — use PCA of sigma_xx
            # as a reasonable approximation for alignment comparison
            U_sigma = np.linalg.svd(sigma_xx[layer], full_matrices=False)[0]
            operator_row: dict[str, Any] = {}
            vals = []
            # Build operators using sigma_xx proxy for X
            # Note: without raw X we can only compute the subspace of U_Σ,
            # not the full Procrustes-aligned operator. Report alignment of
            # U_Σ with W_v (pca_proc is closest to U_Σ by construction)
            U_wv = np.linalg.svd(W_v_orig.astype(np.float64), full_matrices=False)[0]
            score_base = float(np.linalg.norm(
                U_sigma[:, :r_head].T @ U_wv[:, :r_head], "fro") / r_head)
            rand_s   = [float(np.linalg.norm(
                np.linalg.qr(np.random.randn(d, r_head))[0].T
                @ U_sigma[:, :r_head], "fro") / r_head) for _ in range(N_RAND)]
            rand_b   = float(np.mean(rand_s))

            operator_row["wv_pca_proc_vs_Usigma"]   = 100*(score_base - rand_b)/rand_b
            operator_row["note"] = ("wv_pca_proc aligns with U_Σ by construction; "
                           "score reflects W_v's own alignment with U_Σ")
            vals.append(f"{100*(score_base-rand_b)/rand_b:+.0f}%")
            print(f"  L{layer:02d}   {'—':>13} {'—':>14} {vals[0]:>12}")
            all_results["layers"][str(layer)]["wv_operator_alignment"] = operator_row

    # =========================================================================
    # SECTION 2 — G eigenvalue signature
    # =========================================================================
    if should(2):
        hdr("SECTION 2 — G eigenvalue signature: G_true vs G*_pre vs G*_post")
        print(
            "  Key comparison: pre vs post softmax.\n"
            "  Claim: G is indefinite (psd_frac < 0.5) in early layers.\n"
            "  G*_pre should partially recover indefiniteness.\n"
            "  G*_post is structurally PSD (softmax destroys sign) — negative control.\n"
        )
        print(f"  {'Layer':<6} {'G_psd':>7} {'G*pre_psd':>10} {'G*post_psd':>11}  "
              f"{'pre_Δ':>7} {'post_Δ':>7}  {'pre_sign':>9} {'post_sign':>10}")
        print("  " + "─" * 75)

        for layer in layers:
            G_true    = get_true_g(model, layer, d, nh)
            sig_G     = eig_signature(G_true)

            eigs_l, V_l = np.linalg.eigh(sigma_xx[layer])
            floor_l     = REG_EPS * float(eigs_l[-1])
            eigs_l      = np.maximum(eigs_l, floor_l)
            Prec_l      = (V_l * (1.0/eigs_l)[np.newaxis, :]) @ V_l.T

            G_pre  = Prec_l @ C_xs_pre[layer]  @ Prec_l
            G_post = Prec_l @ C_xs_post[layer] @ Prec_l
            sig_pre  = eig_signature(G_pre)
            sig_post = eig_signature(G_post)

            dp = sig_pre["psd_fraction"]  - sig_G["psd_fraction"]
            dpo= sig_post["psd_fraction"] - sig_G["psd_fraction"]
            pre_sign  = "✓" if (sig_pre["psd_fraction"]  < 0.5) == (sig_G["psd_fraction"] < 0.5) else "✗"
            post_sign = "✓" if (sig_post["psd_fraction"] < 0.5) == (sig_G["psd_fraction"] < 0.5) else "✗"

            print(f"  L{layer:02d}   "
                  f"{sig_G['psd_fraction']:>7.3f} "
                  f"{sig_pre['psd_fraction']:>10.3f} "
                  f"{sig_post['psd_fraction']:>11.3f}  "
                  f"{dp:>+7.3f} {dpo:>+7.3f}  "
                  f"{pre_sign:>9} {post_sign:>10}")

            all_results["layers"][str(layer)]["section2_signature"] = {
                "G_true":  {k:v for k,v in sig_G.items()   if k!="eig_sorted_desc"},
                "G_pre":   {k:v for k,v in sig_pre.items() if k!="eig_sorted_desc"},
                "G_post":  {k:v for k,v in sig_post.items()if k!="eig_sorted_desc"},
                "pre_delta_psd":  float(dp),
                "post_delta_psd": float(dpo),
            }

        # Summary
        early = [l for l in layers if l < nl // 4]
        if early:
            g_early_psd   = np.mean([all_results["layers"][str(l)]["section2_signature"]["G_true"]["psd_fraction"]  for l in early])
            pre_early_psd = np.mean([all_results["layers"][str(l)]["section2_signature"]["G_pre"]["psd_fraction"]   for l in early])
            post_early    = np.mean([all_results["layers"][str(l)]["section2_signature"]["G_post"]["psd_fraction"]  for l in early])
            print(f"\n  Early layers (L0–L{early[-1]}) mean psd_frac:")
            print(f"    G_true  = {g_early_psd:.3f}  (target: < 0.5)")
            print(f"    G*_pre  = {pre_early_psd:.3f}  "
                  f"({'partially recovers indefiniteness' if pre_early_psd < 0.5 else 'still PSD — finite-sample ceiling'})")
            print(f"    G*_post = {post_early:.3f}  "
                  f"(expected ≈ 1.0 — softmax destroys sign)")

    # =========================================================================
    # SECTION 3 — PMI signature analysis
    # =========================================================================
    if should(3):
        hdr("SECTION 3 — PMI signature analysis")
        print(
            "  Claim: G implements PMI lifted into representation space.\n"
            "  PMI predicts sign structure; training amplifies by ~5x.\n"
        )
        print(f"  {'Layer':<6} {'G_psd':>7} {'Gpmi_psd':>9} {'Δpsd':>6}  "
              f"{'sign_agr':>9} {'rank_cor':>9}  {'mean_amp':>9}  {'grass%':>7}")
        print("  " + "─" * 78)

        for layer in layers:
            G_true = get_true_g(model, layer, d, nh)
            # normalise G_pmi to match G_true scale
            PMI_El   = PMI.astype(np.float32) @ E_layers[layer].astype(np.float32)
            G_pmi_l  = (E_layers[layer].T.astype(np.float64)
                        @ PMI_El.astype(np.float64))
            ntr  = np.linalg.norm(G_true, "fro")
            npm  = np.linalg.norm(G_pmi_l, "fro")
            if npm > 1e-10:
                G_pmi_l *= ntr / npm

            sig_G   = eig_signature(G_true)
            sig_pmi = eig_signature(G_pmi_l)
            comp    = pmi_compare(sig_G, sig_pmi)
            amp     = amplification(G_true, G_pmi_l, top_k=20)

            print(f"  L{layer:02d}   "
                  f"{sig_G['psd_fraction']:>7.3f} "
                  f"{sig_pmi['psd_fraction']:>9.3f} "
                  f"{comp['psd_delta']:>+6.3f}  "
                  f"{comp['sign_agreement']:>9.3f} "
                  f"{comp['rank_correlation']:>9.4f}  "
                  f"{amp['mean_amplification']:>9.2f}  "
                  f"{amp['grassmannian_above_rand_pct']:>+7.1f}%")

            all_results["layers"][str(layer)]["section3_pmi"] = {
                "G_true": {k:v for k,v in sig_G.items()   if k!="eig_sorted_desc"},
                "G_pmi":  {k:v for k,v in sig_pmi.items() if k!="eig_sorted_desc"},
                "comparison":   comp,
                "amplification": {k: v for k, v in amp.items()
                                  if k != "sign_match_top"},
            }

        if layers:
            mean_rank_corr = np.mean([
                all_results["layers"][str(l)]["section3_pmi"]["comparison"]["rank_correlation"]
                for l in layers
            ])
            mean_amp = np.mean([
                all_results["layers"][str(l)]["section3_pmi"]["amplification"]["mean_amplification"]
                for l in layers
            ])
            print(f"\n  Mean rank_correlation: {mean_rank_corr:.4f}  "
                  f"(PMI predicts which directions training amplified)")
            print(f"  Mean amplification:    {mean_amp:.2f}x")

    # =========================================================================
    # SECTION 4 — Operator fidelity table (surgery bridge, correct metrics)
    # =========================================================================
    if should(4):
        hdr("SECTION 4 — G operator fidelity to G_true (surgery bridge)")
        print(
            "  METRICS FOR G:\n"
            "  psd_delta   : |psd_frac(cand) - psd_frac(G_true)|  → smaller = better\n"
            "                Measures whether the operator has the same proportion\n"
            "                of repulsive directions as G_true.\n"
            "  sign_agree  : fraction of eigenvalues where candidate agrees with\n"
            "                G_true on sign (attractive vs repulsive), after\n"
            "                rank-aligning by |eigenvalue|.\n"
            "  Prediction: lower psd_delta + higher sign_agree → less LAMBADA damage.\n"
        )

        op_names = ["g_pmi_layer", "g_pmi_layer_proc"]

        print(f"  {'Layer':<6} {'metric':<12}", end="")
        for name in op_names:
            print(f"  {name:>15}", end="")   # was: {op_names:>15} — bug fixed
        print()
        print("  " + "─" * (6 + 14 + 17 * len(op_names)))

        for layer in layers:
            G_true  = get_true_g(model, layer, d, nh)
            W_q, W_k, _ = get_weights(model, layer, d)
            X_layer: Optional[np.ndarray] = raw_X.get(layer)
            ops     = build_g_operators(
                PMI, E_layers[layer], W_q, W_k, X_layer
            )

            sig_true = eig_signature(G_true)
            row: Dict[str, Dict] = {}

            # psd_delta row
            print(f"  L{layer:02d}   {'psd_delta':<12}", end="")
            for name in op_names:
                M = ops.get(name)
                if M is None:
                    print(f"  {'N/A':>15}", end="")
                    continue
                sig_cand = eig_signature(M)
                delta    = abs(sig_cand["psd_fraction"] - sig_true["psd_fraction"])
                row.setdefault(name, {})["psd_delta"] = float(delta)
                row[name]["psd_frac_cand"] = sig_cand["psd_fraction"]
                print(f"  {delta:>15.3f}", end="")
            print()

            # sign_agree row
            print(f"  {'':6}   {'sign_agree':<12}", end="")
            for name in op_names:
                M = ops.get(name)
                if M is None:
                    print(f"  {'':>15}", end="")
                    continue
                sig_cand  = eig_signature(M)
                comp      = pmi_compare(sig_true, sig_cand)
                sign_agr  = comp["sign_agreement"]
                row[name]["sign_agree"] = float(sign_agr)
                print(f"  {sign_agr:>15.3f}", end="")
            print()

            all_results["layers"][str(layer)]["section4_fidelity"] = row

        # Ranking summary for the primary target layer
        last_l = layers[-1]
        last   = str(last_l)
        if "section4_fidelity" in all_results["layers"][last]:
            sec4 = all_results["layers"][last]["section4_fidelity"]
            # Rank by psd_delta ascending (lower = better sign match)
            ranked_psd = sorted(
                [(n, sec4[n]["psd_delta"]) for n in op_names if n in sec4],
                key=lambda x: x[1]
            )
            ranked_sign = sorted(
                [(n, sec4[n]["sign_agree"]) for n in op_names if n in sec4],
                key=lambda x: -x[1]
            )
            print(f"\n  Layer {last_l} ranking by psd_delta (↓ = better sign structure match):")
            for rank_i, (name, val) in enumerate(ranked_psd, 1):
                psd_cand = sec4[name]["psd_frac_cand"]
                sig_true_l = eig_signature(get_true_g(model, last_l, d, nh))
                print(f"    {rank_i}. {name:<22}  Δpsd={val:.3f}  "
                      f"(G_true={sig_true_l['psd_fraction']:.3f}  cand={psd_cand:.3f})")

            print(f"\n  Layer {last_l} ranking by sign_agreement (↑ = better):")
            for rank_i, (name, val) in enumerate(ranked_sign, 1):
                print(f"    {rank_i}. {name:<22}  sign_agree={val:.3f}")

            print(f"\n  Claim: psd_delta rank and sign_agree rank should both")
            print(f"  match surgery rank. Grassmannian rank does NOT match")
            print(f"  (it rewards high-variance overlap, not sign fidelity).")

    # ── Save ──────────────────────────────────────────────────────────────────
    path = save_json_results(all_results, model_name, "phase2_unified", ts, RESULTS_DIR)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2 unified: W_v isometry + G signature + PMI + alignment"
    )
    p.add_argument("--model",    default="gpt2")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only-sections", nargs="+", default=None,
                   metavar="N",
                   help="Run only sections e.g. --only-sections 1 4")
    p.add_argument("--layer", type=int, default=None,
                   help="Analyse only one layer (fast debug mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model_name     = args.model,
        device         = args.device,
        only_sections  = args.only_sections,
        target_layer   = args.layer,
    )