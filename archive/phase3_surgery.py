"""
Surgery for W_v (Layer 0) and
G = W_q^T W_k (Layer 11), for every variant.

THEORETICAL BASIS
--------------------------------------------------------
W_v is an isometric operator.
  - Phase 0: near-orthogonal columns
  - Section D: rotation carries all function
  - Section C: training left W_v near-random-orthogonal
  - Phase 3: PCA+Procrustes recovers 97% of capability gap
  Prediction: any isometry close to W_v's principal axes should work.
  The polar rotation Q is the closest possible isometry to W_v — derived
  from W_v itself with no data.

G implements representation-space PMI.
  - Phase 0 Section G: G is indefinite
  - Phase 2: G_pmi = E_l.T @ PMI @ E_l shares G's indefinite signature
  - Phase 2: PMI predicts which
    directions training amplified most
  The remaining gap vs trained G measures how much function comes from the amplification
  that data statistics alone cannot recover.

EXPERIMENTS
-----------

CONTROLS
  baseline            Unmodified GPT-2.
  wv_identity         Reinject W_v unchanged. Must match baseline exactly.
  g_identity          Reinject W_q, W_k unchanged. Must match baseline exactly.
  wv_random_null      Random orthonormal W_v. Null floor for W_v surgeries.
  g_random_null       Random orthonormal W_q, W_k. Null floor for G surgeries.

W_v SURGERIES  (Layer 0)
    wv_polar_q — ablation proving rotation carries all function
    wv_pca_proc — primary statistical replacement (zero orientation cost)
    wv_pca_raw — baseline showing Procrustes is essential

G (layer 11):
    g_pmi_activation — primary operator
    g_pmi_activation_proc
    g_pmi_layer — comparison (mean-per-token vs contextual)
"""

import gc
import json
import math
import os
import warnings
import numpy as np
import torch
import transformers
from datasets import load_dataset
from scipy.linalg import orthogonal_procrustes, polar
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME   = "gpt2"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_CHUNKS     = 128
CHUNK_LENGTH = 1024
V_LAYER   = 0
QK_LAYER  = 11
REG_EPS = 1e-4
# PMI settings
PMI_WINDOW    = 5     # co-occurrence window (tokens either side)
PMI_MIN_COUNT = 2     # min marginal count — was 5 (only 3-6% vocab); 2 gives ~16%

RESULTS_DIR = "results/phase3_results"

# =============================================================================
# CORPUS LOADING  (shared across all harvesters)
# =============================================================================
def load_corpus_tokens(tokenizer):
    """
    Load WikiText-2 train split into a flat token tensor.
    Returned once and reused by all harvesters.
    """
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(t for t in ds["text"] if t.strip())
    parts = []
    for i in range(0, len(full_text), 100_000):
        ids = tokenizer(
            full_text[i:i + 100_000],
            return_tensors="pt", truncation=False, add_special_tokens=False,
        )["input_ids"].squeeze(0)
        parts.append(ids)
    return torch.cat(parts, dim=0)

def corpus_chunks(all_tokens, n_chunks, chunk_length):
    """Return list of start indices evenly spaced across the corpus."""
    return np.linspace(
        0, all_tokens.shape[0] - chunk_length, n_chunks, dtype=int
    ).tolist()

# =============================================================================
# ACTIVATION HARVESTING
# =============================================================================
def harvest_activations(model, all_tokens, n_layers):
    """
    Harvest layer-input (post-LN) activations for all layers simultaneously.
    Returns dict {layer_idx: np.ndarray [N_CHUNKS*CHUNK_LENGTH, n_embd]}.
    """
    store      = {}
    layer_acts = {i: [] for i in range(n_layers)}

    def make_hook(idx):
        def hook(m, inp, out):
            store[idx] = inp[0].detach().cpu()
        return hook

    handles = [
        model.transformer.h[i].attn.register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]

    starts = corpus_chunks(all_tokens, N_CHUNKS, CHUNK_LENGTH)
    print(f"  Harvesting activations: {N_CHUNKS}×{CHUNK_LENGTH} = "
          f"{N_CHUNKS*CHUNK_LENGTH:,} tokens  ({n_layers} layers)...")
    model.eval()
    with torch.no_grad():
        for s in tqdm(starts, desc="  chunks", leave=False):
            chunk = all_tokens[s:s+CHUNK_LENGTH].unsqueeze(0).to(DEVICE)
            model(input_ids=chunk)
            for i in range(n_layers):
                layer_acts[i].append(store[i].squeeze(0).numpy())

    for h in handles:
        h.remove()

    return {
        i: np.concatenate(layer_acts[i], axis=0).astype(np.float32)
        for i in range(n_layers)
    }


def harvest_pmi_activation(
        model, all_tokens, layer_idx: int, PMI: np.ndarray,
        n_chunks: int, chunk_len: int, pmi_window: int, device: str,
) -> np.ndarray:
    """
    Compute M = (1/N) Σ_{(i,j) in window} PMI[tok_i, tok_j] * x_i.T @ x_j

    - x_i = hook capture = post-LayerNorm activations, pre W_q/W_k/W_v
    - PMI can be negative → M can have negative eigenvalues → captures repulsion
    - Uses context-specific activations, not mean-per-token

    Returns M: [d, d] float64
    """
    n_embd = model.config.n_embd
    vocab  = model.config.vocab_size
    M      = np.zeros((n_embd, n_embd), dtype=np.float64)
    n_pairs = 0

    hook_store: dict = {}
    def _hook(module, attn_input, output):
        hook_store["x"] = attn_input[0].detach().float().squeeze(0).cpu().numpy()
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(_hook)

    starts = np.linspace(0, all_tokens.shape[0] - chunk_len,
                         n_chunks, dtype=int).tolist()
    model.eval()
    with torch.no_grad():
        for s in tqdm(starts, desc="  pmi-activation", leave=False):
            chunk   = all_tokens[s:s+chunk_len].unsqueeze(0).to(device)
            tok_ids = all_tokens[s:s+chunk_len].tolist()
            model(input_ids=chunk)

            x = hook_store["x"]   # [T, d] — post-LN, pre W_q/W_k/W_v
            T = len(tok_ids)

            for pos_i in range(T):
                tok_i = tok_ids[pos_i]
                if tok_i >= vocab:
                    continue
                lo = max(0, pos_i - pmi_window)
                hi = min(T, pos_i + pmi_window + 1)
                for pos_j in range(lo, hi):
                    if pos_j == pos_i:
                        continue
                    tok_j = tok_ids[pos_j]
                    if tok_j >= vocab:
                        continue
                    pmi_w = float(PMI[tok_i, tok_j])
                    if pmi_w == 0.0:
                        continue
                    xi = x[pos_i].astype(np.float64)
                    xj = x[pos_j].astype(np.float64)
                    M       += pmi_w * np.outer(xi, xj)
                    n_pairs += 1

    handle.remove()
    if n_pairs > 0:
        M /= n_pairs

    eigs     = np.linalg.eigvalsh((M + M.T) / 2)
    psd_frac = float(np.mean(eigs > 0))
    n_neg    = int(np.sum(eigs < 0))
    print(f"  g_pmi_activation: {n_pairs:,} pairs  "
          f"psd_frac={psd_frac:.3f}  n_negative_eigs={n_neg}")
    if psd_frac >= 0.5:
        print(f"  WARNING: psd_frac={psd_frac:.3f} ≥ 0.5 — "
              f"operator is PSD-dominant; fewer repulsive pairs than expected")
    return M


def op_g_pmi_activation(M_raw: np.ndarray,
                        w_q_orig: np.ndarray,
                        w_k_orig: np.ndarray) -> tuple:
    """
    SVD-factorise the pmi_activation bilinear form M into (W_q, W_k).
    No Procrustes — raw PMI structure, orientation not corrected.
    Analogous to wv_pca_raw: tests statistical structure without orientation assist.
    """
    return svd_factorise_g(M_raw, w_q_orig, w_k_orig, label="g_pmi_activation")


def op_g_pmi_activation_proc(M_raw: np.ndarray,
                              w_q_orig: np.ndarray,
                              w_k_orig: np.ndarray) -> tuple:
    """
    PMI-activation bilinear form WITH row-space Procrustes alignment to G_true.

    Formula: find R = argmin_R ||M_raw.T @ R - G_true.T||_F  s.t. R^T R = I
             G_aligned = R.T @ M_raw
    Then SVD-factorise G_aligned into W_q, W_k.

    The gap  g_pmi_activation → g_pmi_activation_proc  is the orientation cost for G.
    The remaining gap after Procrustes is the irreducible amplification gap.
    """
    G_true = (w_q_orig.astype(np.float64) @ w_k_orig.astype(np.float64).T).astype(np.float32)
    R, _   = orthogonal_procrustes(M_raw.T, G_true.T)
    G_aligned = (R.T @ M_raw).astype(np.float32)
    dist_before = float(np.linalg.norm(M_raw   - G_true, "fro"))
    dist_after  = float(np.linalg.norm(G_aligned - G_true, "fro"))
    print(f"    Procrustes alignment: ||G_pmi - G_true||_F "
          f"before={dist_before:.2f} → after={dist_after:.2f}")
    return svd_factorise_g(G_aligned, w_q_orig, w_k_orig, label="g_pmi_activation_proc")


def op_g_pmi_proc(PMI: np.ndarray, E: np.ndarray,
                  w_q_orig: np.ndarray, w_k_orig: np.ndarray,
                  label: str = "g_pmi_layer_proc") -> tuple:
    """
    PMI @ E_layer WITH row-space Procrustes alignment to G_true.
    Analogous to wv_pca_proc but for the mean-activation PMI operator.
    """
    PMI_E  = PMI.astype(np.float32) @ E.astype(np.float32)
    G_pmi  = (E.T.astype(np.float64) @ PMI_E.astype(np.float64)).astype(np.float32)
    G_true = (w_q_orig.astype(np.float64) @ w_k_orig.astype(np.float64).T).astype(np.float32)
    R, _   = orthogonal_procrustes(G_pmi.T, G_true.T)
    G_aligned = (R.T @ G_pmi).astype(np.float32)
    return svd_factorise_g(G_aligned, w_q_orig, w_k_orig, label=label)



def harvest_mean_token_activations(model, all_tokens, layer_idx, vocab_size):
    """
    Compute E_l[tok_id] = mean activation of token tok at layer l.
    Returns [vocab_size, n_embd] float32.
    Used to lift PMI into the correct representation space at layer l.
    """
    n_embd = model.config.n_embd
    store  = {}

    def _hook(m, inp, out):
        store["x"] = inp[0].detach().cpu().numpy().astype(np.float32)
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(_hook)

    sum_acts = np.zeros((vocab_size, n_embd), dtype=np.float64)
    counts   = np.zeros(vocab_size, dtype=np.float64)

    # Use a smaller subset for speed — 64 chunks is sufficient for mean estimates
    starts = corpus_chunks(all_tokens, min(64, N_CHUNKS), CHUNK_LENGTH)
    print(f"  Harvesting layer-{layer_idx} mean activations "
          f"({len(starts)} chunks)...")
    model.eval()
    with torch.no_grad():
        for s in tqdm(starts, desc="  mean-act", leave=False):
            chunk    = all_tokens[s:s+CHUNK_LENGTH]
            tok_ids  = chunk.tolist()
            chunk_t  = chunk.unsqueeze(0).to(DEVICE)
            model(input_ids=chunk_t)
            acts = store["x"]
            if acts.ndim == 3:
                acts = acts.squeeze(0)
            for pos, tid in enumerate(tok_ids):
                if 0 <= tid < vocab_size:
                    sum_acts[tid] += acts[pos]
                    counts[tid]   += 1

    handle.remove()

    E_l = np.zeros((vocab_size, n_embd), dtype=np.float32)
    seen = counts > 0
    E_l[seen] = (sum_acts[seen] / counts[seen, np.newaxis]).astype(np.float32)
    print(f"    Tokens seen: {int(seen.sum())}/{vocab_size}")
    return E_l


# =============================================================================
# PMI COMPUTATION
# =============================================================================

def compute_pmi_matrix(all_tokens_list, window, min_count, vocab_size):
    """
    Signed PMI matrix over active vocabulary.
    PMI(i,j) = log P(i,j) / (P(i)·P(j))
    Negative PMI = tokens co-occur less than chance = repulsion.
    Returns dense [vocab_size, vocab_size] float32.

    VOCABULARY COVERAGE:
    min_count=5 on 131k tokens leaves ~3-6% of vocabulary active,
    making G_pmi low-rank relative to the full d=768 space.
    min_count=2 is a better balance: ~16% coverage, still noise-filtered.
    """
    print(f"  Computing PMI: window={window}, min_count={min_count}, "
          f"vocab={vocab_size}...")

    counts_i  = np.zeros(vocab_size, dtype=np.float64)
    counts_ij = defaultdict(float)
    total_i = total_ij = 0

    # Use same chunks as other harvesters
    starts = np.linspace(
        0, len(all_tokens_list) - CHUNK_LENGTH, N_CHUNKS, dtype=int
    ).tolist()

    for s in tqdm(starts, desc="  PMI chunks", leave=False):
        seq = all_tokens_list[s:s + CHUNK_LENGTH]
        T   = len(seq)
        for pos in range(T):
            tid = seq[pos]
            counts_i[tid] += 1
            total_i       += 1
            lo = max(0, pos - window)
            hi = min(T, pos + window + 1)
            for k in range(lo, hi):
                if k != pos:
                    counts_ij[(tid, seq[k])] += 1.0
                    total_ij += 1

    # Coverage diagnostics
    n_active = int(np.sum(counts_i >= min_count))
    n_seen   = int(np.sum(counts_i >= 1))
    print(f"  Tokens seen (≥1):          {n_seen}/{vocab_size} "
          f"({100*n_seen/vocab_size:.1f}%)")
    print(f"  Active in PMI (≥{min_count}):      {n_active}/{vocab_size} "
          f"({100*n_active/vocab_size:.1f}%)  ← G_pmi rank ≤ {n_active}")
    if n_active < 0.10 * vocab_size:
        print(f"  WARNING: PMI active vocab is <10% of total. "
              f"Consider --pmi-min-count 2 for broader coverage.")

    P_i = counts_i / max(total_i, 1)
    PMI = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for (i, j), c_ij in counts_ij.items():
        if counts_i[i] < min_count or counts_i[j] < min_count:
            continue
        p_ij  = c_ij / max(total_ij, 1)
        denom = P_i[i] * P_i[j]
        if denom > 0 and p_ij > 0:
            PMI[i, j] = float(np.log(p_ij / denom))

    n_nonzero = int(np.count_nonzero(PMI))
    n_neg     = int(np.sum(PMI < 0))
    print(f"  PMI: {n_nonzero:,} nonzero entries, "
          f"{100*n_neg/max(n_nonzero,1):.1f}% negative (repulsive pairs)")
    return PMI

# =============================================================================
# EVALUATION
# =============================================================================
def measure_wt2_ppl(model, tokenizer):
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in ds["text"] if t.strip())
    parts = []
    for i in range(0, len(full_text), 100_000):
        ids = tokenizer(
            full_text[i:i+100_000],
            return_tensors="pt", truncation=False, add_special_tokens=False,
        )["input_ids"].squeeze(0)
        parts.append(ids)
    encodings  = torch.cat(parts, dim=0)
    max_length = 1024
    stride     = 512
    nlls       = []
    model.eval()
    for begin in range(0, encodings.size(0) - max_length, stride):
        end        = begin + max_length
        input_ids  = encodings[begin:end].unsqueeze(0).to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :stride] = -100
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=target_ids).loss
        nlls.append(loss.item() * (max_length - stride))
    return round(math.exp(sum(nlls) / (len(nlls) * (max_length - stride))), 4)


def measure_lambada(model, tokenizer):
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except Exception:
        try:
            ds = load_dataset("lambada", split="test")
        except Exception as e:
            print(f"  [LAMBADA] Could not load: {e}")
            return None, None

    model.eval()
    nlls, correct, total = [], 0, 0
    for ex in tqdm(ds, desc="  LAMBADA", leave=False):
        text   = ex.get("text") or ex.get("passage", "")
        tokens = tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True,
        )["input_ids"].squeeze(0)
        if tokens.shape[0] < 2:
            continue
        input_ids  = tokens.unsqueeze(0).to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100
        with torch.no_grad():
            loss   = model(input_ids=input_ids, labels=target_ids).loss
            logits = model(input_ids=input_ids).logits
        nlls.append(loss.item())
        if logits[0, -2, :].argmax().item() == tokens[-1].item():
            correct += 1
        total += 1
    if total == 0:
        return None, None
    return (round(math.exp(sum(nlls) / len(nlls)), 4),
            round(correct / total * 100, 2))


# =============================================================================
# WEIGHT UTILITIES
# =============================================================================

def read_weights(model, layer_idx, n_embd):
    W = model.transformer.h[layer_idx].attn.c_attn.weight.data.cpu().numpy().copy()
    return W[:, :n_embd].copy(), W[:, n_embd:2*n_embd].copy(), W[:, 2*n_embd:].copy()


def inject_weights(model, layer_idx, n_embd, w_q=None, w_k=None, w_v=None):
    W = model.transformer.h[layer_idx].attn.c_attn.weight.data.cpu().numpy().copy()
    if w_q is not None: W[:, :n_embd]         = w_q
    if w_k is not None: W[:, n_embd:2*n_embd] = w_k
    if w_v is not None: W[:, 2*n_embd:]       = w_v
    model.transformer.h[layer_idx].attn.c_attn.weight.data = \
        torch.tensor(W, dtype=torch.float32).to(DEVICE)


def preserve_norm(W_new, W_orig):
    n_orig = np.linalg.norm(W_orig, "fro")
    n_new  = np.linalg.norm(W_new,  "fro")
    if n_new > 1e-8:
        W_new = W_new * (n_orig / n_new)
    return W_new.astype(np.float32)


def procrustes_rowspace(W_stat, W_orig):
    """
    Row-space Procrustes alignment.
    Finds R minimising ||W_stat.T R - W_orig.T||_F, s.t. R^T R = I.
    Returns R.T @ W_stat — W_stat rotated toward W_orig's row space.
    Row space is correct for GPT-2 (Y = X @ W — rows are input directions).
    """
    R, _ = orthogonal_procrustes(W_stat.T, W_orig.T)
    return (R.T @ W_stat).astype(np.float32)

def random_orthonormal(d, seed=None):
    rng  = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
    return Q.astype(np.float32)


def svd_factorise_g(M, w_q_orig, w_k_orig, label=""):
    """
    SVD-factorise M into (W_q, W_k) such that W_q @ W_k.T ∝ M,
    with the PRODUCT G = W_q @ W_k.T scaled to match the original G_true.

    THE BUG THIS FIXES:
    preserve_norm applied independently to W_q and W_k ensures each matrix
    individually matches the original norms, but the PRODUCT W_q @ W_k.T
    ends up scaled by alpha * beta where alpha, beta are individual scale
    factors. This ratio can be ~1.18 (18% off) for typical GPT-2 weights,
    making attention logits systematically too large or too small, and
    causing softmax to operate at the wrong temperature.

    The correct approach: scale the product G = W_q @ W_k.T to match
    norm(w_q_orig @ w_k_orig.T), then apply sqrt of that scale to each
    matrix so W_q and W_k are individually scaled by the same factor.

    abs() guard on singular values: M can be indefinite (especially G_pmi
    and G_regression), but SVD singular values are always non-negative.
    The abs() is a numerical safety guard for near-zero values only.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    sqrt_s   = np.sqrt(np.abs(s)).astype(np.float32)
    eigs_sym = np.linalg.eigvalsh((M + M.T) / 2)
    psd_frac = float(np.mean(eigs_sym > 0))

    # Raw factorisation: W_q_raw @ W_k_raw.T = M exactly
    W_q_raw = (U    * sqrt_s[np.newaxis, :]).astype(np.float32)  # [d, d]
    W_k_raw = (Vt.T * sqrt_s[np.newaxis, :]).astype(np.float32)  # [d, d]

    # Target: norm(G_injected) = norm(G_true) where G_true = w_q_orig @ w_k_orig.T
    # This ensures attention logits have the correct scale for softmax.
    G_orig_norm = float(np.linalg.norm(
        w_q_orig.astype(np.float64) @ w_k_orig.astype(np.float64).T, "fro"
    ))
    G_raw_norm  = float(np.linalg.norm(
        W_q_raw.astype(np.float64) @ W_k_raw.astype(np.float64).T, "fro"
    ))
    sqrt_r = 1.0  # default; overwritten below if G_raw_norm > 1e-8
    if G_raw_norm > 1e-8:
        # Apply sqrt of the scale ratio to each matrix equally:
        # (sqrt_r * W_q_raw) @ (sqrt_r * W_k_raw).T = r * G_raw → norm = G_orig_norm
        sqrt_r  = float(np.sqrt(G_orig_norm / G_raw_norm))
        W_q_new = (W_q_raw * sqrt_r).astype(np.float32)
        W_k_new = (W_k_raw * sqrt_r).astype(np.float32)
    else:
        W_q_new = W_q_raw
        W_k_new = W_k_raw

    # Verify
    G_check_norm = float(np.linalg.norm(
        W_q_new.astype(np.float64) @ W_k_new.astype(np.float64).T, "fro"
    ))

    if label:
        print(f"    {label}: top_sv={s[0]:.4f}  min_sv={s[-1]:.6f}  "
              f"cond={s[0]/max(abs(s[-1]),1e-10):.0f}  psd_frac={psd_frac:.3f}  "
              f"G_norm={G_check_norm:.2f}→{G_orig_norm:.2f}(target)")

    return W_q_new, W_k_new, {
        "sv_max": float(s[0]), "sv_min": float(s[-1]),
        "psd_frac": psd_frac,
        "G_raw_norm": G_raw_norm, "G_orig_norm": G_orig_norm,
        "scale_applied": sqrt_r,
    }

# =============================================================================
# OPERATORS — W_v
# =============================================================================
def op_wv_random(w_v_orig):
    """Null floor: random orthonormal W_v."""
    return preserve_norm(random_orthonormal(w_v_orig.shape[0], seed=0), w_v_orig)

def op_wv_polar_q(w_v_orig):
    """
    W_v's own polar rotation factor Q.

    Polar decomposition: W_v = Q @ S
    Q is the unique orthogonal matrix closest to W_v (pure rotation).
    S is the PSD stretch factor.

    This is NOT a statistical replacement — it uses W_v itself.
    It tests the isometry claim directly: if W_v is functionally a rotation,
    replacing it with Q should preserve almost all capability.

    The gap vs baseline = cost of removing the stretch component S.
    The gap vs wv_pca_proc = cost of using a data-derived isometry instead
                              of W_v's own rotation.
    Q-surgery result from Phase 0: +3.4% WT2 PPL (vs +105% for S-surgery).
    Expected here: smallest WT2 cost of any W_v surgery.
    """
    Q, S = polar(w_v_orig.astype(np.float64))
    Q    = Q.astype(np.float32)
    sv_Q = np.linalg.svd(Q, compute_uv=False)
    print(f"    Polar Q: sv range [{sv_Q[-1]:.6f}, {sv_Q[0]:.6f}]  "
          f"(should be ≈1 for orthogonal matrix)")
    return preserve_norm(Q, w_v_orig)


def op_wv_pca_raw(X, w_v_orig):
    """Level 1: PCA rotation, no Procrustes. Tests orientation cost."""
    pca = PCA(n_components=X.shape[1], svd_solver="full")
    pca.fit(X.astype(np.float32))
    return preserve_norm(pca.components_.T.astype(np.float32), w_v_orig)


def op_wv_pca_proc(X, w_v_orig):
    """Level 1+: PCA + Procrustes."""
    pca   = PCA(n_components=X.shape[1], svd_solver="full")
    pca.fit(X.astype(np.float32))
    V_pca = pca.components_.T.astype(np.float32)
    return preserve_norm(procrustes_rowspace(V_pca, w_v_orig), w_v_orig)

# =============================================================================
# OPERATORS — G = W_q^T W_k
# =============================================================================
def op_g_random(w_q_orig, w_k_orig):
    """Null floor: independent random orthonormal W_q, W_k."""
    W_q = preserve_norm(random_orthonormal(w_q_orig.shape[0], seed=0), w_q_orig)
    W_k = preserve_norm(random_orthonormal(w_k_orig.shape[0], seed=1), w_k_orig)
    return W_q, W_k

def op_g_pmi(PMI, E, w_q_orig, w_k_orig, label="g_pmi"):
    """
    PMI lifted into representation space: G_pmi = E.T @ PMI @ E.

    PMI : [vocab, vocab] signed co-occurrence matrix.
    E   : [vocab, d] token representation matrix (static embedding or E_l).

    This is the theoretically motivated replacement for G based on Phase 2:
    - G's indefinite signature matches G_pmi's
    - Rank correlation between |G_eigs| and |G_pmi_eigs| = 0.96
    - Grassmannian alignment above random (top-20 subspaces)

    PMI correctly predicts WHICH directions are repulsive/attractive.
    Gradient descent amplified those directions.

    The gap between g_pmi and the trained G = the amplification factor.
    """
    # E.T @ PMI @ E: float32 for PMI (large matrix), float64 accumulation
    PMI_E  = PMI.astype(np.float32) @ E.astype(np.float32)   # [vocab, d]
    G_pmi  = E.T.astype(np.float64) @ PMI_E.astype(np.float64)  # [d, d]
    G_pmi  = G_pmi.astype(np.float32)
    return svd_factorise_g(G_pmi, w_q_orig, w_k_orig, label=label)


def grassmannian_alignment(M_stat, M_true, r=64, n_rand=20):
    """
    Measure how close the top-r subspace of M_stat is to M_true on the Grassmannian.
    Frobenius projection score: ||U_true[:,:r].T @ U_stat[:,:r]||_F / r
    = 1.0 when subspaces are identical, ≈ r/d for random.
    Returns (score, random_baseline, above_rand_pct).
    """
    d = M_stat.shape[0]
    U_s, _, _ = np.linalg.svd(M_stat, full_matrices=False)
    U_t, _, _ = np.linalg.svd(M_true, full_matrices=False)
    score = float(np.linalg.norm(U_t[:, :r].T @ U_s[:, :r], "fro") / r)

    rand_scores = []
    for _ in range(n_rand):
        R, _ = np.linalg.qr(np.random.randn(d, r))
        rand_scores.append(
            float(np.linalg.norm(U_t[:, :r].T @ R, "fro") / r)
        )
    rand_base = float(np.mean(rand_scores))
    above_pct = 100.0 * (score - rand_base) / rand_base
    return score, rand_base, above_pct


def run_g_alignment_diagnostics(G_true, PMI, E_layer):
    """
    Measures Grassmannian distance between each valid candidate operator and G_true.
    """
    print("\n  ── G Alignment Diagnostics (Grassmannian distance to G_true) ──")
    print(f"  {'Operator':<22} {'frob_score':>11} {'rand_base':>11} "
          f"{'above_rand%':>12}  {'psd_frac':>9}")
    print("  " + "─" * 70)

    eigs_G = np.linalg.eigvalsh((G_true + G_true.T) / 2)
    psd_G  = float(np.mean(eigs_G > 0))

    def _check(label, M):
        score, rand, pct = grassmannian_alignment(M, G_true, r=64)
        eigs_m = np.linalg.eigvalsh((M + M.T) / 2)
        psd_m  = float(np.mean(eigs_m > 0))
        print(f"  {label:<22} {score:>11.4f} {rand:>11.4f} "
              f"{pct:>+12.1f}%  {psd_m:>9.3f}")
        return score, pct, psd_m

    results = {}
    print(f"  {'G_true (target)':<22} {'—':>11} {'—':>11} {'—':>12}  "
          f"{psd_G:>9.3f}")


    PMI_E_l = PMI.astype(np.float32) @ E_layer.astype(np.float32)
    G_pmi_l = (E_layer.T.astype(np.float64)
               @ PMI_E_l.astype(np.float64)).astype(np.float32)
    results["g_pmi_layer"] = _check("g_pmi_layer", G_pmi_l)

    print(f"  Prediction: pmi_layer should show higher alignment than pmi_static")
    print(f"  because E_layer uses the correct representation space for layer {QK_LAYER}.")
    return results

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def fresh_model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    m.eval()
    return m

def run_experiment(name, layer_idx, n_embd, tokenizer,
                   w_q_new, w_k_new, w_v_new,
                   save_dir, wt2_base, lmda_ppl_base):
    print(f"\n  {'─'*55}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {'─'*55}")

    model = fresh_model()
    inject_weights(model, layer_idx, n_embd,
                   w_q=w_q_new, w_k=w_k_new, w_v=w_v_new)

    wt2_ppl               = measure_wt2_ppl(model, tokenizer)
    lambada_ppl, lmda_acc = measure_lambada(model, tokenizer)

    delta_wt2  = round((wt2_ppl - wt2_base) / wt2_base * 100, 2) \
                 if wt2_ppl  and wt2_base       else None
    delta_lmda = round((lambada_ppl - lmda_ppl_base) / lmda_ppl_base * 100, 2) \
                 if lambada_ppl and lmda_ppl_base else None

    print(f"  WT2 PPL:     {wt2_ppl}  ({delta_wt2:+.1f}% vs baseline)")
    print(f"  LAMBADA PPL: {lambada_ppl}  ({delta_lmda:+.1f}%)  "
          f"acc: {lmda_acc}%")

    model_path = os.path.join(save_dir, name)
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name, "wt2_ppl": wt2_ppl, "wt2_delta_pct": delta_wt2,
        "lambada_ppl": lambada_ppl, "lambada_delta_pct": delta_lmda,
        "lambada_acc": lmda_acc,    "saved_to": model_path,
    }


def identity_check(result, wt2_base, tol=0.02):
    delta = abs(result["wt2_ppl"] - wt2_base)
    if delta >= tol:
        raise AssertionError(
            f"Identity check FAILED: {result['name']}  "
            f"PPL={result['wt2_ppl']} vs baseline={wt2_base}  (Δ={delta:.4f})"
        )
    print(f"  ✓ Identity check passed (Δ={delta:.5f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS_DIR, ts)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'='*65}")
    print(f"  surgery_all_v2.py  —  PMI-informed Surgery Suite")
    print(f"  Model: {MODEL_NAME}   Device: {DEVICE}   {ts}")
    print(f"  W_v layer: {V_LAYER}   G layer: {QK_LAYER}")
    print(f"{'='*65}")

    # ── One-time corpus load ──────────────────────────────────────────────────
    print("\n  Loading corpus...")
    all_tokens      = load_corpus_tokens(tokenizer)
    all_tokens_list = all_tokens.tolist()
    print(f"  Total tokens: {len(all_tokens_list):,}")

    # ── Harvest activations (all layers, one pass) ────────────────────────────
    base_m   = fresh_model()
    n_layers = base_m.config.n_layer
    n_embd   = base_m.config.n_embd
    vocab    = base_m.config.vocab_size

    activations = harvest_activations(base_m, all_tokens, n_layers)

    # Layer-specific mean activations for PMI lifting at QK_LAYER
    E_layer  = harvest_mean_token_activations(
        base_m, all_tokens, QK_LAYER, vocab)

    del base_m; gc.collect()

    # ── PMI matrix (must be computed BEFORE harvest_pmi_activation) ──────────
    PMI = compute_pmi_matrix(all_tokens_list, PMI_WINDOW, PMI_MIN_COUNT, vocab)

    # ── PMI-activation bilinear form (new primary G operator) ────────────────
    # M = (1/N) Σ_{(i,j) in window} PMI[tok_i, tok_j] * x_i.T @ x_j
    print("\n  Computing g_pmi_activation bilinear form...")
    base_m2 = fresh_model()
    M_pmi_act = harvest_pmi_activation(
        base_m2, all_tokens, QK_LAYER, PMI,
        n_chunks=N_CHUNKS, chunk_len=CHUNK_LENGTH,
        pmi_window=PMI_WINDOW, device=DEVICE
    )
    del base_m2; gc.collect()

    # ── Original weights (for norm targets and Procrustes) ───────────────────
    ref_m = fresh_model()
    w_q_orig, w_k_orig, w_v_orig = read_weights(ref_m, QK_LAYER, n_embd)
    _,        _,        w_v0_orig = read_weights(ref_m, V_LAYER,  n_embd)
    del ref_m; gc.collect()

    X_v  = activations[V_LAYER]

    all_results  = []
    surgery_meta = {}

    # =========================================================================
    # SECTION 0 — Baseline and identity checks
    # =========================================================================
    print(f"\n{'='*65}")
    print("Baseline and controls")
    print(f"{'='*65}")

    print("\n  Measuring unmodified baseline...")
    bm = fresh_model()
    wt2_base                     = measure_wt2_ppl(bm, tokenizer)
    lmda_ppl_base, lmda_acc_base = measure_lambada(bm, tokenizer)
    del bm; gc.collect()
    print(f"  BASELINE  WT2={wt2_base}  LMDA_PPL={lmda_ppl_base}  "
          f"LMDA_ACC={lmda_acc_base}%")

    all_results.append({
        "name": "baseline", "section": "control", "target": "none",
        "wt2_ppl": wt2_base, "wt2_delta_pct": 0.0,
        "lambada_ppl": lmda_ppl_base, "lambada_delta_pct": 0.0,
        "lambada_acc": lmda_acc_base, "saved_to": None,
    })

    for exp_name, layer, wq, wk, wv in [
        ("wv_identity",  V_LAYER,   None,          None,          w_v0_orig.copy()),
        ("g_identity",   QK_LAYER,  w_q_orig.copy(), w_k_orig.copy(), None),
    ]:
        r = run_experiment(exp_name, layer, n_embd, tokenizer,
                           wq, wk, wv, save_dir, wt2_base, lmda_ppl_base)
        identity_check(r, wt2_base)
        all_results.append({**r, "section": "control"})

    # Random nulls
    for exp_name, layer, op_fn in [
        ("wv_random_null", V_LAYER,  lambda: (None, None, op_wv_random(w_v0_orig))),
        ("g_random_null",  QK_LAYER, lambda: (*op_g_random(w_q_orig, w_k_orig), None)),
    ]:
        wq, wk, wv = op_fn()
        r = run_experiment(exp_name, layer, n_embd, tokenizer,
                           wq, wk, wv, save_dir, wt2_base, lmda_ppl_base)
        all_results.append({**r, "section": "control"})

    # =========================================================================
    # SECTION 1 — W_v surgeries
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"  W_v surgeries  (Layer {V_LAYER})")
    print(f"  Theoretical claim: W_v is an isometric operator.")
    print(f"  Any isometry close to W_v's principal axes should work.")
    print(f"{'='*65}")

    wv_ops = [
        # (name, operator, description)
        ("wv_polar_q",
         lambda: op_wv_polar_q(w_v0_orig),
         "W_v's own polar Q.  Upper bound for isometry replacement."),
        ("wv_pca_proc",
         lambda: op_wv_pca_proc(X_v, w_v0_orig),
         "PCA + Procrustes.  Data-derived isometry.  Phase 3 result."),
        ("wv_pca_raw",
         lambda: op_wv_pca_raw(X_v, w_v0_orig),
         "Raw PCA, no Procrustes.  Orientation cost baseline."),
    ]

    for name, op_fn, desc in wv_ops:
        print(f"\n  Computing {name}:  {desc}")
        W_v_new = op_fn()
        sv = np.linalg.svd(W_v_new, compute_uv=False)
        surgery_meta[name] = {"sv_max": float(sv[0]), "sv_min": float(sv[-1])}
        r = run_experiment(name, V_LAYER, n_embd, tokenizer,
                           None, None, W_v_new,
                           save_dir, wt2_base, lmda_ppl_base)
        all_results.append({**r, "section": "wv_surgery", "target": "wv",
                             "layer": V_LAYER})

    # =========================================================================
    # SECTION 2 — G surgeries
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"  G surgeries  (Layer {QK_LAYER})")
    print(f"{'='*65}")

    # ── G alignment diagnostics before surgery ───────────────────────────────
    G_true_ref = w_q_orig.astype(np.float64) @ w_k_orig.astype(np.float64).T
    alignment_results = run_g_alignment_diagnostics(
        G_true_ref.astype(np.float32),
        PMI, E_layer           # E_layer = correct input space for layer QK_LAYER
    )
    surgery_meta["g_alignment"] = alignment_results

    g_ops = [
        # ── Without Procrustes: raw PMI structure, orientation not corrected ──
        # Analogous to wv_pca_raw. Tests statistical structure alone.
        ("g_pmi_activation",
         lambda: op_g_pmi_activation(M_pmi_act, w_q_orig, w_k_orig),
         "PMI-weighted all-pairs activation outer products. No Procrustes. "
         "Analogous to wv_pca_raw."),

        ("g_pmi_layer",
         lambda: op_g_pmi(PMI, E_layer, w_q_orig, w_k_orig, "g_pmi_layer"),
         "PMI @ E_layer (mean per-token activations). No Procrustes. "
         "Comparison: contextual (g_pmi_activation) vs mean-per-token."),

        # ── With Procrustes: PMI structure + best achievable orientation ──────
        # Analogous to wv_pca_proc. Gap vs no-Procrustes = orientation cost.
        # Remaining gap after Procrustes = irreducible amplification gap.
        ("g_pmi_activation_proc",
         lambda: op_g_pmi_activation_proc(M_pmi_act, w_q_orig, w_k_orig),
         "PMI-activation + row-space Procrustes toward G_true. "
         "Analogous to wv_pca_proc. Measures orientation cost for G."),

        ("g_pmi_layer_proc",
         lambda: op_g_pmi_proc(PMI, E_layer, w_q_orig, w_k_orig, "g_pmi_layer_proc"),
         "PMI @ E_layer + Procrustes. Mean-activation version with orientation assist."),
    ]


    for name, op_fn, desc in g_ops:
        print(f"\n  Computing {name}:  {desc}")
        W_q_new, W_k_new, meta = op_fn()
        surgery_meta[name] = meta
        r = run_experiment(name, QK_LAYER, n_embd, tokenizer,
                           W_q_new, W_k_new, None,
                           save_dir, wt2_base, lmda_ppl_base)
        all_results.append({**r, "section": "g_surgery", "target": "g",
                             "layer": QK_LAYER})

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*75}")
    print("  COMPLETE RESULTS")
    print(f"{'='*75}")

    for sec_title, sec_key in [
        ("CONTROLS",              "control"),
        (f"W_v  (Layer {V_LAYER}) — isometric operator",  "wv_surgery"),
        (f"G    (Layer {QK_LAYER}) — representation-space PMI", "g_surgery"),
    ]:
        rows = [r for r in all_results
                if r.get("section") == sec_key or r["name"] == "baseline"]
        if sec_key != "control":
            rows = [r for r in all_results if r.get("section") == sec_key]
            # Add baseline for reference
            rows = [all_results[0]] + rows

        print(f"\n  ── {sec_title} ──")
        print(f"  {'Name':<24} {'WT2':>7}  {'WT2Δ%':>7}  "
              f"{'LMDA_PPL':>10}  {'LMDAδ%':>7}  {'LMDA_ACC':>9}")
        print(f"  {'─'*24} {'─'*7}  {'─'*7}  {'─'*10}  {'─'*7}  {'─'*9}")

        for r in rows:
            wt2  = f"{r['wt2_ppl']:.2f}"            if r["wt2_ppl"] else "N/A"
            dw   = f"{r['wt2_delta_pct']:+.1f}%"    if r["wt2_delta_pct"] is not None else "—"
            lp   = f"{r['lambada_ppl']:.2f}"         if r["lambada_ppl"] else "N/A"
            dl   = f"{r['lambada_delta_pct']:+.1f}%" if r.get("lambada_delta_pct") is not None else "—"
            la   = f"{r['lambada_acc']:.1f}%"        if r["lambada_acc"] else "N/A"
            print(f"  {r['name']:<24} {wt2:>7}  {dw:>7}  {lp:>10}  {dl:>7}  {la:>9}")

    # ── Key comparisons ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  KEY COMPARISONS")
    print(f"{'='*65}")

    def get(xname, field):
        for xr in all_results:
            if xr["name"] == xname:
                return xr.get(field)
        return None

    # W_v orientation cost
    pq_wt2 = get("wv_polar_q",   "wt2_delta_pct")
    pp_wt2 = get("wv_pca_proc",  "wt2_delta_pct")
    pr_wt2 = get("wv_pca_raw",   "wt2_delta_pct")
    pq_lm  = get("wv_polar_q",   "lambada_delta_pct")
    pp_lm  = get("wv_pca_proc",  "lambada_delta_pct")
    pr_lm  = get("wv_pca_raw",   "lambada_delta_pct")
    if pq_wt2 is not None and pp_wt2 is not None:
        print(f"\n  W_v orientation cost (Procrustes effect):")
        print(f"     wv_pca_raw   WT2={pr_wt2:+.1f}%  LAMBADA={pr_lm:+.1f}%  (no Procrustes)")
        print(f"     wv_pca_proc  WT2={pp_wt2:+.1f}%  LAMBADA={pp_lm:+.1f}%  (with Procrustes)")
        print(f"     wv_polar_q   WT2={pq_wt2:+.1f}%  LAMBADA={pq_lm:+.1f}%  (ablation — W_v's own Q)")
        wv_orient = (pr_lm - pp_lm) if pr_lm and pp_lm else None
        if wv_orient is not None:
            print(f"     Orientation cost for W_v = {wv_orient:+.1f}pp LAMBADA")

    # G orientation cost (the symmetric experiment)
    pa_lm  = get("g_pmi_activation",      "lambada_delta_pct")
    pac_lm = get("g_pmi_activation_proc",  "lambada_delta_pct")
    pl_lm  = get("g_pmi_layer",           "lambada_delta_pct")
    plc_lm = get("g_pmi_layer_proc",      "lambada_delta_pct")
    if pa_lm is not None:
        print(f"\n  G orientation cost (Procrustes effect — symmetric with W_v above):")
        print(f"     g_pmi_activation       LAMBADA={pa_lm:+.1f}%  (no Procrustes)")
        if pac_lm is not None:
            g_orient = pa_lm - pac_lm
            print(f"     g_pmi_activation_proc  LAMBADA={pac_lm:+.1f}%  (with Procrustes)")
            print(f"     Orientation cost for G = {g_orient:+.1f}pp LAMBADA")
            if pac_lm > 5:
                print(f"     Remaining gap after Procrustes = {pac_lm:+.1f}%")
                print(f"     → This is the irreducible amplification gap (5.89x training effect)")
            else:
                print(f"     → Near-zero remaining gap: PMI finds the right orientation")
        if pl_lm is not None:
            print(f"     g_pmi_layer            LAMBADA={pl_lm:+.1f}%  (no Procrustes)")
        if plc_lm is not None:
            print(f"     g_pmi_layer_proc       LAMBADA={plc_lm:+.1f}%  (with Procrustes)")

    # Dissociation summary
    print(f"\n  DISSOCIATION CHECK:")
    best_wv_lm = min(
        (get(n, "lambada_delta_pct") or 0)
        for n in ["wv_polar_q", "wv_pca_proc"]
    )
    best_g_lm = min(
        v for n in ["g_pmi_activation", "g_pmi_layer",
                    "g_pmi_activation_proc", "g_pmi_layer_proc"]
        if (v := get(n, "lambada_delta_pct")) is not None
    )
    print(f"  Best W_v LAMBADA Δ = {best_wv_lm:+.1f}%  "
          f"({'✓ improves' if best_wv_lm < 0 else '✗ degrades'})")
    print(f"  Best G   LAMBADA Δ = {best_g_lm:+.1f}%  "
          f"({'✓ all operators damage LAMBADA' if best_g_lm > 20 else 'weaker than expected'})")

    # =========================================================================
    # SAVE
    # =========================================================================
    output = {
        "meta": {
            "model": MODEL_NAME, "timestamp": ts, "device": DEVICE,
            "n_chunks": N_CHUNKS, "chunk_length": CHUNK_LENGTH,
            "v_layer": V_LAYER, "qk_layer": QK_LAYER,
            "pmi_window": PMI_WINDOW, "pmi_min_count": PMI_MIN_COUNT,
            "reg_eps": REG_EPS,
            "baseline_wt2_ppl": wt2_base,
            "baseline_lambada_ppl": lmda_ppl_base,
            "baseline_lambada_acc": lmda_acc_base,
        },
        "theoretical_basis": {
            "Wv_claim": "W_v is an isometric operator. Polar rotation Q carries "
                        "all functional information. Training left W_v near-random-"
                        "orthogonal (0 MP outliers in layers 7-11). Any isometry "
                        "close to W_v's principal axes should approximate its function.",
            "G_claim":  "G implements representation-space PMI. PMI predicts G's "
                        "sign structure (indefinite signature, psd_frac 0.47). "
                        "Rank correlation |G_eigs| vs |G_pmi_eigs| = 0.96. Training "
                        "amplifies PMI directions by 5.56x on average. The 5.56x "
                        "amplification is what statistical operators cannot recover.",
        },
        "surgery_meta":  surgery_meta,
        "results":       all_results,
    }

    json_path = os.path.join(save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n  Results saved → {json_path}")
    print(f"  Models saved  → {save_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()