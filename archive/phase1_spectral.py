"""
Spectral Characterisation
============================================================
Produces a spectral blueprint for GPT-2 model.

What it measures
----------------
  Weight ranks      : Shannon effective rank of W_q, W_k, W_v per layer
  Activation ranks  : Shannon rank + 90%-variance rank of residual-stream
                      activations per layer (harvested from WikiText-2)
  Head diversity    : Std of per-head effective rank within each W_q/W_k/W_v
  Cumulative EVR    : Fraction of activation variance explained by top-k
                      PCA components (used to motivate rank choice in zero-shot surgery phases.)

Outputs (per model, per run)
-------
  results/spectral_{model}_{timestamp}.json    -- all computed data
  results/spectral_{model}_{timestamp}_rank_profile.png
  results/spectral_{model}_{timestamp}_head_diversity.png
  results/spectral_{model}_{timestamp}_cumvar_heatmap.png

Role in the paper
-----------------
  Phase 1 is DIAGNOSTIC.
  It motivates the operator-class hypothesis by showing that:
    (a) attention weights have concentrated spectral structure,
    (b) early layers have lower activation rank than late layers,
    (c) head diversity is low, justifying full-layer replacement in Phase 3.
  Phase 2 (alignment) hardcodes rank = head_dim regardless of Phase 1 output.

Usage
-----
  python spectral_analysis.py                         # all models
  python spectral_analysis.py --no-plots              # JSON only
  python spectral_analysis.py --chunks 64             # faster run
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_TO_RUN = ["gpt2"]
N_CHUNKS       = 128      # number of WikiText-2 chunks for activation harvest
CHUNK_LENGTH   = 1024     # tokens per chunk (GPT-2 hard max is 1024)
VAR_THRESHOLD  = 0.90     # variance threshold for rank_90 computation
TOP_SVS        = 50       # how many singular values to store in JSON
RESULTS_DIR    = "results/phase1_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sns.set_style('whitegrid')

# =============================================================================
# METRICS
# =============================================================================

def shannon_rank(matrix: np.ndarray) -> float:
    """
    Shannon effective rank using squared singular values as probability weights.

    exp( -sum(p_i * log(p_i)) )  where  p_i = sigma_i^2 / sum(sigma_j^2)

    Squared SV normalisation is required (not raw SVs). This gives the entropy
    of the variance distribution, which equals the number of components when
    all singular values are equal and approaches 1 when one dominates.

    Zero singular values are excluded before computing the entropy sum to avoid
    log(0); strictly positive values need no epsilon guard.
    """
    sv = np.linalg.svd(matrix.astype(np.float32), compute_uv=False)
    sv = sv[sv > 0]
    p  = (sv ** 2) / (sv ** 2).sum()
    entropy = float(np.sum(p * np.log(p)))
    return float(np.exp(-entropy))


def rank_90(matrix: np.ndarray, threshold: float = VAR_THRESHOLD):
    """
    Minimum number of PCA components to explain `threshold` fraction of
    variance. Uses numpy SVD directly (identical to sklearn PCA with
    svd_solver='full' but faster and dependency-free).

    Returns (rank, full_evr_array).
    """
    X = matrix.astype(np.float32)
    X = X - X.mean(axis=0)          # centre before SVD (PCA convention)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    var     = S ** 2 / max(X.shape[0] - 1, 1)
    evr     = var / var.sum()
    cumvar  = np.cumsum(evr)
    k       = int(np.searchsorted(cumvar, threshold)) + 1
    k       = min(k, len(evr))
    return k, evr.tolist()


def head_diversity(W: np.ndarray, n_heads: int) -> dict:
    """
    Per-head Shannon effective rank for a weight matrix W [d_model, d_model].

    Each head h occupies columns h*head_dim : (h+1)*head_dim of W.
    Returns mean, std, min, max, and per-head ranks.

    Low std across heads → homogeneous heads → full-layer replacement in
    Layer-wise surgery is justified, no need to treat heads independently.
    """
    d      = W.shape[1]
    hd     = d // n_heads
    ranks  = [shannon_rank(W[:, h * hd:(h + 1) * hd]) for h in range(n_heads)]
    return {
        "per_head_ranks": [round(r, 4) for r in ranks],
        "mean": round(float(np.mean(ranks)), 4),
        "std":  round(float(np.std(ranks)),  4),
        "min":  round(float(np.min(ranks)),  4),
        "max":  round(float(np.max(ranks)),  4),
    }


def top_svs(W: np.ndarray, n: int = TOP_SVS) -> list:
    sv = np.linalg.svd(W.astype(np.float32), compute_uv=False)
    return [round(float(v), 6) for v in sv[:n]]


# =============================================================================
# ACTIVATION HARVEST
# =============================================================================

def harvest_activations(model, layer_idx: int,
                        chunks: list) -> np.ndarray:
    """
    Collect residual-stream activations at the INPUT of transformer block
    `layer_idx` across all chunks, then concatenate into a single matrix
    [N_total_tokens, d_model].

    WHY CONCATENATE BEFORE RANK:
      Per-chunk rank is dominated by within-context correlations and
      underestimates the true population rank. The full concatenated matrix
      gives the correct population-level rank (~100–500 for GPT-2).
      Rank must be computed on the full matrix.

    WHY INPUT NOT OUTPUT:
      The input to block `layer_idx` is the residual stream *before* that
      layer's attention and MLP. This is the signal that W_q, W_k, W_v
      actually receive — the correct distribution for PCA alignment in Phase 2.

    The forward hook is always removed via try/finally to prevent hook
    accumulation if model() raises mid-chunk.
    """
    store  = {}
    handle = model.transformer.h[layer_idx].register_forward_hook(
        lambda m, inp, out: store.update({"act": inp[0].detach().cpu()})
    )
    acts = []
    try:
        model.eval()
        with torch.no_grad():
            for chunk in chunks:
                model(input_ids=chunk)
                acts.append(store["act"].squeeze(0).numpy())
    finally:
        handle.remove()
    return np.concatenate(acts, axis=0).astype(np.float32)


# =============================================================================
# PER-LAYER PROFILING
# =============================================================================

def profile_layer(model, layer_idx: int, chunks: list,
                  n_heads: int) -> dict:
    """
    Compute all metrics for one layer. Returns a dict ready for JSON.
    """
    # Weight matrices from fused c_attn [d_model, 3*d_model]
    W_fused = model.transformer.h[layer_idx].attn.c_attn.weight.data.cpu().numpy()
    d       = W_fused.shape[0]
    w_q, w_k, w_v = W_fused[:, :d], W_fused[:, d:2*d], W_fused[:, 2*d:]

    wr_q = shannon_rank(w_q)
    wr_k = shannon_rank(w_k)
    wr_v = shannon_rank(w_v)
    hd_q = head_diversity(w_q, n_heads)
    hd_k = head_diversity(w_k, n_heads)
    hd_v = head_diversity(w_v, n_heads)
    sv_q = top_svs(w_q)
    sv_k = top_svs(w_k)
    sv_v = top_svs(w_v)

    X     = harvest_activations(model, layer_idx, chunks)
    ar_s  = shannon_rank(X)
    ar_v, evr = rank_90(X)

    # Cumulative EVR at key checkpoints — adapt to model's head_dim
    head_dim = d // n_heads
    evr_arr  = np.array(evr)
    checkpoints = [
        k for k in [10, 32, head_dim, head_dim * 2, head_dim * 4]
        if k <= len(evr_arr)
    ]
    cumvar_at = {
        str(k): round(float(np.sum(evr_arr[:k])), 4)
        for k in checkpoints
    }

    return {
        "layer_idx": layer_idx,
        "weight_ranks": {
            "q_shannon": round(wr_q, 3),
            "k_shannon": round(wr_k, 3),
            "v_shannon": round(wr_v, 3),
        },
        "weight_singular_values_top50": {
            "q": sv_q, "k": sv_k, "v": sv_v
        },
        "activation_ranks": {
            "shannon":  round(ar_s, 3),
            "var90":    ar_v,
            "explained_variance_ratio_top100": [round(v, 6) for v in evr[:100]],
            "cumvar_at": cumvar_at,
        },
        "head_diversity": {"q": hd_q, "k": hd_k, "v": hd_v},
    }


# =============================================================================
# PLOTTING  (separate from computation, can re-plot from saved JSON)
# =============================================================================

def plot_rank_profile(blueprint: dict, save_path: str):
    layers  = [r["layer_idx"] for r in blueprint["layers"]]
    wr_q    = [r["weight_ranks"]["q_shannon"]    for r in blueprint["layers"]]
    wr_k    = [r["weight_ranks"]["k_shannon"]    for r in blueprint["layers"]]
    wr_v    = [r["weight_ranks"]["v_shannon"]    for r in blueprint["layers"]]
    ar_s    = [r["activation_ranks"]["shannon"]  for r in blueprint["layers"]]
    ar_v90  = [r["activation_ranks"]["var90"]    for r in blueprint["layers"]]
    head_dim = blueprint["meta"]["head_dim"]
    model    = blueprint["meta"]["model_name"].upper()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
    fig.suptitle(f"Phase 1: Spectral Rank Profile — {model}",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(layers, wr_q, "o-", label="W_Q (Shannon)", color="#2980b9", lw=2)
    ax.plot(layers, wr_k, "s-", label="W_K (Shannon)", color="#8e44ad", lw=2)
    ax.plot(layers, wr_v, "^-", label="W_V (Shannon)", color="#27ae60", lw=2)
    ax.plot(layers, ar_s, "D--", label="Activation (Shannon)",
            color="#e67e22", lw=2)
    ax.axhline(y=head_dim, color="red", linestyle=":",
               lw=1.5, label=f"head_dim={head_dim}")
    ax.set_title("Shannon Effective Rank", fontsize=12)
    ax.set_xlabel("Layer"); ax.set_ylabel("Effective Rank")
    ax.legend(fontsize=8, frameon=True)

    ax = axes[1]
    ax.plot(layers, ar_v90, "D--", label=f"Activation ({int(VAR_THRESHOLD*100)}% Var)",
            color="#e67e22", lw=2)
    ax.axhline(y=head_dim, color="red", linestyle=":",
               lw=1.5, label=f"head_dim={head_dim}")
    ax.set_title(f"Variance Threshold Rank ({int(VAR_THRESHOLD*100)}%)", fontsize=12)
    ax.set_xlabel("Layer"); ax.set_ylabel("# Components")
    ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_head_diversity(blueprint: dict, save_path: str):
    layers  = [r["layer_idx"]                  for r in blueprint["layers"]]
    hd_q    = [r["head_diversity"]["q"]["std"] for r in blueprint["layers"]]
    hd_k    = [r["head_diversity"]["k"]["std"] for r in blueprint["layers"]]
    hd_v    = [r["head_diversity"]["v"]["std"] for r in blueprint["layers"]]
    model   = blueprint["meta"]["model_name"].upper()

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(layers, hd_q, "o-", label="Q diversity (std)", color="#2980b9", lw=2)
    ax.plot(layers, hd_k, "s-", label="K diversity (std)", color="#8e44ad", lw=2)
    ax.plot(layers, hd_v, "^-", label="V diversity (std)", color="#27ae60", lw=2)
    ax.axhline(y=0, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.set_title(
        f"Head Diversity Proxy — {model}\n"
        "Low std → homogeneous heads → full-layer replacement justified",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Layer"); ax.set_ylabel("Std of Per-Head Effective Rank")
    ax.legend(fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cumvar_heatmap(blueprint: dict, save_path: str):
    """
    Heatmap rows = layers, columns = top-k PCA components.
    Checkpoints are inferred from what was stored in the JSON.
    """
    checkpoints = sorted(
        int(k) for k in blueprint["layers"][0]["activation_ranks"]["cumvar_at"]
    )
    cumvar_matrix = np.array([
        [r["activation_ranks"]["cumvar_at"][str(k)] for k in checkpoints]
        for r in blueprint["layers"]
    ])
    n_layers = len(blueprint["layers"])
    model    = blueprint["meta"]["model_name"].upper()

    fig, ax = plt.subplots(figsize=(8, max(6, n_layers // 3)), dpi=150)
    sns.heatmap(
        cumvar_matrix,
        annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=[f"top-{k}" for k in checkpoints],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        vmin=0, vmax=1, ax=ax
    )
    ax.set_title(
        f"Cumulative Explained Variance — {model}\n"
        "Fraction of activation variance by top-k PCA components",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Top-k Components"); ax.set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE  (one model at a time)
# =============================================================================

def run_model(model_name: str, n_chunks: int, chunk_length: int,
              make_plots: bool = True) -> str:
    """
    Run full analysis for one model.

    Parameters
    ----------
    model_name   : HuggingFace model identifier.
    n_chunks     : Number of WikiText-2 chunks to harvest activations from.
    chunk_length : Tokens per chunk (must be <= 1024 for GPT-2).
    make_plots   : Whether to generate PNG figures alongside the JSON.

    Returns
    -------
    str : Path to the saved JSON file.
    """
    print(f"\n{'=' * 65}")
    print(f"  Phase 1 Spectral Analysis — {model_name}")
    print(f"{'=' * 65}")

    print(f"  Loading {model_name}...")
    hf_model  = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    hf_model.eval()

    n_layers = hf_model.config.n_layer
    n_embd   = hf_model.config.n_embd
    n_heads  = hf_model.config.n_head
    head_dim = n_embd // n_heads

    print(f"  Config: {n_layers} layers | {n_embd} hidden | "
          f"{n_heads} heads | {head_dim} head_dim")

    # Tokenise WikiText-2 in batches to avoid large single-call allocations
    print(f"  Loading WikiText-2 ({n_chunks} chunks x {chunk_length} tokens)...")
    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(t for t in dataset["text"] if t.strip())

    CHAR_BATCH = 100_000
    parts = [
        tokenizer(
            full_text[i:i + CHAR_BATCH],
            return_tensors="pt", truncation=False, add_special_tokens=False
        )["input_ids"].squeeze(0)
        for i in range(0, len(full_text), CHAR_BATCH)
    ]
    all_tokens = torch.cat(parts, dim=0)

    max_start = all_tokens.shape[0] - chunk_length
    if max_start <= 0:
        raise ValueError("Corpus too small for the requested chunk length.")
    starts = np.linspace(0, max_start, n_chunks, dtype=int)
    chunks = [
        all_tokens[int(s):int(s) + chunk_length].unsqueeze(0).to(DEVICE)
        for s in starts
    ]
    total_tokens = n_chunks * chunk_length
    print(f"  -> {n_chunks} x {chunk_length} = {total_tokens:,} tokens")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blueprint = {
        "meta": {
            "model_name":    model_name,
            "n_embd":        n_embd,
            "n_heads":       n_heads,
            "head_dim":      head_dim,
            "n_layers":      n_layers,
            "n_chunks":      n_chunks,
            "chunk_length":  chunk_length,
            "total_tokens":  total_tokens,
            "var_threshold": VAR_THRESHOLD,
            "dataset":       "wikitext-2-raw-v1 train",
            "device":        DEVICE,
            "timestamp":     timestamp,
        },
        "layers": []
    }

    # Per-layer computation
    for li in tqdm(range(n_layers), desc="Layers"):
        layer_data = profile_layer(hf_model, li, chunks, n_heads)
        blueprint["layers"].append(layer_data)

        wr   = layer_data["weight_ranks"]
        ar   = layer_data["activation_ranks"]
        hd_v = layer_data["head_diversity"]["v"]
        tqdm.write(
            f"  L{li:02d} | "
            f"Wq={wr['q_shannon']:.1f} Wk={wr['k_shannon']:.1f} Wv={wr['v_shannon']:.1f} | "
            f"Act_S={ar['shannon']:.1f} Var90={ar['var90']} | "
            f"HDiv_v(std)={hd_v['std']:.3f}"
        )

    # Save JSON — timestamped, never overwrites
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    base      = f"{RESULTS_DIR}/spectral_{safe_name}_{timestamp}"
    json_path = f"{base}.json"

    with open(json_path, "w") as f:
        json.dump(blueprint, f, indent=4)
    print(f"\n  JSON saved -> {json_path}  "
          f"({os.path.getsize(json_path) / 1024:.1f} KB)")

    if make_plots:
        print("  Generating plots...")
        plot_rank_profile(blueprint,   f"{base}_rank_profile.png")
        plot_head_diversity(blueprint, f"{base}_head_diversity.png")
        plot_cumvar_heatmap(blueprint, f"{base}_cumvar_heatmap.png")

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json_path


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 spectral analysis of GPT-2 family attention weights"
    )
    parser.add_argument(
        "--chunks", type=int, default=N_CHUNKS,
        help=f"Number of WikiText-2 chunks (default: {N_CHUNKS})"
    )
    parser.add_argument(
        "--chunk-length", type=int, default=CHUNK_LENGTH,
        help=f"Tokens per chunk, max 1024 (default: {CHUNK_LENGTH})"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (JSON only)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.chunks <= 0:
        raise ValueError(f"--chunks must be a positive integer, got {args.chunks}")

    chunk_length = args.chunk_length
    if chunk_length > 1024:
        print(f"  Warning: --chunk-length {chunk_length} exceeds GPT-2 max "
              f"(1024); clamped to 1024.")
        chunk_length = 1024
    if chunk_length <= 0:
        raise ValueError(f"--chunk-length must be a positive integer, "
                         f"got {chunk_length}")

    print(f"\nPhase 1: Spectral Characterisation")
    print(f"  Chunks       : {args.chunks} x {chunk_length} = "
          f"{args.chunks * chunk_length:,} tokens")
    print(f"  Device       : {DEVICE}")
    print(f"  Output dir   : {RESULTS_DIR}/")

    saved = []
    for model_name in args.models:
        try:
            path = run_model(model_name,
                             n_chunks=args.chunks,
                             chunk_length=chunk_length,
                             make_plots=not args.no_plots)
            saved.append(path)
        except Exception as e:
            print(f"\n  ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 65}")
    print(f"Phase 1 complete. Saved {len(saved)} file(s):")
    for p in saved:
        print(f"  {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()