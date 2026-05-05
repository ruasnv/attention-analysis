"""
plot_near_isometry.py
=====================
Plots the singular value spectra of W_v, W_o, and W_ov = W_o @ W_v
from pre-trained GPT-2 to demonstrate the near-isometry of W_v.

Averaged across all 12 layers for robustness.
Run from your project directory — requires transformers and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from transformers import GPT2LMHeadModel

# =============================================================================
# STYLE — matches the poster palette
# =============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 10,
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.97,
    "legend.edgecolor": "#E2E8F0",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "lines.linewidth": 1.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#1E293B",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": "#1E293B",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

# Poster palette
COLOR_WV  = "#0D9488"   # teal  — our focus matrix
COLOR_WO  = "#475569"   # slate
COLOR_OV  = "#1E293B"   # near-black


def _clean_ax(ax):
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(length=2.5, width=0.5, colors="#475569")
    ax.grid(False)


def extract_ov_matrices(model):
    """
    Extract W_v and W_o from every attention layer of GPT-2.
    GPT-2 uses Conv1D with transposed weights: shape is (nx, nf).
    c_attn fused: [d, 3d] → columns are [W_q | W_k | W_v]
    c_proj is W_o: [d, d]
    """
    wv_list, wo_list = [], []

    for block in model.transformer.h:
        # c_attn weight: shape (d, 3d) in Conv1D convention
        W_attn = block.attn.c_attn.weight.data  # (768, 2304)
        d = W_attn.shape[0]
        W_v = W_attn[:, 2*d:]                   # (768, 768)

        # c_proj weight: shape (d, d)
        W_o = block.attn.c_proj.weight.data      # (768, 768)

        wv_list.append(W_v.float().cpu())
        wo_list.append(W_o.float().cpu())

    return wv_list, wo_list


def compute_mean_singular_values(matrix_list):
    """
    Compute singular values for each matrix, normalise by max,
    then average across layers. Returns array of shape (d,).
    """
    all_svs = []
    for W in matrix_list:
        svs = torch.linalg.svdvals(W).numpy()
        svs_norm = svs / svs.max()   # normalise so max = 1
        all_svs.append(svs_norm)
    return np.mean(all_svs, axis=0)


def main():
    print("Loading pre-trained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("Extracting OV matrices from 12 layers...")
    wv_list, wo_list = extract_ov_matrices(model)

    # Compute OV = W_o @ W_v for each layer
    wov_list = [wo @ wv for wo, wv in zip(wo_list, wv_list)]

    print("Computing mean singular value spectra...")
    svs_wv  = compute_mean_singular_values(wv_list)
    svs_wo  = compute_mean_singular_values(wo_list)
    svs_wov = compute_mean_singular_values(wov_list)

    d = len(svs_wv)
    idx = np.arange(1, d + 1)

    # ==========================================================================
    # PLOT
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    ax.plot(idx, svs_wo,  color=COLOR_WO,  linewidth=1.4,
            label=r"$W_o$ (output projection)", alpha=0.85)
    ax.plot(idx, svs_wov, color=COLOR_OV,  linewidth=1.4,
            linestyle=(0, (4, 2)),
            label=r"$W_o W_v$ (OV circuit)", alpha=0.85)
    ax.plot(idx, svs_wv,  color=COLOR_WV,  linewidth=2.0,
            label=r"$W_v$ (value matrix) — near-isometric")

    # Flat reference line for perfect isometry
    ax.axhline(1.0, color="#94A3B8", linewidth=0.7,
               linestyle=":", alpha=0.6)
    ax.text(d * 0.98, 1.015, r"perfect isometry",
            fontsize=7.5, color="#94A3B8", ha="right")

    # Annotate the flat W_v region
    ax.annotate(r"$W_v \approx I$ (flat spectrum)",
                xy=(d * 0.5, svs_wv[int(d * 0.5)]),
                xytext=(d * 0.45, 0.72),
                fontsize=8,
                color=COLOR_WV,
                arrowprops=dict(
                    arrowstyle="->",
                    color=COLOR_WV,
                    lw=0.8,
                ))

    ax.set_xlabel(r"Singular Value Index")
    ax.set_ylabel(r"Normalised Singular Value")
    ax.set_title(
        r"Pre-trained GPT-2: Singular Value Spectra of OV Circuit",
        fontsize=10
    )
    ax.set_xlim(1, d)
    ax.set_ylim(0, 1.12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(128))
    ax.legend(loc="lower left", fontsize=8)
    _clean_ax(ax)

    fig.tight_layout()
    fig.savefig("near_isometry_wv.pdf", dpi=300)
    fig.savefig("near_isometry_wv.png", dpi=300)
    plt.close(fig)
    print("  → near_isometry_wv.pdf / .png")

    # Print summary stats
    print(f"\nSummary (mean normalised singular values):")
    print(f"  W_v  — mean: {svs_wv.mean():.4f}  std: {svs_wv.std():.4f}  "
          f"(ratio max/min: {svs_wv.max()/svs_wv.min():.2f})")
    print(f"  W_o  — mean: {svs_wo.mean():.4f}  std: {svs_wo.std():.4f}  "
          f"(ratio max/min: {svs_wo.max()/svs_wo.min():.2f})")
    print(f"  W_ov — mean: {svs_wov.mean():.4f}  std: {svs_wov.std():.4f}  "
          f"(ratio max/min: {svs_wov.max()/svs_wov.min():.2f})")
    print("\nA flat W_v spectrum (low std, max/min near 1) confirms near-isometry.")


if __name__ == "__main__":
    main()