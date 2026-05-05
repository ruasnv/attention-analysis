"""
Intrinsic structure analysis of GPT-2 models attention weights.

Runs all six analysis sections across one or more models and saves a
separate, fully self-contained JSON result file per model.

Sections
--------
  A        : Orthogonality, SV uniformity, head-level W_v analysis
  B        : Polar decomposition  (W = Q @ S)
  C        : Marchenko-Pastur deviation  (trained signal mass)
  D        : Gram matrix verification  (Teo & Nguyen per-head bridge)
  E        : Q-surgery experiment  (polar factor functional test)
  F        : Frame bounds  (tight frame / isometry analysis)
  G        : Metric tensor  (G = W_q^T W_k PSD structure)

Usage
-----
  python phase0_geometric_analysis.py                    # runs all models
  python phase0_geometric_analysis.py --models gpt2      # single model
  python phase0_geometric_analysis.py --skip-sections CD # skip slow sections

Results
-------
  results/geometric_<model_name>_<timestamp>.json
"""

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from scipy.linalg import polar
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.helpers import save_json_results
warnings.filterwarnings("ignore")

MODELS_TO_RUN = ["gpt2"]

RUN_SECTIONS = {
    "A":      True,
    "B":      True,
    "C":      True,
    "D":      True,
    "E":      True,
    "F":      True,
    "G":      True,
}

N_CHUNKS       = 64
CHUNK_LEN      = 512
EVAL_TOKENS    = 1024
TOP_K_EIGS     = 64
N_RANDOM       = 20

RESULTS_DIR    = "phase0_results"

@dataclass
class ModelConfig:
    model_name: str
    n_layers:   int
    d_model:    int
    n_heads:    int
    head_dim:   int

    @classmethod
    def from_hf(cls, model_name: str, hf_config) -> "ModelConfig":
        d_model  = hf_config.n_embd
        n_heads  = hf_config.n_head
        n_layers = hf_config.n_layer
        head_dim = d_model // n_heads
        return cls(model_name=model_name, n_layers=n_layers,
                   d_model=d_model, n_heads=n_heads, head_dim=head_dim)


def header(title: str):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)


def get_weights(model, layer: int, cfg: ModelConfig):
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    d = cfg.d_model
    return W[:, :d].copy(), W[:, d:2*d].copy(), W[:, 2*d:].copy()


def sv_metrics(W: np.ndarray) -> dict:
    sv      = np.linalg.svd(W, compute_uv=False)
    orth    = np.linalg.norm(W.T @ W - np.eye(W.shape[1]), 'fro')
    cv      = float(np.std(sv) / np.mean(sv))
    var     = sv ** 2
    cumvar  = np.cumsum(var) / var.sum()
    k90     = int(np.searchsorted(cumvar, 0.90)) + 1
    p       = sv / sv.sum()
    p       = p[p > 0]
    eff_rank = float(np.exp(-1.0 * np.sum(p * np.log(p))))
    return {
        "orth_dev": float(orth), "sv_cv": cv, "k90": k90,
        "eff_rank": eff_rank,
        "sv_max":   float(sv[0]), "sv_min": float(sv[-1]),
        "sv":       sv,
    }


def random_sv_baseline(shape, n: int = N_RANDOM) -> dict:
    rows = {k: [] for k in ["orth_dev", "sv_cv", "k90", "eff_rank"]}
    for _ in range(n):
        m = sv_metrics(np.random.randn(*shape))
        for k in rows:
            rows[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in rows.items()}


def analyze_heads(W_v: np.ndarray, cfg: ModelConfig) -> List[dict]:
    results = []
    for h in range(cfg.n_heads):
        W_h  = W_v[:, h * cfg.head_dim : (h + 1) * cfg.head_dim]
        sv_h = np.linalg.svd(W_h, compute_uv=False)
        results.append({
            "head":     h,
            "orth_dev": float(np.linalg.norm(W_h.T @ W_h - np.eye(cfg.head_dim), 'fro')),
            "sv_cv":    float(np.std(sv_h) / np.mean(sv_h)),
        })
    return results


def joint_qk_metrics(W_q: np.ndarray, W_k: np.ndarray) -> dict:
    P = W_q @ W_k.T
    m = sv_metrics(P)
    m["asymmetry"] = float(
        np.linalg.norm(W_q - W_k, 'fro') / np.linalg.norm(W_q, 'fro')
    )
    return m


def load_wikitext_tokens(tokenizer, split: str = "train"):
    from datasets import load_dataset
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tokenizer.encode(text, return_tensors="pt")


def compute_wt2_ppl(model, tokenizer, device: str,
                    max_tokens: int = EVAL_TOKENS) -> float:
    from datasets import load_dataset
    ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text  = "\n\n".join(ds["text"])
    toks  = tokenizer.encode(text, return_tensors="pt")[:, :max_tokens].to(device)
    model.eval()
    with torch.no_grad():
        loss = model(toks, labels=toks).loss.item()
    return float(np.exp(loss))


def run_section_a(model, cfg: ModelConfig) -> dict:
    header(f"SECTION A — ORTHOGONALITY & SV ANALYSIS  [{cfg.model_name}]")

    rand_full = random_sv_baseline((cfg.d_model, cfg.d_model))
    rand_head = random_sv_baseline((cfg.d_model, cfg.head_dim))

    print(f"  Random [{cfg.d_model}x{cfg.d_model}] baseline:"
          f"  orth_dev={rand_full['orth_dev']:.2f}  sv_cv={rand_full['sv_cv']:.4f}")
    print(f"  Random [{cfg.d_model}x{cfg.head_dim}] head baseline:"
          f"  orth_dev={rand_head['orth_dev']:.2f}  sv_cv={rand_head['sv_cv']:.4f}")

    layer_data = {}
    for layer in range(cfg.n_layers):
        W_q, W_k, W_v = get_weights(model, layer, cfg)
        mv  = sv_metrics(W_v)
        mq  = sv_metrics(W_q)
        mk  = sv_metrics(W_k)
        jqk = joint_qk_metrics(W_q, W_k)
        heads = analyze_heads(W_v, cfg)
        head_devs = [h["orth_dev"] for h in heads]
        ratio = mv["orth_dev"] / rand_full["orth_dev"]
        print(f"  L{layer:02d}  Wv_orth={mv['orth_dev']:8.2f}  "
              f"Wq_orth={mq['orth_dev']:8.2f}  Wk_orth={mk['orth_dev']:8.2f}  "
              f"QK_orth={jqk['orth_dev']:10.0f}  Wv/rand={ratio:.3f}")
        layer_data[str(layer)] = {
            "Wv_orth":     mv["orth_dev"],
            "Wq_orth":     mq["orth_dev"],
            "Wk_orth":     mk["orth_dev"],
            "QK_orth":     jqk["orth_dev"],
            "QK_asymmetry": jqk["asymmetry"],
            "Wv_sv_cv":    mv["sv_cv"],
            "Wv_k90":      mv["k90"],
            "Wv_eff_rank": mv["eff_rank"],
            "Wv_rand_ratio": ratio,
            "Wv_head_mean_orth": float(np.mean(head_devs)),
            "Wv_head_min_orth":  float(np.min(head_devs)),
            "Wv_head_max_orth":  float(np.max(head_devs)),
        }

    return {
        "random_baseline_full": rand_full,
        "random_baseline_head": rand_head,
        "layers": layer_data,
    }


def run_section_b(model, cfg: ModelConfig) -> dict:
    header(f"SECTION B — POLAR DECOMPOSITION  [{cfg.model_name}]")
    print(f"  {'Layer':<6} {'Matrix':<6} {'Q_orth_dev':>11} {'S_iso_dev':>10} "
          f"{'Q_fidelity':>11} {'stretch_var':>12}")
    print("  " + "-" * 60)

    results = {}
    for layer in range(cfg.n_layers):
        W_q, W_k, W_v = get_weights(model, layer, cfg)
        results[str(layer)] = {}
        for name, W in [("W_v", W_v), ("W_q", W_q), ("W_k", W_k)]:
            Q, S = polar(W)
            Q_orth_dev = float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 'fro'))
            S_norm     = S / (np.linalg.norm(S, 'fro') / np.sqrt(S.shape[0]))
            I_norm     = np.eye(S.shape[0]) / np.sqrt(S.shape[0])
            S_iso_dev  = float(np.linalg.norm(S_norm - I_norm, 'fro'))
            W_unit     = W / np.linalg.norm(W, 'fro')
            Q_unit     = Q / np.linalg.norm(Q, 'fro')
            Q_fidelity = float(np.linalg.norm(Q_unit - W_unit, 'fro'))
            S_eigs     = np.linalg.eigvalsh(S)
            stretch_var = float(np.std(S_eigs) / np.mean(S_eigs))
            results[str(layer)][name] = {
                "Q_orth_dev": Q_orth_dev, "S_iso_dev": S_iso_dev,
                "Q_fidelity": Q_fidelity, "stretch_var": stretch_var,
            }
            print(f"  L{layer:02d}   {name:<6} {Q_orth_dev:>11.6f} {S_iso_dev:>10.4f} "
                  f"{Q_fidelity:>11.4f} {stretch_var:>12.4f}")

    print(f"\n  SUMMARY — Mean across {cfg.n_layers} layers:")
    print(f"  {'Matrix':<6} {'S_iso_dev':>10} {'Q_fidelity':>11} {'stretch_var':>12}")
    for name in ["W_v", "W_q", "W_k"]:
        iso   = np.mean([results[str(l)][name]["S_iso_dev"]   for l in range(cfg.n_layers)])
        fidel = np.mean([results[str(l)][name]["Q_fidelity"]  for l in range(cfg.n_layers)])
        strv  = np.mean([results[str(l)][name]["stretch_var"] for l in range(cfg.n_layers)])
        print(f"  {name:<6} {iso:>10.4f} {fidel:>11.4f} {strv:>12.4f}")

    return results


def _mp_analysis(sv: np.ndarray, n_rows: int, n_cols: int) -> dict:
    gamma      = n_cols / n_rows
    sigma_est  = np.median(sv)
    mad_sigma  = np.median(np.abs(sv - np.median(sv))) / 0.6745
    mp_upper   = (1 + np.sqrt(gamma))**2 * mad_sigma + sigma_est * 0.5
    outlier_sv = sv[sv > mp_upper]
    bulk_sv    = sv[sv <= mp_upper]
    var_total  = np.sum(sv**2)
    n_out      = int(len(outlier_sv))
    out_mass = float(np.divide(np.sum(outlier_sv ** 2), var_total)) if n_out > 0 else 0.0
    bulk_flat  = float(np.std(bulk_sv) / np.mean(bulk_sv)) if len(bulk_sv) > 1 else 0.0
    sig_ratio  = float(out_mass / max(1 - out_mass, 1e-10))
    return {"mp_upper": float(mp_upper), "n_outliers": n_out,
            "outlier_mass": out_mass, "bulk_flatness": bulk_flat,
            "signal_ratio": sig_ratio}


def run_section_c(model, cfg: ModelConfig) -> dict:
    header(f"SECTION C — MARCHENKO-PASTUR DEVIATION  [{cfg.model_name}]")
    print(f"  {'Layer':<6} {'Matrix':<6} {'n_outliers':>11} {'outlier_mass':>13} "
          f"{'signal_ratio':>13} {'bulk_flatness':>14}")
    print("  " + "-" * 65)

    results = {}
    for layer in range(cfg.n_layers):
        W_q, W_k, W_v = get_weights(model, layer, cfg)
        results[str(layer)] = {}
        for name, W in [("W_v", W_v), ("W_q", W_q), ("W_k", W_k)]:
            sv  = np.linalg.svd(W, compute_uv=False)
            res = _mp_analysis(sv, W.shape[0], W.shape[1])
            results[str(layer)][name] = res
            print(f"  L{layer:02d}   {name:<6} {res['n_outliers']:>11d} "
                  f"{res['outlier_mass']:>13.4f} {res['signal_ratio']:>13.4f} "
                  f"{res['bulk_flatness']:>14.4f}")

    print(f"\n  SUMMARY — Mean across {cfg.n_layers} layers:")
    print(f"  {'Matrix':<6} {'n_outliers':>11} {'outlier_mass':>13} {'signal_ratio':>13}")
    for name in ["W_v", "W_q", "W_k"]:
        n_out  = np.mean([results[str(l)][name]["n_outliers"]   for l in range(cfg.n_layers)])
        o_mass = np.mean([results[str(l)][name]["outlier_mass"] for l in range(cfg.n_layers)])
        s_rat  = np.mean([results[str(l)][name]["signal_ratio"] for l in range(cfg.n_layers)])
        print(f"  {name:<6} {n_out:>11.1f} {o_mass:>13.4f} {s_rat:>13.4f}")

    print("\n  Random matrix MP baselines (n=5):")
    for _ in range(5):
        sv  = np.linalg.svd(np.random.randn(cfg.d_model, cfg.d_model),
                             compute_uv=False)
        res = _mp_analysis(sv, cfg.d_model, cfg.d_model)
        print(f"    n_outliers={res['n_outliers']:3d}  "
              f"outlier_mass={res['outlier_mass']:.4f}  "
              f"signal_ratio={res['signal_ratio']:.4f}")

    return results


def run_section_d(model, cfg: ModelConfig, tokens: torch.Tensor) -> dict:
    header(f"SECTION D — GRAM MATRIX VERIFICATION (per-head)  [{cfg.model_name}]")

    top_k = min(TOP_K_EIGS, cfg.head_dim)
    random_base = top_k / CHUNK_LEN

    attn_ph: Dict = {l: {h: [] for h in range(cfg.n_heads)}
                     for l in range(cfg.n_layers)}
    V_ph:    Dict = {l: {h: [] for h in range(cfg.n_heads)}
                     for l in range(cfg.n_layers)}

    def make_hook(layer_idx: int):
        def hook_fn(module, attn_input, output):
            x   = attn_input[0].detach().float()          # [1, T, d_model]
            W   = module.c_attn.weight.data.float()       # [d_model, 3*d_model]
            b   = module.c_attn.bias.data.float()         # [3*d_model]
            qkv = x @ W + b                               # [1, T, 3*d_model]
            T   = x.shape[1]
            d   = cfg.d_model
            hd  = cfg.head_dim
            scale = hd ** 0.5

            Q_a = qkv[0, :, :d]
            K_a = qkv[0, :, d:2*d]
            V_a = qkv[0, :, 2*d:]

            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            mask = mask.to(x.device)

            for h in range(cfg.n_heads):
                sl  = slice(h * hd, (h + 1) * hd)
                Q_h = Q_a[:, sl]
                K_h = K_a[:, sl]
                xV_h = V_a[:, sl]

                scores = (Q_h @ K_h.T / scale).masked_fill(mask, float('-inf'))
                xA_h    = torch.softmax(scores, dim=-1).cpu().numpy()  # [T, T]

                attn_ph[layer_idx][h].append(xA_h)
                V_ph[layer_idx][h].append(xV_h.cpu().numpy())
        return hook_fn

    hooks = [
        model.transformer.h[l].attn.register_forward_hook(make_hook(l))
        for l in range(cfg.n_layers)
    ]

    total = tokens.shape[1]
    starts = np.linspace(0, total - CHUNK_LEN - 1, N_CHUNKS).astype(int).tolist()
    print(f"  Running {N_CHUNKS} forward passes (per-head storage)...")
    model.eval()
    with torch.no_grad():
        for i, start in enumerate(starts):
            chunk = tokens[:, start : start + CHUNK_LEN]
            model(chunk)
            if (i + 1) % 16 == 0:
                print(f"    Chunk {i+1}/{N_CHUNKS}")

    for hk in hooks:
        hk.remove()

    print(f"\n  random baseline = {top_k}/{CHUNK_LEN} = {random_base:.5f}\n")
    print(f"  {'Layer':<6} {'mean_align':>11} {'best_head':>10} "
          f"{'worst_head':>11} {'ratio':>8}  {'heads>rand':>10}")
    print("  " + "-" * 65)

    gram_results = {}
    H = np.eye(CHUNK_LEN) - np.ones((CHUNK_LEN, CHUNK_LEN)) / CHUNK_LEN

    for layer in range(cfg.n_layers):
        head_aligns = []
        for h in range(cfg.n_heads):
            A_h = attn_ph[layer][h][0]
            V_h = V_ph[layer][h][0]

            K_tilde = H @ A_h @ H
            eigs, U_gram = np.linalg.eigh(K_tilde)
            U_gram_top   = U_gram[:, np.argsort(eigs)[::-1][:top_k]]

            U_v, _, _ = np.linalg.svd(V_h, full_matrices=False)
            U_v_top   = U_v[:, :top_k]

            cos2    = (U_gram_top.T @ U_v_top) ** 2
            align_h = float(np.mean(np.diag(cos2)))
            head_aligns.append(align_h)

        mean_align  = float(np.mean(head_aligns))
        ratio       = mean_align / random_base
        n_above     = int(np.sum(np.array(head_aligns) > random_base))
        gram_results[str(layer)] = {
            "mean_align":   mean_align,
            "best_align":   float(np.max(head_aligns)),
            "worst_align":  float(np.min(head_aligns)),
            "ratio":        ratio,
            "n_above_rand": n_above,
            "head_aligns":  [float(a) for a in head_aligns],
        }
        flag = "↑↑ STRONG" if ratio > 3.0 else ("↑ above" if ratio > 1.0 else "")
        print(f"  L{layer:02d}   {mean_align:>11.6f} {np.max(head_aligns):>10.6f} "
              f"{np.min(head_aligns):>11.6f} {ratio:>8.3f}  {n_above:>4}/{cfg.n_heads}  {flag}")

    return {"random_baseline": float(random_base), "layers": gram_results}


def run_section_e(model, tokenizer, cfg: ModelConfig, device: str) -> dict:
    header(f"SECTION E — Q-SURGERY EXPERIMENT  [{cfg.model_name}]")

    baseline_ppl = compute_wt2_ppl(model, tokenizer, device)
    print(f"  Baseline PPL = {baseline_ppl:.4f}")

    print(f"\n  {'Layer':<6} {'Q_ppl':>10} {'S_ppl':>10} "
          f"{'Q_delta%':>10} {'S_delta%':>10}  Q vs S")
    print("  " + "-" * 62)

    results = {}
    for layer in range(cfg.n_layers):
        _, _, W_v = get_weights(model, layer, cfg)
        Q_mat, S_mat = polar(W_v)

        attn = model.transformer.h[layer].attn
        orig = attn.c_attn.weight.data.clone()
        d    = cfg.d_model

        # Q-surgery: replace W_v slice with its unitary polar factor Q
        try:
            patched = orig.clone()
            patched[:, 2*d:] = torch.tensor(Q_mat, dtype=torch.float32)
            attn.c_attn.weight.data.copy_(patched)
            ppl_Q = compute_wt2_ppl(model, tokenizer, device)
        finally:
            attn.c_attn.weight.data.copy_(orig)

        # S-surgery: replace W_v slice with its PSD stretch factor S
        try:
            patched = orig.clone()
            patched[:, 2*d:] = torch.tensor(S_mat, dtype=torch.float32)
            attn.c_attn.weight.data.copy_(patched)
            ppl_S = compute_wt2_ppl(model, tokenizer, device)
        finally:
            attn.c_attn.weight.data.copy_(orig)

        delta_Q = 100.0 * (ppl_Q - baseline_ppl) / baseline_ppl
        delta_S = 100.0 * (ppl_S - baseline_ppl) / baseline_ppl
        results[str(layer)] = {
            "ppl_Q": float(ppl_Q), "ppl_S": float(ppl_S),
            "delta_Q_pct": float(delta_Q), "delta_S_pct": float(delta_S),
        }
        print(f"  L{layer:02d}   {ppl_Q:>10.4f} {ppl_S:>10.4f} "
              f"{delta_Q:>+10.2f}% {delta_S:>+10.2f}%  "
              f"{'Q better' if ppl_Q < ppl_S else 'S better'}")

    q_d   = np.array([results[str(l)]["delta_Q_pct"] for l in range(cfg.n_layers)])
    s_d   = np.array([results[str(l)]["delta_S_pct"] for l in range(cfg.n_layers)])
    early = slice(0, 4)
    late  = slice(cfg.n_layers - 4, cfg.n_layers)

    print(f"\n  Q-surgery: mean={np.mean(q_d):+.2f}%  "
          f"early(L0-3)={np.mean(q_d[early]):+.2f}%  "
          f"late(last 4)={np.mean(q_d[late]):+.2f}%")
    print(f"  S-surgery: mean={np.mean(s_d):+.2f}%  "
          f"early(L0-3)={np.mean(s_d[early]):+.2f}%  "
          f"late(last 4)={np.mean(s_d[late]):+.2f}%")

    return {
        "baseline_ppl": float(baseline_ppl),
        "layers": results,
        "summary": {
            "Q_mean_delta":  float(np.mean(q_d)),
            "Q_early_delta": float(np.mean(q_d[early])),
            "Q_late_delta":  float(np.mean(q_d[late])),
            "S_mean_delta":  float(np.mean(s_d)),
            "S_early_delta": float(np.mean(s_d[early])),
            "S_late_delta":  float(np.mean(s_d[late])),
        }
    }


def run_section_f(model, cfg: ModelConfig) -> dict:
    header(f"SECTION F — FRAME BOUNDS  [{cfg.model_name}]")

    rand_tight, rand_ecv = [], []
    for _ in range(N_RANDOM):
        sv = np.linalg.svd(np.random.randn(cfg.d_model, cfg.d_model),
                            compute_uv=False)
        rand_tight.append(float(sv[-1] / sv[0]))
        rand_ecv.append(float(np.std(sv**2) / np.mean(sv**2)))
    print(f"  Random [{cfg.d_model}x{cfg.d_model}] baseline: "
          f"tightness={np.mean(rand_tight):.6f}  energy_cv={np.mean(rand_ecv):.4f}\n")

    results = {}
    print(f"  {'Layer':<6} {'Matrix':<6} {'tightness':>10} "
          f"{'condition':>10} {'energy_cv':>10} {'sv_min':>10} {'sv_max':>10}")
    print("  " + "-" * 70)

    for layer in range(cfg.n_layers):
        W_q, W_k, W_v = get_weights(model, layer, cfg)
        results[str(layer)] = {}
        for name, W in [("W_v", W_v), ("W_q", W_q), ("W_k", W_k)]:
            sv        = np.linalg.svd(W, compute_uv=False)
            tightness = float(sv[-1] / sv[0])
            condition = float(sv[0] / max(sv[-1], 1e-12))
            energy_cv = float(np.std(sv**2) / np.mean(sv**2))
            results[str(layer)][name] = {
                "tightness": tightness, "condition": condition,
                "energy_cv": energy_cv,
                "sv_min": float(sv[-1]), "sv_max": float(sv[0]),
                "A_bound": float(sv[-1]**2), "B_bound": float(sv[0]**2),
            }
            print(f"  L{layer:02d}   {name:<6} {tightness:>10.6f} "
                  f"{condition:>10.2f} {energy_cv:>10.4f} "
                  f"{sv[-1]:>10.4f} {sv[0]:>10.4f}")

    print(f"\n  SUMMARY — Mean tightness:")
    print(f"  {'Matrix':<6} {'mean_tight':>11} {'mean_cond':>11} {'mean_energy_cv':>15}")
    for name in ["W_v", "W_q", "W_k"]:
        ts = [results[str(l)][name]["tightness"]  for l in range(cfg.n_layers)]
        cs = [results[str(l)][name]["condition"]   for l in range(cfg.n_layers)]
        ec = [results[str(l)][name]["energy_cv"]   for l in range(cfg.n_layers)]
        print(f"  {name:<6} {np.mean(ts):>11.6f} {np.mean(cs):>11.2f} "
              f"{np.mean(ec):>15.4f}")

    print(f"\n  W_v tightness by layer:")
    for layer in range(cfg.n_layers):
        t   = results[str(layer)]["W_v"]["tightness"]
        tq  = results[str(layer)]["W_q"]["tightness"]
        tk  = results[str(layer)]["W_k"]["tightness"]
        bar = "█" * max(1, int(t * 50))
        print(f"  L{layer:02d}  Wv={t:.6f}  Wq={tq:.6f}  Wk={tk:.6f}  |{bar}")

    return {
        "random_baseline": {
            "mean_tightness": float(np.mean(rand_tight)),
            "mean_energy_cv": float(np.mean(rand_ecv)),
        },
        "layers": results,
    }


def run_section_g(model, cfg: ModelConfig) -> dict:
    header(f"SECTION G — METRIC TENSOR G = W_q^T W_k  [{cfg.model_name}]")

    results = {}
    print(f"  {'Layer':<6} {'n_pos':>7} {'n_neg':>7} {'psd_frac':>9} "
          f"{'sym_dev':>9} {'eff_rank_G':>11} {'spectral_gap':>13}")
    print("  " + "-" * 65)

    for layer in range(cfg.n_layers):
        W_q, W_k, _ = get_weights(model, layer, cfg)
        G        = W_q.T @ W_k
        G_sym    = (G + G.T) / 2
        eigs_sym = np.linalg.eigvalsh(G_sym)

        n_pos       = int(np.sum(eigs_sym > 0))
        n_neg       = int(np.sum(eigs_sym < 0))
        psd_frac    = float(n_pos / cfg.d_model)
        sym_dev     = float(np.linalg.norm(G - G.T, 'fro') / np.linalg.norm(G, 'fro'))
        eig_range   = float(eigs_sym[-1] - eigs_sym[0])
        top_ratio   = float(eigs_sym[-1] / np.mean(eigs_sym[-10:]))

        pos_e = eigs_sym[eigs_sym > 0]
        neg_e = eigs_sym[eigs_sym < 0]
        spec_gap = float(pos_e.min() - neg_e.max()) if (len(pos_e) > 0 and len(neg_e) > 0) \
                   else (float('inf') if len(neg_e) == 0 else 0.0)

        sv_G     = np.linalg.svd(G, compute_uv=False)
        p_G      = sv_G / sv_G.sum()
        eff_rank = float(np.exp(- 1.0 * np.sum(p_G * np.log(p_G + 1e-12))))

        results[str(layer)] = {
            "n_positive":    n_pos,    "n_negative":   n_neg,
            "psd_fraction":  psd_frac, "sym_dev":      sym_dev,
            "eig_range":     eig_range, "spectral_gap": spec_gap,
            "top_eig_ratio": top_ratio, "eff_rank_G":   eff_rank,
            "eig_max":       float(eigs_sym[-1]),
            "eig_min":       float(eigs_sym[0]),
        }
        print(f"  L{layer:02d}   {n_pos:>7d} {n_neg:>7d} "
              f"{psd_frac:>9.3f} {sym_dev:>9.4f} "
              f"{eff_rank:>11.1f} {spec_gap:>13.4f}")

    print(f"\n  Random G = Rq^T Rk baselines (n=5):")
    for _ in range(5):
        Rq = np.random.randn(cfg.d_model, cfg.d_model)
        Rk = np.random.randn(cfg.d_model, cfg.d_model)
        Gs = (Rq.T @ Rk)
        Gs = (Gs + Gs.T) / 2
        er = np.linalg.eigvalsh(Gs)
        n_p = int(np.sum(er > 0))
        sd  = float(np.linalg.norm(Rq.T@Rk - Rk.T@Rq, 'fro') /
                    np.linalg.norm(Rq.T@Rk, 'fro'))
        print(f"    n_pos={n_p}/{cfg.d_model}  psd_frac={n_p/cfg.d_model:.3f}  sym_dev={sd:.4f}")

    return results


def run_analysis(model_name: str, sections: dict, device: str = "cpu") -> str:
    """
    Load a GPT-2 family model, run all enabled analysis sections, and save
    results to a timestamped JSON file under RESULTS_DIR.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'gpt2' or 'gpt2-medium'.
    sections : dict
        Mapping of section key → bool indicating which sections to run.
    device : str
        PyTorch device string ('cpu' or 'cuda').

    Returns
    -------
    str
        Path to the saved JSON results file.
    """
    print(f"\n{'#' * 65}")
    print(f"#  MODEL: {model_name}")
    print(f"{'#' * 65}")

    print(f"  Loading {model_name}...")
    hf_model  = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    hf_model.eval()
    hf_model.to(device)

    cfg = ModelConfig.from_hf(model_name, hf_model.config)
    print(f"  Config: {cfg.n_layers} layers, {cfg.d_model} hidden, "
          f"{cfg.n_heads} heads, {cfg.head_dim} head_dim")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "meta": {
            "model_name":   model_name,
            "n_layers":     cfg.n_layers,
            "d_model":      cfg.d_model,
            "n_heads":      cfg.n_heads,
            "head_dim":     cfg.head_dim,
            "device":       device,
            "timestamp":    timestamp,
            "sections_run": [k for k, v in sections.items() if v],
        }
    }

    needs_data = sections.get("C", False) or sections.get("D", False)
    tokens = None
    if needs_data:
        print("  Loading WikiText-2 tokens...")
        tokens = load_wikitext_tokens(tokenizer, split="train").to(device)

    if sections.get("A", False):
        all_results["A"] = run_section_a(hf_model, cfg)

    if sections.get("B", False):
        all_results["section_B_polar"] = run_section_b(hf_model, cfg)

    if sections.get("C", False):
        all_results["section_C_mp"] = run_section_c(hf_model, cfg)

    if sections.get("D", False):
        all_results["section_D_gram"] = run_section_d(hf_model, cfg, tokens)

    if sections.get("E", False):
        all_results["section_E_surgery"] = run_section_e(
            hf_model, tokenizer, cfg, device)

    if sections.get("F", False):
        all_results["section_F_frame"] = run_section_f(hf_model, cfg)

    if sections.get("G", False):
        all_results["section_G_metric"] = run_section_g(hf_model, cfg)

    out_path = save_json_results(all_results, model_name, "geometric", timestamp, RESULTS_DIR)
    print(f"    Sections: {list(all_results.keys())}")

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometric analysis of GPT-2 family attention weights"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=MODELS_TO_RUN,
        help="Models to analyse (default: all in MODELS_TO_RUN)"
    )
    parser.add_argument(
        "--skip-sections", nargs="+", default=[],
        metavar="SECTION",
        help="Sections to skip, e.g. --skip-sections C D"
    )
    parser.add_argument(
        "--only-sections", nargs="+", default=[],
        metavar="SECTION",
        help="Run only these sections, e.g. --only-sections A B E F"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda if available)"
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    sections = dict(RUN_SECTIONS)

    if args.only_sections:
        sections = {k: False for k in sections}
        for s in args.only_sections:
            sections[s] = True

    for s in args.skip_sections:
        if s in sections:
            sections[s] = False
        else:
            print(f"Warning: unknown section '{s}' — valid keys: "
                  f"{list(sections.keys())}")

    print(f"\nGeometric Analysis of GPT-2 Attention Weights")
    print(f"  Models  : {args.models}")
    print(f"  Sections: {[k for k, v in sections.items() if v]}")
    print(f"  Device  : {args.device}")
    print(f"  Output  : {RESULTS_DIR}/")

    saved_files = []
    for model_name in args.models:
        try:
            path = run_analysis(model_name, sections, device=args.device)
            saved_files.append(path)
        except Exception as e:
            print(f"\n  ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 65}")
    print(f"All done. Saved {len(saved_files)} file(s):")
    for p in saved_files:
        print(f"  {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()