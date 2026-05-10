"""
Operator characterization analysis for GPT-2 W_q and W_k matrices.

Runs Polar Decomposition and Q/S-Surgery exclusively on the Query and Key
matrices to determine their geometric structure and functional dependence
on orthogonal rotation vs. scaling.

Usage
-----
  python qk_operator_analysis.py
  python qk_operator_analysis.py --models gpt2

Results
-------
  results/qk_analysis_<model_name>_<timestamp>.json
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from scipy.linalg import polar
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

warnings.filterwarnings("ignore")

MODELS_TO_RUN = ["gpt2"]
RESULTS_DIR = "results"
EVAL_TOKENS = 1024


@dataclass
class ModelConfig:
    model_name: str
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int

    @classmethod
    def from_hf(cls, model_name: str, hf_config) -> "ModelConfig":
        d_model = hf_config.n_embd
        n_heads = hf_config.n_head
        n_layers = hf_config.n_layer
        head_dim = d_model // n_heads
        return cls(model_name=model_name, n_layers=n_layers,
                   d_model=d_model, n_heads=n_heads, head_dim=head_dim)


def header(title: str):
    print(f"\n{'=' * 75}")
    print(title)
    print("=" * 75)


def get_weights(model, layer: int, cfg: ModelConfig):
    """Extracts W_q and W_k from the fused c_attn weight."""
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    d = cfg.d_model
    # W_q is [:, :d], W_k is [:, d:2*d]
    return W[:, :d].copy(), W[:, d:2 * d].copy()


def compute_wt2_ppl(model, tokenizer, device: str, max_tokens: int = EVAL_TOKENS) -> float:
    """Measures WikiText-2 perplexity."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    toks = tokenizer.encode(text, return_tensors="pt")[:, :max_tokens].to(device)
    model.eval()
    with torch.no_grad():
        loss = model(toks, labels=toks).loss.item()
    return float(np.exp(loss))


def run_polar_decomposition(model, cfg: ModelConfig) -> dict:
    header(f"SECTION A — POLAR DECOMPOSITION (W_q, W_k)  [{cfg.model_name}]")
    print(f"  {'Layer':<6} {'Matrix':<6} {'Q_orth_dev':>12} {'S_iso_dev':>12} "
          f"{'Q_fidelity':>12} {'stretch_var':>12}")
    print("  " + "-" * 66)

    results = {}
    for layer in range(cfg.n_layers):
        W_q, W_k = get_weights(model, layer, cfg)
        results[str(layer)] = {}

        for name, W in [("W_q", W_q), ("W_k", W_k)]:
            Q, S = polar(W.astype(np.float64))

            Q_orth_dev = float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 'fro'))
            S_norm = S / (np.linalg.norm(S, 'fro') / np.sqrt(S.shape[0]))
            I_norm = np.eye(S.shape[0]) / np.sqrt(S.shape[0])
            S_iso_dev = float(np.linalg.norm(S_norm - I_norm, 'fro'))

            W_unit = W / np.linalg.norm(W, 'fro')
            Q_unit = Q / np.linalg.norm(Q, 'fro')
            Q_fidelity = float(np.linalg.norm(Q_unit - W_unit, 'fro'))

            S_eigs = np.linalg.eigvalsh(S)
            stretch_var = float(np.std(S_eigs) / np.mean(S_eigs))

            results[str(layer)][name] = {
                "Q_orth_dev": Q_orth_dev,
                "S_iso_dev": S_iso_dev,
                "Q_fidelity": Q_fidelity,
                "stretch_var": stretch_var,
            }
            print(f"  L{layer:02d}   {name:<6} {Q_orth_dev:>12.6f} {S_iso_dev:>12.4f} "
                  f"{Q_fidelity:>12.4f} {stretch_var:>12.4f}")

    print(f"\n  SUMMARY — Mean across {cfg.n_layers} layers:")
    print(f"  {'Matrix':<6} {'S_iso_dev':>12} {'Q_fidelity':>12} {'stretch_var':>12}")
    for name in ["W_q", "W_k"]:
        iso = np.mean([results[str(l)][name]["S_iso_dev"] for l in range(cfg.n_layers)])
        fidel = np.mean([results[str(l)][name]["Q_fidelity"] for l in range(cfg.n_layers)])
        strv = np.mean([results[str(l)][name]["stretch_var"] for l in range(cfg.n_layers)])
        print(f"  {name:<6} {iso:>12.4f} {fidel:>12.4f} {strv:>12.4f}")

    return results


def run_q_surgery(model, tokenizer, cfg: ModelConfig, device: str) -> dict:
    header(f"SECTION B — Q/S-SURGERY EXPERIMENT (W_q, W_k)  [{cfg.model_name}]")

    baseline_ppl = compute_wt2_ppl(model, tokenizer, device)
    print(f"  Baseline PPL = {baseline_ppl:.4f}")

    print(f"\n  {'Layer':<6} {'Matrix':<6} {'Q_ppl':>10} {'S_ppl':>10} "
          f"{'Q_delta%':>10} {'S_delta%':>10}")
    print("  " + "-" * 66)

    results = {}
    d = cfg.d_model

    for layer in range(cfg.n_layers):
        W_q, W_k = get_weights(model, layer, cfg)
        attn = model.transformer.h[layer].attn
        orig = attn.c_attn.weight.data.clone()

        results[str(layer)] = {}

        for name, W, slice_start, slice_end in [
            ("W_q", W_q, 0, d),
            ("W_k", W_k, d, 2 * d)
        ]:
            Q_mat, S_mat = polar(W.astype(np.float64))

            # Q-surgery
            try:
                patched = orig.clone()
                patched[:, slice_start:slice_end] = torch.tensor(Q_mat, dtype=torch.float32)
                attn.c_attn.weight.data.copy_(patched)
                ppl_Q = compute_wt2_ppl(model, tokenizer, device)
            finally:
                attn.c_attn.weight.data.copy_(orig)

            # S-surgery
            try:
                patched = orig.clone()
                patched[:, slice_start:slice_end] = torch.tensor(S_mat, dtype=torch.float32)
                attn.c_attn.weight.data.copy_(patched)
                ppl_S = compute_wt2_ppl(model, tokenizer, device)
            finally:
                attn.c_attn.weight.data.copy_(orig)

            delta_Q = 100.0 * (ppl_Q - baseline_ppl) / baseline_ppl
            delta_S = 100.0 * (ppl_S - baseline_ppl) / baseline_ppl

            results[str(layer)][name] = {
                "ppl_Q": float(ppl_Q),
                "ppl_S": float(ppl_S),
                "delta_Q_pct": float(delta_Q),
                "delta_S_pct": float(delta_S),
            }

            print(f"  L{layer:02d}   {name:<6} {ppl_Q:>10.4f} {ppl_S:>10.4f} "
                  f"{delta_Q:>+10.2f}% {delta_S:>+10.2f}%")

    return {
        "baseline_ppl": float(baseline_ppl),
        "layers": results
    }


def run_analysis(model_name: str, device: str = "cpu") -> str:
    print(f"\n{'#' * 75}")
    print(f"#  MODEL: {model_name}")
    print(f"{'#' * 75}")

    print(f"  Loading {model_name}...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
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
            "model_name": model_name,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "head_dim": cfg.head_dim,
            "device": device,
            "timestamp": timestamp,
        }
    }

    # Run Sections
    all_results["polar_decomposition"] = run_polar_decomposition(hf_model, cfg)
    all_results["q_surgery"] = run_q_surgery(hf_model, tokenizer, cfg, device)

    # Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"qk_analysis_{model_name}_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n  [+] Results saved to: {out_path}")

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator analysis of W_q and W_k")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_RUN,
                        help="Models to analyse (default: gpt2)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device (default: cuda if available)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nOperator Characterization Analysis (W_q, W_k)")
    print(f"  Models : {args.models}")
    print(f"  Device : {args.device}")
    print(f"  Output : {RESULTS_DIR}/")

    saved_files = []
    for model_name in args.models:
        try:
            path = run_analysis(model_name, device=args.device)
            saved_files.append(path)
        except Exception as e:
            print(f"\n  ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 75}")
    print(f"All done. Saved {len(saved_files)} file(s):")
    for p in saved_files:
        print(f"  {p}")
    print("=" * 75)


if __name__ == "__main__":
    main()