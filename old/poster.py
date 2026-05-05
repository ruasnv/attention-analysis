"""
paft_analysis.py
=======================================================================
PAFT Longitudinal & Cross-Sectional Analysis Suite
=======================================================================
Protocols:
  A. High-Resolution Longitudinal Sweep (every 50 steps, 0-1000)
     Tasks:   Python, Legal
     Methods: Pure-PAFT, LoRA, Standard-FT

  B. Cross-Sectional Snapshot at 1000 steps
     Methods: Standard-FT, LoRA, SVF, OFT, Pure-PAFT
     Extras:  Geometric analysis, Zero-shot suite, Hardware profiling

Output:
    paft_analysis_results_longitudinal.csv
    paft_analysis_results_crosssectional.csv

Run:
    python poster.py --task python
    python poster.py --task legal
    python poster.py --task python --skip_longitudinal
    python poster.py --task python --skip_crosssectional

Resume:
    Re-run the same command — completed arms are detected from the CSV
    and skipped automatically. Partial longitudinal sweeps resume from
    the last logged step.
=======================================================================
"""

import os
import csv
import time
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import safetensors.torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D
from peft import LoraConfig, OFTConfig, get_peft_model

from paft_core import convert_model_to_paft
from data_and_eval import get_dataloader, evaluate_ppl, evaluate_loss

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME        = "gpt2"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
LR                = 2e-5
BATCH_SIZE        = 4
SEQ_LEN           = 512
LONGITUDINAL_STEPS   = 1000
LOG_INTERVAL         = 50
CROSSSECTIONAL_STEPS = 1000
ZEROSHOT_TASKS = ["winogrande", "hellaswag", "arc_easy", "piqa", "lambada_openai"]

# =============================================================================
# RESUME HELPERS
# =============================================================================

def load_completed_crosssectional(csv_path, task):
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("protocol") == "crosssectional" and row.get("task") == task:
                completed.add(row["method"])
    return completed


def load_last_longitudinal_step(csv_path, task, method):
    last_step = 0
    if not os.path.exists(csv_path):
        return last_step
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("protocol") == "longitudinal"
                    and row.get("task") == task
                    and row.get("method") == method):
                try:
                    s = int(row["step"])
                    if s > last_step:
                        last_step = s
                except (ValueError, KeyError):
                    pass
    return last_step

# =============================================================================
# SVF IMPLEMENTATION
# =============================================================================

class SVF_Attention_Wv(nn.Module):
    def __init__(self, original_c_attn):
        super().__init__()
        W = original_c_attn.weight.data
        b = original_c_attn.bias.data
        d = W.shape[0]
        W_q = W[:, :d]
        W_k = W[:, d:2*d]
        W_v = W[:, 2*d:]
        U, S_vals, Vh = torch.linalg.svd(W_v.float(), full_matrices=True)
        self.register_buffer('W_q', W_q)
        self.register_buffer('W_k', W_k)
        self.register_buffer('U_v', U)
        self.register_buffer('Vh_v', Vh)
        self.S_v  = nn.Parameter(S_vals)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        W_v     = self.U_v @ torch.diag(self.S_v) @ self.Vh_v
        W_fused = torch.cat([self.W_q, self.W_k, W_v], dim=1)
        return torch.matmul(x, W_fused) + self.bias


class SVF_Output_Wo(nn.Module):
    def __init__(self, original_c_proj):
        super().__init__()
        W = original_c_proj.weight.data
        b = original_c_proj.bias.data
        U, S_vals, Vh = torch.linalg.svd(W.float(), full_matrices=True)
        self.register_buffer('U_o', U)
        self.register_buffer('Vh_o', Vh)
        self.S_o  = nn.Parameter(S_vals)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        W_o = self.U_o @ torch.diag(self.S_o) @ self.Vh_o
        return torch.matmul(x, W_o) + self.bias


def convert_model_to_svf(model):
    print("  [SVF] Converting...")
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].attn.c_attn = SVF_Attention_Wv(
            model.transformer.h[i].attn.c_attn)
        model.transformer.h[i].attn.c_proj = SVF_Output_Wo(
            model.transformer.h[i].attn.c_proj)
    for name, param in model.named_parameters():
        param.requires_grad = ".S_v" in name or ".S_o" in name or ".bias" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [SVF] Trainable: {trainable:,}")
    return model

# =============================================================================
# OFT: Conv1D → nn.Linear CONVERSION
# =============================================================================

def convert_conv1d_to_linear(model):
    """Convert ALL Conv1D layers to nn.Linear for OFT compatibility."""
    for name, module in model.named_modules():
        parent_name = ".".join(name.split(".")[:-1])
        child_name  = name.split(".")[-1]
        parent      = model.get_submodule(parent_name) if parent_name else model
        if isinstance(module, Conv1D):
            new = nn.Linear(module.nx, module.nf, bias=(module.bias is not None))
            new.weight.data = module.weight.data.T.clone()
            if module.bias is not None:
                new.bias.data = module.bias.data.clone()
            setattr(parent, child_name, new)
    return model

# =============================================================================
# PAFT WEIGHT FUSION
# =============================================================================

def fuse_paft_weights(model):
    print("  [PAFT] Fusing Q @ S -> W_final...")
    for block in model.transformer.h:
        if hasattr(block.attn.c_attn, 'Q_v') and hasattr(block.attn.c_attn, 'S_v'):
            W_v_fused = torch.matmul(block.attn.c_attn.Q_v, block.attn.c_attn.S_v)
            W_q = block.attn.c_attn.W_q
            W_k = block.attn.c_attn.W_k
            W_fused = torch.cat([W_q, W_k, W_v_fused], dim=-1)
            in_f, out_f = W_fused.shape
            new_c_attn = Conv1D(out_f, in_f).to(W_fused.device)
            new_c_attn.weight.data = W_fused.data
            if hasattr(block.attn.c_attn, 'bias') and block.attn.c_attn.bias is not None:
                new_c_attn.bias.data = block.attn.c_attn.bias.data
            block.attn.c_attn = new_c_attn
        if hasattr(block.attn.c_proj, 'Q_o') and hasattr(block.attn.c_proj, 'S_o'):
            W_o_fused = torch.matmul(block.attn.c_proj.Q_o, block.attn.c_proj.S_o)
            in_f, out_f = W_o_fused.shape
            new_c_proj = Conv1D(out_f, in_f).to(W_o_fused.device)
            new_c_proj.weight.data = W_o_fused.data
            if hasattr(block.attn.c_proj, 'bias') and block.attn.c_proj.bias is not None:
                new_c_proj.bias.data = block.attn.c_proj.bias.data
            block.attn.c_proj = new_c_proj
    print("  [PAFT] Fusion complete.")
    return model

# =============================================================================
# GEOMETRIC ANALYSIS
# =============================================================================

def compute_geometric_metrics(model):
    ranks, entropies = [], []
    with torch.no_grad():
        for name, module in model.named_modules():
            if "c_attn" not in name and "c_proj" not in name:
                continue
            if hasattr(module, 'S_v') and hasattr(module, 'Q_v'):
                W = torch.matmul(module.Q_v, module.S_v).float()
            elif hasattr(module, 'S_o') and hasattr(module, 'Q_o'):
                W = torch.matmul(module.Q_o, module.S_o).float()
            elif hasattr(module, 'weight'):
                W = module.weight.data.float()
            else:
                continue
            try:
                _, S_vals, _ = torch.linalg.svd(W, full_matrices=False)
                s = S_vals.cpu().numpy()
                stable_rank = float(np.sum(s**2) / (np.max(s)**2))
                p = s / np.sum(s)
                entropy = float(-np.sum(p * np.log(p + 1e-12)))
                ranks.append(stable_rank)
                entropies.append(entropy)
            except Exception:
                continue
    return (
        round(float(np.mean(ranks)),     6) if ranks     else 0.0,
        round(float(np.mean(entropies)), 6) if entropies else 0.0
    )

# =============================================================================
# HARDWARE PROFILING
# =============================================================================

def get_peak_vram_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    return 0.0


def reset_vram_counter():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measure_latency(model, tokenizer, num_runs=100):
    dummy = tokenizer(
        "The geometric manifold encodes syntactic structure in the residual stream.",
        return_tensors="pt"
    ).to(DEVICE)
    model.eval()
    for _ in range(10):
        with torch.no_grad():
            _ = model(**dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return round((time.perf_counter() - start) / num_runs * 1000, 4)

# =============================================================================
# ZERO-SHOT EVALUATION
# =============================================================================

def run_zeroshot_suite(model, tokenizer):
    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=DEVICE)
        results = evaluator.simple_evaluate(
            model=lm, tasks=ZEROSHOT_TASKS,
            num_fewshot=0, batch_size=8, log_samples=False,
        )
        out = {}
        for task in ZEROSHOT_TASKS:
            try:
                tr = results["results"][task]
                if "acc,none" in tr:
                    out[task] = round(tr["acc,none"], 4)
                elif "acc_norm,none" in tr:
                    out[task] = round(tr["acc_norm,none"], 4)
                else:
                    out[task] = None
            except (KeyError, TypeError):
                out[task] = None
        return out
    except ImportError:
        print("  [WARN] lm-eval not installed — zero-shot skipped.")
        return {t: None for t in ZEROSHOT_TASKS}
    except Exception as e:
        print(f"  [WARN] Zero-shot eval failed: {e}")
        return {t: None for t in ZEROSHOT_TASKS}

# =============================================================================
# DERIVED COEFFICIENTS
# =============================================================================

def stability_coefficient(target_loss, wt2_ppl):
    if wt2_ppl and wt2_ppl > 0:
        return round(target_loss / wt2_ppl, 6)
    return None


def generalization_efficiency(target_loss, step, wt2_ppl):
    if step > 0 and wt2_ppl and wt2_ppl > 0:
        return round(target_loss / (step * wt2_ppl), 8)
    return None

# =============================================================================
# CSV HELPERS  —  separate files, separate headers, no column mismatch ever
# =============================================================================

LONGITUDINAL_FIELDS = [
    "protocol", "task", "method", "step",
    "target_loss", "wt2_ppl",
    "alpha_stability", "beta_gen_efficiency",
    "ms_per_step", "peak_vram_mb",
    "trainable_params", "total_params", "param_pct",
]

CROSSSECTIONAL_FIELDS = [
    "protocol", "task", "method", "step",
    "target_loss", "wt2_ppl",
    "alpha_stability", "beta_gen_efficiency",
    "stable_rank", "spectral_entropy",
    "latency_baseline_ms", "latency_method_ms", "latency_fused_ms",
    "peak_vram_mb", "trainable_params", "total_params", "param_pct",
] + ZEROSHOT_TASKS


def open_csv(path, fields):
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if not exists:
        writer.writeheader()
    return f, writer

# =============================================================================
# MODEL SETUP FACTORIES
# =============================================================================

def make_fresh_model():
    return GPT2LMHeadModel.from_pretrained(MODEL_NAME)

def setup_standard_ft(model): return model

def setup_lora(model):
    # Matches original LoRA paper config and run_paft_experiment.py exactly.
    # c_attn is the fused [W_q, W_k, W_v] matrix in GPT-2.
    # c_proj (W_o) is intentionally excluded to match the original experiment.
    return get_peft_model(model, LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", fan_in_fan_out=True,
    ))

def setup_svf(model): return convert_model_to_svf(model)

def setup_oft(model):
    model = convert_conv1d_to_linear(model)
    return get_peft_model(model, OFTConfig(
        oft_block_size=32,
        target_modules=["c_attn", "c_proj"],
        module_dropout=0.0, init_weights=True,
        task_type="CAUSAL_LM",
    ))

def setup_paft(model): return convert_model_to_paft(model, mode="pure")

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total, round(trainable / total * 100, 2) if total else 0.0

# =============================================================================
# PROTOCOL A — LONGITUDINAL SWEEP
# =============================================================================

LONGITUDINAL_METHODS = {
    "Pure-PAFT":   setup_paft,
    "LoRA":        setup_lora,
    "Standard-FT": setup_standard_ft,
}


def run_longitudinal(task, tokenizer, csv_path):
    print(f"\n{'='*60}")
    print(f"  PROTOCOL A — LONGITUDINAL | Task: {task.upper()}")
    print(f"{'='*60}")

    f, writer = open_csv(csv_path, LONGITUDINAL_FIELDS)

    # Baseline
    if load_last_longitudinal_step(csv_path, task, "Baseline") > 0:
        print("  Baseline already logged — skipping.")
    else:
        m = make_fresh_model().to(DEVICE)
        bppl = evaluate_ppl(m, tokenizer, DEVICE)
        bloss = evaluate_loss(m, get_dataloader(task, tokenizer, "test", BATCH_SIZE), DEVICE)
        del m; torch.cuda.empty_cache()
        print(f"  Baseline | PPL: {bppl:.4f} | Loss: {bloss:.4f}")
        writer.writerow({
            "protocol": "longitudinal", "task": task, "method": "Baseline", "step": 0,
            "target_loss": bloss, "wt2_ppl": bppl,
            "alpha_stability": stability_coefficient(bloss, bppl),
            "beta_gen_efficiency": None,
            "ms_per_step": 0, "peak_vram_mb": 0,
            "trainable_params": 0, "total_params": 124439808, "param_pct": 0.0,
        })
        f.flush()

    for method_name, setup_fn in LONGITUDINAL_METHODS.items():
        last_step = load_last_longitudinal_step(csv_path, task, method_name)
        if last_step >= LONGITUDINAL_STEPS:
            print(f"\n  [{method_name}] fully complete — skipping.")
            continue
        if last_step > 0:
            print(f"\n  [{method_name}] resuming from step {last_step}")
        else:
            print(f"\n  [{method_name}] starting fresh")

        model = setup_fn(make_fresh_model()).to(DEVICE)
        trainable, total, pct = count_params(model)
        print(f"     Trainable: {trainable:,} ({pct:.1f}%)")
        optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        train_iter = iter(get_dataloader(task, tokenizer, "train", BATCH_SIZE))
        step_times = []
        reset_vram_counter()

        if last_step > 0:
            print(f"     Fast-forwarding {last_step} steps...")
            bp = tqdm(total=last_step, desc="  [fast-forward]", ncols=90)
            for _ in range(last_step):
                model.train()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(get_dataloader(task, tokenizer, "train", BATCH_SIZE))
                    batch = next(train_iter)
                optimizer.zero_grad()
                model(batch[0].to(DEVICE), labels=batch[1].to(DEVICE)).loss.backward()
                optimizer.step()
                bp.update(1)
            bp.close()

        pbar = tqdm(total=LONGITUDINAL_STEPS - last_step, desc=f"  [{method_name}]", ncols=90)
        for step in range(last_step + 1, LONGITUDINAL_STEPS + 1):
            model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(get_dataloader(task, tokenizer, "train", BATCH_SIZE))
                batch = next(train_iter)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            loss = model(batch[0].to(DEVICE), labels=batch[1].to(DEVICE)).loss
            loss.backward()
            optimizer.step()
            step_times.append((time.perf_counter() - t0) * 1000)
            pbar.update(1)

            if step % LOG_INTERVAL == 0:
                model.eval()
                wt2_ppl     = evaluate_ppl(model, tokenizer, DEVICE)
                target_loss = evaluate_loss(model, get_dataloader(task, tokenizer, "test", BATCH_SIZE), DEVICE)
                avg_ms      = round(float(np.mean(step_times[-100:])), 4)
                alpha       = stability_coefficient(target_loss, wt2_ppl)
                beta        = generalization_efficiency(target_loss, step, wt2_ppl)
                writer.writerow({
                    "protocol": "longitudinal", "task": task, "method": method_name, "step": step,
                    "target_loss": round(float(target_loss), 6),
                    "wt2_ppl": round(float(wt2_ppl), 6),
                    "alpha_stability": alpha, "beta_gen_efficiency": beta,
                    "ms_per_step": avg_ms, "peak_vram_mb": get_peak_vram_mb(),
                    "trainable_params": trainable, "total_params": total, "param_pct": pct,
                })
                f.flush()
                pbar.set_postfix({"loss": f"{target_loss:.4f}", "ppl": f"{wt2_ppl:.2f}", "α": f"{alpha:.4f}"})

        pbar.close()
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    f.close()
    print(f"\n  [Protocol A] Done → {csv_path}")

# =============================================================================
# PROTOCOL B — CROSS-SECTIONAL SNAPSHOTS
# =============================================================================

CROSSSECTIONAL_METHODS = {
    "Standard-FT": setup_standard_ft,
    "LoRA":        setup_lora,
    "SVF":         setup_svf,
    "OFT":         setup_oft,
    "Pure-PAFT":   setup_paft,
}


def run_crosssectional(task, tokenizer, csv_path):
    print(f"\n{'='*60}")
    print(f"  PROTOCOL B — CROSS-SECTIONAL | Task: {task.upper()}")
    print(f"{'='*60}")

    completed = load_completed_crosssectional(csv_path, task)
    if completed:
        print(f"  Already completed: {', '.join(sorted(completed))} — skipping.")

    f, writer = open_csv(csv_path, CROSSSECTIONAL_FIELDS)

    # ── Baseline ─────────────────────────────────────────────────────────────
    # Always measure baseline latency — needed for PAFT fusion comparison
    baseline_model   = make_fresh_model().to(DEVICE)
    latency_baseline = measure_latency(baseline_model, tokenizer)

    if "Baseline" in completed:
        print("\n  ARM: Baseline — already logged, skipping.")
        del baseline_model; torch.cuda.empty_cache()
    else:
        print("\n  ARM: Baseline")
        baseline_ppl  = evaluate_ppl(baseline_model, tokenizer, DEVICE)
        baseline_loss = evaluate_loss(
            baseline_model,
            get_dataloader(task, tokenizer, "test", BATCH_SIZE), DEVICE
        )
        sr_b, h_b = compute_geometric_metrics(baseline_model)
        zs_b      = run_zeroshot_suite(baseline_model, tokenizer)
        alpha_b   = stability_coefficient(baseline_loss, baseline_ppl)
        row = {
            "protocol": "crosssectional", "task": task, "method": "Baseline", "step": 0,
            "target_loss": round(float(baseline_loss), 6),
            "wt2_ppl":     round(float(baseline_ppl), 6),
            "alpha_stability": alpha_b, "beta_gen_efficiency": None,
            "stable_rank": sr_b, "spectral_entropy": h_b,
            "latency_baseline_ms": latency_baseline,
            "latency_method_ms":   latency_baseline,
            "latency_fused_ms":    latency_baseline,
            "peak_vram_mb":  get_peak_vram_mb(),
            "trainable_params": 0,
            "total_params":  sum(p.numel() for p in baseline_model.parameters()),
            "param_pct": 0.0,
        }
        row.update(zs_b)
        writer.writerow(row)
        f.flush()
        del baseline_model; torch.cuda.empty_cache()

    # ── Training arms ────────────────────────────────────────────────────────
    for method_name, setup_fn in CROSSSECTIONAL_METHODS.items():
        if method_name in completed:
            print(f"\n  ARM: {method_name} — already logged, skipping.")
            continue

        print(f"\n  ARM: {method_name}")
        model = setup_fn(make_fresh_model()).to(DEVICE)
        trainable, total, pct = count_params(model)
        print(f"     Trainable: {trainable:,} ({pct:.1f}%)")

        optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        train_iter = iter(get_dataloader(task, tokenizer, "train", BATCH_SIZE))
        reset_vram_counter()

        pbar = tqdm(total=CROSSSECTIONAL_STEPS, desc=f"  [{method_name}]", ncols=90)
        for step in range(1, CROSSSECTIONAL_STEPS + 1):
            model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(get_dataloader(task, tokenizer, "train", BATCH_SIZE))
                batch = next(train_iter)
            optimizer.zero_grad()
            model(batch[0].to(DEVICE), labels=batch[1].to(DEVICE)).loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        peak_vram   = get_peak_vram_mb()
        model.eval()
        wt2_ppl     = evaluate_ppl(model, tokenizer, DEVICE)
        target_loss = evaluate_loss(model, get_dataloader(task, tokenizer, "test", BATCH_SIZE), DEVICE)
        sr, h       = compute_geometric_metrics(model)
        lat_method  = measure_latency(model, tokenizer)
        zeroshot    = run_zeroshot_suite(model, tokenizer)
        alpha       = stability_coefficient(target_loss, wt2_ppl)
        beta        = generalization_efficiency(target_loss, CROSSSECTIONAL_STEPS, wt2_ppl)

        lat_fused = lat_method
        if method_name == "Pure-PAFT":
            print("  [PAFT] Weight fusion for deployment simulation...")
            model_fused = fuse_paft_weights(model)
            lat_fused   = measure_latency(model_fused, tokenizer)
            print(f"  [PAFT] Unfused: {lat_method}ms | Fused: {lat_fused}ms | Baseline: {latency_baseline}ms")
            del model_fused; torch.cuda.empty_cache()

        row = {
            "protocol": "crosssectional", "task": task, "method": method_name,
            "step": CROSSSECTIONAL_STEPS,
            "target_loss": round(float(target_loss), 6),
            "wt2_ppl":     round(float(wt2_ppl), 6),
            "alpha_stability": alpha, "beta_gen_efficiency": beta,
            "stable_rank": sr, "spectral_entropy": h,
            "latency_baseline_ms": latency_baseline,
            "latency_method_ms":   lat_method,
            "latency_fused_ms":    lat_fused,
            "peak_vram_mb": peak_vram,
            "trainable_params": trainable, "total_params": total, "param_pct": pct,
        }
        row.update(zeroshot)
        writer.writerow(row)
        f.flush()

        print(f"  ✓ PPL:{wt2_ppl:.4f} | Loss:{target_loss:.4f} | α:{alpha:.4f} | SR:{sr:.4f} | H:{h:.4f}")
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    f.close()
    print(f"\n  [Protocol B] Done → {csv_path}")

# =============================================================================
# TERMINAL SUMMARY TABLE
# =============================================================================

def print_summary(csv_long, csv_cross, task):
    # ── Longitudinal ─────────────────────────────────────────────────────────
    long_rows = []
    if os.path.exists(csv_long):
        with open(csv_long, "r") as f:
            long_rows = [
                r for r in csv.DictReader(f)
                if r.get("task") == task
                and r.get("protocol") == "longitudinal"
                and int(r.get("step", 0)) == LONGITUDINAL_STEPS
            ]
    if long_rows:
        print(f"\n{'='*80}")
        print(f"  LONGITUDINAL FINAL (step {LONGITUDINAL_STEPS}) — {task.upper()}")
        print(f"{'='*80}")
        print(f"{'Method':<15} | {'Loss':<10} | {'WT2 PPL':<10} | {'α':<10} | {'ms/step':<10}")
        print("─" * 60)
        for r in long_rows:
            print(f"{r['method']:<15} | "
                  f"{float(r['target_loss']):<10.4f} | "
                  f"{float(r['wt2_ppl']):<10.4f} | "
                  f"{float(r.get('alpha_stability') or 0):<10.6f} | "
                  f"{float(r.get('ms_per_step') or 0):<10.2f}")
        print("=" * 80)

    # ── Cross-sectional ───────────────────────────────────────────────────────
    cross_rows = []
    if os.path.exists(csv_cross):
        with open(csv_cross, "r") as f:
            cross_rows = [r for r in csv.DictReader(f) if r.get("task") == task]
    if cross_rows:
        print(f"\n{'='*95}")
        print(f"  CROSS-SECTIONAL SNAPSHOT (step {CROSSSECTIONAL_STEPS}) — {task.upper()}")
        print(f"{'='*95}")
        print(f"{'Method':<15} | {'Loss':<8} | {'PPL':<8} | {'α':<8} | "
              f"{'Stable Rank':<12} | {'Entropy':<10} | {'Lat Method':<12} | {'Lat Fused':<10}")
        print("─" * 95)
        for r in cross_rows:
            print(f"{r['method']:<15} | "
                  f"{float(r.get('target_loss') or 0):<8.4f} | "
                  f"{float(r.get('wt2_ppl') or 0):<8.4f} | "
                  f"{float(r.get('alpha_stability') or 0):<8.4f} | "
                  f"{float(r.get('stable_rank') or 0):<12.4f} | "
                  f"{float(r.get('spectral_entropy') or 0):<10.4f} | "
                  f"{float(r.get('latency_method_ms') or 0):<12.4f} | "
                  f"{float(r.get('latency_fused_ms') or 0):<10.4f}")
        print("=" * 95)

        print(f"\n  Zero-Shot Benchmarks — {task.upper()}")
        print(f"{'Method':<15}", end="")
        for t in ZEROSHOT_TASKS:
            print(f" | {t[:12]:<12}", end="")
        print()
        print("─" * (15 + 16 * len(ZEROSHOT_TASKS)))
        for r in cross_rows:
            print(f"{r['method']:<15}", end="")
            for t in ZEROSHOT_TASKS:
                val = r.get(t) or "N/A"
                try:
                    val = f"{float(val):.4f}"
                except (ValueError, TypeError):
                    val = "N/A"
                print(f" | {val:<12}", end="")
            print()
        print("=" * (15 + 16 * len(ZEROSHOT_TASKS)))

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="python", choices=["python", "legal"])
    parser.add_argument("--skip_longitudinal",    action="store_true")
    parser.add_argument("--skip_crosssectional",  action="store_true")
    parser.add_argument("--csv",    default="paft_analysis_results.csv")
    args = parser.parse_args()

    csv_long  = args.csv.replace(".csv", "_longitudinal.csv")
    csv_cross = args.csv.replace(".csv", "_crosssectional.csv")

    print(f"\n{'='*60}")
    print(f"  PAFT Analysis Suite | Task: {args.task.upper()} | Device: {DEVICE}")
    print(f"  Longitudinal  → {csv_long}")
    print(f"  Crosssectional → {csv_cross}")
    print(f"{'='*60}")

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_longitudinal:
        run_longitudinal(args.task, tokenizer, csv_long)

    if not args.skip_crosssectional:
        run_crosssectional(args.task, tokenizer, csv_cross)

    print_summary(csv_long, csv_cross, args.task)
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()