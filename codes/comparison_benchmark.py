"""
comparison_benchmark.py
=======================================================================
Fine-Tuning Method Comparison Benchmark
Methods: Baseline, Standard-FT, BitFit, LoRA, IA³, OFT, SVF, Pure-PAFT
Task: Configurable — default python (loss), also supports legal/medical/english_web

Checkpoint loading priority:
  1. {task}_{arm}_step1000   (from checkpoint experiment)
  2. {task}_{arm}            (from original runs)
  3. Train fresh and save

Run:
    python comparison_benchmark.py --task python --steps 1000
    python comparison_benchmark.py --task english_web --steps 1000
=======================================================================
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import safetensors.torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D
from peft import LoraConfig, IA3Config, OFTConfig, get_peft_model, PeftModel

from paft_core import convert_model_to_paft
from data_and_eval import get_dataloader, evaluate_ppl, evaluate_loss, evaluate_accuracy

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "gpt2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LR         = 2e-5
BATCH_SIZE = 4
SEQ_LEN    = 512


# =============================================================================
# LABEL → FOLDER NAME MAPPING
# Normalises arm labels to the folder names used by run_paft_experiment.py
# =============================================================================
LABEL_TO_FOLDER = {
    "Baseline":    None,           # No folder — always load from HF
    "Standard-FT": "standard_ft",
    "BitFit":      "bitfit",
    "LoRA":        "lora",
    "IA3":         "ia3",
    "OFT":         "oft",
    "SVF":         "svf",
    "Pure-PAFT":   "pure_paft",
}

# Methods that save via HuggingFace PeftModel (need PeftModel.from_pretrained)
PEFT_METHODS = {"LoRA", "IA3", "OFT"}

# Methods that need their custom skeleton rebuilt before loading weights
CUSTOM_SKELETON = {"Pure-PAFT", "SVF"}


# =============================================================================
# SVF — Singular Value Fine-Tuning
# =============================================================================

class SVF_Attention_Wv(nn.Module):
    """Trains only the d diagonal singular values of W_v. Freezes U, Vh, W_q, W_k."""
    def __init__(self, original_c_attn):
        super().__init__()
        W = original_c_attn.weight.data   # [768, 2304]
        b = original_c_attn.bias.data
        d = W.shape[0]                    # 768

        W_q = W[:, :d]
        W_k = W[:, d:2*d]
        W_v = W[:, 2*d:]                  # [768, 768]

        U, S_vals, Vh = torch.linalg.svd(W_v.float(), full_matrices=True)

        self.register_buffer('W_q', W_q)
        self.register_buffer('W_k', W_k)
        self.register_buffer('U_v',  U)
        self.register_buffer('Vh_v', Vh)
        self.S_v  = nn.Parameter(S_vals)  # d scalars — only trainable params
        self.bias = nn.Parameter(b)

    def forward(self, x):
        W_v     = self.U_v @ torch.diag(self.S_v) @ self.Vh_v
        W_fused = torch.cat([self.W_q, self.W_k, W_v], dim=1)
        return torch.matmul(x, W_fused) + self.bias


class SVF_Output_Wo(nn.Module):
    """Trains only the d diagonal singular values of W_o."""
    def __init__(self, original_c_proj):
        super().__init__()
        W = original_c_proj.weight.data   # [768, 768]
        b = original_c_proj.bias.data

        U, S_vals, Vh = torch.linalg.svd(W.float(), full_matrices=True)

        self.register_buffer('U_o',  U)
        self.register_buffer('Vh_o', Vh)
        self.S_o  = nn.Parameter(S_vals)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        W_o = self.U_o @ torch.diag(self.S_o) @ self.Vh_o
        return torch.matmul(x, W_o) + self.bias


def convert_model_to_svf(model):
    print("\n  [SVF] Converting model to SVF mode...")
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].attn.c_attn = SVF_Attention_Wv(
            model.transformer.h[i].attn.c_attn)
        model.transformer.h[i].attn.c_proj = SVF_Output_Wo(
            model.transformer.h[i].attn.c_proj)

    for name, param in model.named_parameters():
        param.requires_grad = (
            ".S_v" in name or ".S_o" in name or ".bias" in name
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [SVF] Complete. Trainable Parameters: {trainable:,}")
    return model


# =============================================================================
# BITFIT
# =============================================================================

def convert_model_to_bitfit(model):
    print("\n  [BitFit] Enabling bias parameters only...")
    for name, param in model.named_parameters():
        param.requires_grad = "bias" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [BitFit] Complete. Trainable Parameters: {trainable:,}")
    return model


# =============================================================================
# CHECKPOINT RESOLUTION
# =============================================================================

def resolve_checkpoint_path(task, label):
    """
    Returns path of an existing checkpoint or None if not found.
    Priority: {task}_{arm}_step1000  ->  {task}_{arm}
    """
    folder = LABEL_TO_FOLDER.get(label)
    if folder is None:
        return None  # Baseline always loads from HF

    candidates = [
        f"./models/{task}_{folder}_step1000",   # checkpoint-experiment format
        f"./models/{task}_{folder}",             # original run format
    ]

    for path in candidates:
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors"))     or
            os.path.exists(os.path.join(path, "pytorch_model.bin"))     or
            os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
            os.path.exists(os.path.join(path, "adapter_model.bin"))
        )
        if has_weights:
            return path

    return None  # Nothing found — must train


def load_checkpoint(task, label, path):
    """
    Loads a saved model given its label and resolved path.
    Handles HF models, PEFT adapters, and custom-skeleton methods.
    """
    base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    if label in PEFT_METHODS:
        model = PeftModel.from_pretrained(base, path)

    elif label == "Pure-PAFT":
        model = convert_model_to_paft(base, mode="pure")
        sf   = os.path.join(path, "model.safetensors")
        bin_ = os.path.join(path, "pytorch_model.bin")
        sd   = safetensors.torch.load_file(sf) if os.path.exists(sf) \
               else torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)

    elif label == "SVF":
        model = convert_model_to_svf(base)
        sf   = os.path.join(path, "model.safetensors")
        bin_ = os.path.join(path, "pytorch_model.bin")
        sd   = safetensors.torch.load_file(sf) if os.path.exists(sf) \
               else torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)

    else:
        # Standard-FT, BitFit — saved as full HF models
        model = GPT2LMHeadModel.from_pretrained(path)

    return model.to(DEVICE)


# =============================================================================
# SETUP FUNCTIONS  (only called when training fresh)
# =============================================================================

def setup_standard_ft(m): return m

def setup_bitfit(m):      return convert_model_to_bitfit(m)

def setup_lora(m):
    cfg = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", fan_in_fan_out=True
    )
    return get_peft_model(m, cfg)

def setup_ia3(m):
    cfg = IA3Config(
        target_modules=["c_attn", "c_proj", "mlp.c_proj"],
        feedforward_modules=["mlp.c_proj"],
        task_type="CAUSAL_LM", fan_in_fan_out=True
    )
    return get_peft_model(m, cfg)


def setup_oft(m):
    # GPT-2 specific fix: Convert Conv1D to Linear so PEFT/OFT accepts it
    for name, module in m.named_modules():
        if isinstance(module, Conv1D):
            # Get parent and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = m.get_submodule(parent_name)

            # Create standard Linear layer
            new_layer = nn.Linear(module.weight.shape[0], module.weight.shape[1])
            new_layer.weight.data = module.weight.data.T  # Transpose weight for Linear
            new_layer.bias.data = module.bias.data

            setattr(parent, attr_name, new_layer)

    cfg = OFTConfig(
        r=8,
        oft_block_size=0,
        target_modules=["c_attn", "c_proj"],
        module_dropout=0.0,
        init_weights=True,
        fan_in_fan_out=False,  # Set to False now that we are using standard Linear
        task_type="CAUSAL_LM"
    )
    return get_peft_model(m, cfg)

def setup_svf(m):       return convert_model_to_svf(m)
def setup_pure_paft(m): return convert_model_to_paft(m, mode="pure")

SETUP_FNS = {
    "Standard-FT": setup_standard_ft,
    "BitFit":      setup_bitfit,
    "LoRA":        setup_lora,
    "IA3":         setup_ia3,
    "OFT":         setup_oft,
    "SVF":         setup_svf,
    "Pure-PAFT":   setup_pure_paft,
}


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(model, dataloader, steps, lr, label):
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    step = 0
    pbar = tqdm(total=steps, desc=f"  [{label}]")
    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break
            input_ids = batch[0].to(DEVICE)
            labels    = batch[1].to(DEVICE)
            optimizer.zero_grad()
            model(input_ids, labels=labels).loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
    pbar.close()
    return model


# =============================================================================
# LATENCY MEASUREMENT
# =============================================================================

def measure_latency(model, tokenizer, num_runs=50):
    dummy = tokenizer(
        "The geometric manifold encodes syntactic structure.",
        return_tensors="pt"
    ).to(DEVICE)
    model.eval()
    for _ in range(5):
        _ = model(**dummy)
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**dummy)
    return round((time.time() - start) / num_runs * 1000, 4)


# =============================================================================
# ARM ORDER
# =============================================================================
ARMS = [
    #("Baseline",    False),
    #("Standard-FT", True),
    #("BitFit",      True),
    #("LoRA",        True),
    #("IA3",         True),
    ("OFT",         True),
    #("SVF",         True),
    #("Pure-PAFT",   True),
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="python",
                        choices=["python", "legal", "medical", "english_web"])
    parser.add_argument("--steps", type=int,   default=1000)
    parser.add_argument("--lr",    type=float, default=LR)
    args = parser.parse_args()

    task  = args.task
    steps = args.steps
    lr    = args.lr

    print(f"\n{'='*60}")
    print(f"  PAFT Comparison Benchmark")
    print(f"  Task: {task.upper()} | Steps: {steps} | Device: {DEVICE}")
    print(f"{'='*60}")

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    is_accuracy = (task == "medical")
    metric_key  = "Accuracy" if is_accuracy else "Target_Loss"

    def get_metric(m):
        if is_accuracy:
            return evaluate_accuracy(m, tokenizer, task, DEVICE)
        fresh = get_dataloader(task, tokenizer, split="test", batch_size=BATCH_SIZE)
        return evaluate_loss(m, fresh, DEVICE)

    results = {}

    for label, needs_training in ARMS:
        print(f"\n{'─'*55}")
        print(f"  ARM: {label}")

        try:
            ckpt_path = resolve_checkpoint_path(task, label)

            # ── Load from checkpoint ───────────────────────────────────────
            if label == "Baseline":
                model  = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
                source = "pretrained"
                for p in model.parameters():
                    p.requires_grad = False

            elif ckpt_path is not None:
                print(f"  [CACHE HIT] {ckpt_path}")
                model  = load_checkpoint(task, label, ckpt_path)
                source = f"cached ({os.path.basename(ckpt_path)})"

            # ── Train fresh ────────────────────────────────────────────────
            else:
                print(f"  [NO CACHE] Training fresh for {steps} steps...")
                train_loader = get_dataloader(
                    task, tokenizer, split="train", batch_size=BATCH_SIZE
                )
                base  = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
                model = SETUP_FNS[label](base).to(DEVICE)
                model = train(model, train_loader, steps=steps, lr=lr, label=label)

                save_path = f"./models/{task}_{LABEL_TO_FOLDER[label]}_step{steps}"
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                print(f"  [SAVED] -> {save_path}")
                source = "trained+saved"

            # ── Evaluate ───────────────────────────────────────────────────
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            wt2_ppl   = evaluate_ppl(model, tokenizer, DEVICE)
            metric    = get_metric(model)
            latency   = measure_latency(model, tokenizer)

            results[label] = {
                "Source":           source,
                "Trainable_Params": trainable,
                "Total_Params":     total,
                "Param_Pct":        round(trainable / total * 100, 2),
                "WT2_PPL":          round(float(wt2_ppl), 4),
                metric_key:         round(float(metric),  4),
                "Latency_ms":       latency,
            }

            print(f"  ✓ PPL: {wt2_ppl:.4f} | {metric_key}: {metric:.4f} "
                  f"| Params: {trainable:,} ({trainable/total*100:.1f}%) "
                  f"| Latency: {latency}ms | Source: {source}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            results[label] = {"error": str(e)}

        finally:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()

    # =========================================================================
    # PRINT TABLE
    # =========================================================================
    metric_label = "Accuracy" if is_accuracy else f"Loss ({task})"
    c = dict(arm=15, src=26, params=17, pct=8, ppl=10, metric=14, lat=13)

    def header_str():
        return (
            f"{'Method':<{c['arm']}} | "
            f"{'Source':<{c['src']}} | "
            f"{'Trainable Params':<{c['params']}} | "
            f"{'% Total':<{c['pct']}} | "
            f"{'WT2 PPL':<{c['ppl']}} | "
            f"{metric_label:<{c['metric']}} | "
            f"{'Latency (ms)':<{c['lat']}}"
        )

    def print_row(label, data, delta=False, base_ppl=None, base_metric=None):
        if "error" in data:
            print(f"{label:<{c['arm']}} | ERROR: {data['error'][:50]}")
            return
        src    = str(data.get("Source", "─"))[:c['src']]
        params = f"{data.get('Trainable_Params', 0):,}"
        pct    = f"{data.get('Param_Pct', 0):.1f}%"
        lat    = str(data.get("Latency_ms", "N/A"))
        if delta and base_ppl is not None:
            pd  = data["WT2_PPL"] - base_ppl
            md  = data.get(metric_key, 0) - (base_metric or 0)
            ppl = f"{'+' if pd>0 else ''}{pd:.4f}"
            met = f"{'+' if md>0 else ''}{md:.4f}"
        else:
            ppl = str(data.get("WT2_PPL", "N/A"))
            met = str(data.get(metric_key, "N/A"))
        print(
            f"{label:<{c['arm']}} | {src:<{c['src']}} | "
            f"{params:<{c['params']}} | {pct:<{c['pct']}} | "
            f"{ppl:<{c['ppl']}} | {met:<{c['metric']}} | {lat:<{c['lat']}}"
        )

    hdr = header_str()
    div = "─" * len(hdr)

    # Raw results
    print(f"\n\n{'='*len(hdr)}")
    print(f"  Results — {task.upper()} | {steps} steps | {DEVICE}")
    print(f"{'='*len(hdr)}")
    print(hdr); print(div)
    for lbl, data in results.items():
        print_row(lbl, data)
    print("=" * len(hdr))

    # Delta vs Baseline
    if "Baseline" in results and "error" not in results["Baseline"]:
        base_ppl    = results["Baseline"]["WT2_PPL"]
        base_metric = results["Baseline"].get(metric_key)
        direction   = "↑" if is_accuracy else "↓"

        print(f"\n{'='*len(hdr)}")
        print(f"  Δ vs Baseline  "
              f"(↓ WT2 PPL = less forgetting | {direction} {metric_key} = better)")
        print(f"{'='*len(hdr)}")
        print(hdr); print(div)
        for lbl, data in results.items():
            if lbl == "Baseline" or "error" in data:
                continue
            print_row(lbl, data, delta=True,
                      base_ppl=base_ppl, base_metric=base_metric)
        print("=" * len(hdr))

    # =========================================================================
    # SMART SAVE (MERGE)
    # =========================================================================
    # 1. Create the data object for the current run
    current_run_data = {
        "config": {
            "task": task,
            "steps": steps,
            "lr": lr,
            "device": DEVICE,
            "model": MODEL_NAME
        },
        "results": results
    }

    fname = f"comparison_results_{task}.json"

    # 2. Load existing data if the file exists
    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                final_output = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # If the file is corrupted, start fresh
            final_output = {"config": {}, "results": {}}
    else:
        final_output = {"config": {}, "results": {}}

    # 3. Merge current results into the historical file
    final_output["config"].update(current_run_data["config"])
    final_output["results"].update(current_run_data["results"])

    # 4. Write back to disk
    with open(fname, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"\n  Successfully Merged and Saved -> {fname}\n")


if __name__ == "__main__":
    main()