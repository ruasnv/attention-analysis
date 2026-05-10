"""
victory_lap.py
=======================================================================
The Ultimate targeted OV Surgery: Head-Wise Spectral Clipping
=======================================================================
Intervention:
  - Target: Layer 0, W_v matrix.
  - Spatial Focus: Attention Heads 9 and 10 ONLY.
  - Spectral Focus: Clip the top 5 singular values of those specific heads.

Validation:
  - Healing: OpenWebText (Streaming) to prevent WT2 data leakage.
  - Evaluation: WikiText-2 (Fluency), LAMBADA (Long-Range), HellaSwag (Logic).
  - Output: Terminal table logging + JSON export.
"""

import argparse
import gc
import json
import math
import os
import time
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "gpt2"
DEVICE = "cuda"
D_MODEL = 768
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS

DEFAULT_FT_STEPS = 1000
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 512


# =============================================================================
# MODEL UTILITIES & SURGERY
# =============================================================================

def fresh_model():
    return GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32).eval().to(DEVICE)


def apply_targeted_spectral_clipping(model, layer, target_heads, clip_count=5):
    """
    Applies SVD clipping to specific heads within the W_v matrix.
    """
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    W_v = W[:, 2 * D_MODEL:].copy()

    diagnostic_logs = {}

    for head_idx in target_heads:
        head_start = head_idx * D_HEAD
        head_end = (head_idx + 1) * D_HEAD
        W_head = W_v[:, head_start:head_end]

        # Decompose the specific 768x64 head
        U, S, Vh = np.linalg.svd(W_head, full_matrices=False)

        # Clip the aggressive local bias outliers
        S_clipped = S.copy()
        clip_threshold = S_clipped[clip_count]
        S_clipped[:clip_count] = clip_threshold

        # Reconstruct the head
        W_head_new = U @ np.diag(S_clipped) @ Vh

        # Preserve local Frobenius norm
        norm_orig = np.linalg.norm(W_head, 'fro')
        norm_new = np.linalg.norm(W_head_new, 'fro')
        W_head_new = W_head_new * (norm_orig / norm_new)

        # Inject back into W_v slice
        W_v[:, head_start:head_end] = W_head_new

        diagnostic_logs[f"Head_{head_idx}"] = {
            "original_top_5": [float(val) for val in S[:5]],
            "clipped_top_5": [float(val) for val in S_clipped[:5]]
        }

    # Inject the modified W_v back into the fused c_attn weight
    W_new = W.copy()
    W_new[:, 2 * D_MODEL:] = W_v
    model.transformer.h[layer].attn.c_attn.weight.data = torch.tensor(W_new, dtype=torch.float32).to(DEVICE)

    return W_v, diagnostic_logs


def freeze_wv_circuit(model, layer):
    """Locks the W_v matrix so the optimizer must adapt around the surgery."""
    attn = model.transformer.h[layer].attn

    def hook(grad):
        grad_copy = grad.clone()
        grad_copy[:, 2 * D_MODEL:] = 0.0  # Zero out gradients for the W_v slice
        return grad_copy

    return attn.c_attn.weight.register_hook(hook)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    stride, max_len, nlls = 512, 1024, []
    model.eval()
    with torch.no_grad():
        for b in range(0, enc.size(0) - max_len, stride):
            inp = enc[b:b + max_len].unsqueeze(0).to(DEVICE)
            tgt = inp.clone();
            tgt[:, :stride] = -100
            nlls.append(model(input_ids=inp, labels=tgt).loss.item() * (max_len - stride))
    return round(math.exp(sum(nlls) / (len(nlls) * (max_len - stride))), 4)


def measure_lambada_acc(model, tokenizer):
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except:
        ds = load_dataset("lambada", split="test")
    correct, total = 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  LAMBADA", leave=False):
        text = ex.get("text") or ex.get("passage", "")
        tokens = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)["input_ids"].squeeze(0)
        if tokens.shape[0] < 2: continue
        with torch.no_grad():
            logits = model(input_ids=tokens.unsqueeze(0).to(DEVICE)).logits
        if logits[0, -2, :].argmax().item() == tokens[-1].item():
            correct += 1
        total += 1
    return round(correct / total * 100, 2)


def measure_hellaswag_acc(model, tokenizer, n_examples=500):
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n_examples))
    correct, total = 0, 0
    model.eval()
    for ex in tqdm(ds, desc="  HellaSwag", leave=False):
        label = int(ex["label"])
        lls = []
        for ending in ex["endings"]:
            tokens = tokenizer(ex["ctx"] + " " + ending, return_tensors="pt")["input_ids"].squeeze(0).to(DEVICE)
            ctx_len = len(tokenizer(ex["ctx"], return_tensors="pt")["input_ids"].squeeze(0))
            with torch.no_grad():
                outputs = model(input_ids=tokens.unsqueeze(0))
                logits = outputs.logits[0, ctx_len - 1:-1, :]
                loss = torch.nn.functional.cross_entropy(logits, tokens[ctx_len:], reduction='sum')
                lls.append(-loss.item())
        if np.argmax(lls) == label: correct += 1
        total += 1
    return round(correct / total * 100, 2)


def evaluate(model, tokenizer, label="", hswag_n=500):
    print(f"\n  Evaluating: {label}")
    t0 = time.time()
    wt2 = measure_wt2_ppl(model, tokenizer)
    lmda = measure_lambada_acc(model, tokenizer)
    hswag = measure_hellaswag_acc(model, tokenizer, n_examples=hswag_n)
    elapsed = round(time.time() - t0, 1)
    return {"wt2_ppl": wt2, "lambada_acc": lmda, "hellaswag_acc": hswag, "eval_time_sec": elapsed}


# =============================================================================
# DATA STREAMING (OpenWebText)
# =============================================================================

class TokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len=512, max_tokens=5_000_000):
        self.ds, self.tokenizer, self.seq_len, self.max_tokens = hf_dataset, tokenizer, seq_len, max_tokens

    def __iter__(self):
        buffer, total = [], 0
        for ex in self.ds:
            ids = self.tokenizer(ex.get("text", ""), add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)
                total += self.seq_len
                if total >= self.max_tokens: return


def load_ood_corpus(tokenizer):
    print("  Loading OpenWebText (Streaming) for honest healing...")
    ds = load_dataset("openwebtext", split="train", streaming=True)
    next(iter(ds))  # Peek to ensure stream is active
    return TokenDataset(ds, tokenizer)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  THE VICTORY LAP: Targeted Head-Wise Spectral Clipping (L0, H9 & H10)")
    print("=" * 70)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    results_payload = {}

    # 1. Baseline
    print("\n  [1/4] Measuring Pure Baseline...")
    bm = fresh_model()
    baseline = evaluate(bm, tokenizer, label="Baseline")
    results_payload["Baseline"] = baseline
    del bm;
    gc.collect()

    # 2. Surgery Application
    print("\n  [2/4] Applying Spectral Scalpel to L0, Heads 9 & 10 (Top 5)...")
    model = fresh_model()
    frozen_wv_snapshot, diagnostics = apply_targeted_spectral_clipping(model, layer=0, target_heads=[9, 10],
                                                                       clip_count=5)
    results_payload["Diagnostics"] = diagnostics

    # Evaluate Zero-Shot impact
    pre_ft = evaluate(model, tokenizer, label="Pre-FT (Zero-Shot Surgery)")
    results_payload["Pre_FT"] = pre_ft

    # 3. OpenWebText Healing
    print("\n  [3/4] Locking W_v Geometry and Fine-Tuning on OpenWebText...")
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.requires_grad = False
    hook = freeze_wv_circuit(model, layer=0)

    loader = DataLoader(load_ood_corpus(tokenizer), batch_size=DEFAULT_BATCH_SIZE)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DEFAULT_LR)

    model.train()
    for step, (inp, tgt) in enumerate(tqdm(loader, total=DEFAULT_FT_STEPS, desc="  Healing (OOD)")):
        if step >= DEFAULT_FT_STEPS: break
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        optim.zero_grad()
        model(input_ids=inp, labels=tgt).loss.backward()
        optim.step()

        # Hard lock re-injection for safety
        with torch.no_grad():
            W = model.transformer.h[0].attn.c_attn.weight.data.cpu().numpy().copy()
            W[:, 2 * D_MODEL:] = frozen_wv_snapshot
            model.transformer.h[0].attn.c_attn.weight.data = torch.tensor(W, dtype=torch.float32).to(DEVICE)

    hook.remove()

    # 4. Final Evaluation
    print("\n  [4/4] Final Evaluation on Healed Model...")
    post_ft = evaluate(model, tokenizer, label="Post-FT (Healed)")
    results_payload["Post_FT"] = post_ft
    del model;
    gc.collect()

    # Save to JSON
    json_path = "results/victory_lap_results.json"
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=4)

    # Print Terminal Table
    print("\n" + "=" * 70)
    print("  FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Metric':<15} | {'Baseline':<12} | {'Pre-FT (Zero-Shot)':<20} | {'Post-FT (Healed)':<18}")
    print("  " + "-" * 68)

    metrics = [
        ("WT2 PPL", "wt2_ppl", "", False),
        ("LAMBADA Acc", "lambada_acc", "%", True),
        ("HellaSwag Acc", "hellaswag_acc", "%", True)
    ]

    for display_name, key, symbol, higher_is_better in metrics:
        b_val = baseline[key]
        pre_val = pre_ft[key]
        post_val = post_ft[key]

        pre_delta = pre_val - b_val
        post_delta = post_val - b_val

        pre_color_char = "+" if pre_delta > 0 else ""
        post_color_char = "+" if post_delta > 0 else ""

        print(
            f"  {display_name:<15} | {b_val:>6.2f}{symbol:<5} | {pre_val:>6.2f}{symbol} ({pre_color_char}{pre_delta:.2f}{symbol})      | {post_val:>6.2f}{symbol} ({post_color_char}{post_delta:.2f}{symbol})")

    print("=" * 70)
    print(f"  Diagnostics saved to {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()