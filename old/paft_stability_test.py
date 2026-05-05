"""
paft_stability_test.py
=======================================================================
Forensic Analysis Suite for PAFT vs LoRA vs Baseline
Fixed: Native PyTorch state_dict loading for custom PAFT architectures.
=======================================================================
"""

import torch
import time
import os
import numpy as np
import safetensors.torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel
from transformers.pytorch_utils import Conv1D

# Import your core conversion script to build the skeleton
from paft_core import convert_model_to_paft

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

# Fixed the LoRA path to match what the triple-loop script saved
MODELS_TO_TEST = {
    "Baseline": None,
    "Pure-PAFT": "./models/python_pure_paft",
    "Pure-PAFT-Unfused": "./models/python_pure_paft",  # Same path, different loading
    "LoRA": "./models/python_lora",
    "Pure-PAFT-Unfused (Python)": "./models/python_pure_paft",
    "Pure-PAFT-Unfused (Legal)":  "./models/legal_pure_paft",
    "Pure-PAFT-Unfused (EnWeb)":  "./models/english_web_pure_paft",
}

# =============================================================================
# METRIC 1: GEOMETRIC DIVERSITY (STABLE RANK & ENTROPY)
# =============================================================================

def analyze_geometric_diversity(model):
    ranks = []
    entropies = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if "c_attn" in name or "c_proj" in name:

                # Unfused PAFT path — measure S directly
                if hasattr(module, 'S_v') and hasattr(module, 'Q_v'):
                    W = module.S_v.data.float()
                    print(f"  [DEBUG] {name} | S_v mean: {W.mean().item():.6f} | S_v shape: {W.shape}")
                elif hasattr(module, 'S_o') and hasattr(module, 'Q_o'):
                    W = module.S_o.data.float()
                    print(f"  [DEBUG] {name} | S_o mean: {W.mean().item():.6f} | S_o shape: {W.shape}")
                # Fused PAFT and all other models
                elif hasattr(module, 'weight'):
                    W = module.weight.data.float()
                    print(f"  [DEBUG] {name} | weight mean: {W.mean().item():.6f} | shape: {W.shape}")
                else:
                    continue

                U, S, V = torch.svd(W)
                s_vals = S.cpu().numpy()
                stable_rank = (np.sum(s_vals**2)) / (np.max(s_vals)**2)
                ranks.append(stable_rank)
                p = s_vals / np.sum(s_vals)
                entropy = -np.sum(p * np.log(p + 1e-10))
                entropies.append(entropy)

    return np.mean(ranks), np.mean(entropies)

# =============================================================================
# METRIC 2: NEEDLE IN A HAYSTACK (CONTEXT RETRIEVAL)
# =============================================================================

def run_niah_test(model, tokenizer, depth=0.5):
    model.eval()
    needle = "The secret password is: SPECTRAL_SURGERY."
    filler = "The quick brown fox jumps over the lazy dog. " * 20

    full_text = filler[:int(len(filler)*depth)] + needle + filler[int(len(filler)*depth):]
    prompt = full_text + " Question: What is the secret password? Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return "SPECTRAL_SURGERY" in response

# =============================================================================
# METRIC 3: COMPUTE COST (INFERENCE LATENCY)
# =============================================================================

def measure_latency(model, tokenizer, num_runs=50):
    dummy_input = tokenizer("Test latency of the geometric manifold.", return_tensors="pt").to(DEVICE)
    model.eval()

    for _ in range(5): _ = model(**dummy_input)

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**dummy_input)
    end = time.time()

    return (end - start) / num_runs

def fuse_paft_weights(model):
    """
    Fuses the PAFT Q and S matrices back into standard GPT-2 Conv1D layers.
    This eliminates the runtime math tax (Q @ S) and restores native inference speed.
    """
    print("  [PAFT] Fusing geometric stretch (S) into routing manifold (Q)...")

    # Iterate through every transformer block in GPT-2
    for block in model.transformer.h:

        # ==========================================
        # 1. FUSE THE ATTENTION LAYER (c_attn)
        # ==========================================
        # GPT-2's c_attn is a single matrix holding Q, K, and V.
        # PAFT split them to isolate V. We must fuse V and stitch them back together.
        if hasattr(block.attn.c_attn, 'Q_v') and hasattr(block.attn.c_attn, 'S_v'):

            # Perform the geometric fusion: W = Q * S
            W_v_fused = torch.matmul(block.attn.c_attn.Q_v, block.attn.c_attn.S_v)

            # Fetch the frozen Q and K weights
            W_q = block.attn.c_attn.W_q
            W_k = block.attn.c_attn.W_k

            # Stitch Q, K, and the new V back into one massive matrix
            W_fused = torch.cat([W_q, W_k, W_v_fused], dim=-1)

            # Create a brand new, native GPT-2 Conv1D layer
            in_features, out_features = W_fused.shape
            new_c_attn = Conv1D(out_features, in_features).to(W_fused.device)
            new_c_attn.weight.data = W_fused.data

            # Port the bias over if it exists
            if hasattr(block.attn.c_attn, 'bias') and block.attn.c_attn.bias is not None:
                new_c_attn.bias.data = block.attn.c_attn.bias.data

            # Perform the surgical swap
            block.attn.c_attn = new_c_attn

        # ==========================================
        # 2. FUSE THE PROJECTION LAYER (c_proj)
        # ==========================================
        # c_proj is just the Output matrix (O).
        if hasattr(block.attn.c_proj, 'Q_o') and hasattr(block.attn.c_proj, 'S_o'):

            # Perform the geometric fusion: W = Q * S
            W_o_fused = torch.matmul(block.attn.c_proj.Q_o, block.attn.c_proj.S_o)

            # Create a new Conv1D layer
            in_features, out_features = W_o_fused.shape
            new_c_proj = Conv1D(out_features, in_features).to(W_o_fused.device)
            new_c_proj.weight.data = W_o_fused.data

            if hasattr(block.attn.c_proj, 'bias') and block.attn.c_proj.bias is not None:
                new_c_proj.bias.data = block.attn.c_proj.bias.data

            # Perform the surgical swap
            block.attn.c_proj = new_c_proj

    print("  [PAFT] Fusion complete! Model restored to native architecture.")
    return model

# =============================================================================
# EXECUTION
# =============================================================================

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    final_stats = {}

    for name, path in MODELS_TO_TEST.items():
        if path is None:
            print(f"{name}: (pretrained, no path)")
            continue
        # LoRA saves differently than standard models
        sf = os.path.join(path, "adapter_model.safetensors" if "lora" in name.lower() else "model.safetensors")
        if os.path.exists(sf):
            print(f"{name}: {os.path.getsize(sf)} bytes, modified {os.path.getmtime(sf)}")
        else:
            print(f"{name}: [FILE NOT FOUND] {sf}")

        if name == "Baseline":
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

        elif name == "LoRA":
            base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            model = PeftModel.from_pretrained(base, path).to(DEVICE)

        elif "PAFT" in name:
            base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            mode = "pure" if "Pure" in name else "hybrid"
            model = convert_model_to_paft(base, mode=mode)
            safetensors_path = os.path.join(path, "model.safetensors")
            bin_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path)
            else:
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)

            # THIS is the key branch — fuse or don't fuse
            if "Unfused" not in name:
                model = fuse_paft_weights(model)

        # Run Tests
        rank, entropy = analyze_geometric_diversity(model)
        latency = measure_latency(model, tokenizer)

        niah_10 = run_niah_test(model, tokenizer, 0.1)
        niah_50 = run_niah_test(model, tokenizer, 0.5)
        niah_90 = run_niah_test(model, tokenizer, 0.9)

        final_stats[name] = {
            "Stable_Rank": round(rank, 4),
            "Spectral_Entropy": round(entropy, 4),
            "Inference_Latency_ms": round(latency * 1000, 4),
            "NIAH_10%": niah_10,
            "NIAH_50%": niah_50,
            "NIAH_90%": niah_90
        }

        # Free memory before next load
        del model; torch.cuda.empty_cache()

    # Print Results
    print("\n" + "="*85)
    print(f"{'Arm':<15} | {'Stable Rank':<12} | {'Entropy':<12} | {'Latency (ms)':<14} | {'NIAH (Mid)':<8}")
    print("-" * 85)
    for name, s in final_stats.items():
        print(f"{name:<15} | {s['Stable_Rank']:<12} | {s['Spectral_Entropy']:<12} | {s['Inference_Latency_ms']:<14} | {s['NIAH_50%']}")
    print("="*85)

if __name__ == "__main__":
    main()