"""
g_matrix_clean_healing.py
=======================================================================
G-Matrix Refinement: Honest OOD Healing
=======================================================================
Healing: OpenWebText (Train) - Diversity-driven adaptation
Evaluation: WikiText-2 (Test) - Unseen, clean fluency benchmark
"""

import gc
import math
import time
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME   = "gpt2"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS     = 12
D_MODEL      = 768
N_HEADS      = 12
D_HEAD       = D_MODEL // N_HEADS

DEFAULT_FT_STEPS   = 1000
DEFAULT_LR         = 2e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN    = 512
WARMUP_STEPS       = 50

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def fresh_model():
    # Using 'dtype' instead of deprecated 'torch_dtype'
    return GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.float32).eval().to(DEVICE)

def inject_boosted_g_negative(model, layer, alpha):
    """Surgically amplifies ONLY negative eigenvalues (Repulsion) in the QK circuit."""
    W = model.transformer.h[layer].attn.c_attn.weight.data.cpu().numpy()
    W_q = W[:, :D_MODEL]
    W_k = W[:, D_MODEL:2*D_MODEL]
    W_q_new = np.zeros_like(W_q)

    for h in range(N_HEADS):
        wq = W_q[:, h*D_HEAD:(h+1)*D_HEAD]
        wk = W_k[:, h*D_HEAD:(h+1)*D_HEAD]

        # G_small (64x64) shares eigenvalues with the full bilinear form
        G_small = wk.T @ wq
        vals, vecs = np.linalg.eig(G_small)

        vals_new = vals.copy()
        mask = vals.real < 0
        vals_new[mask] *= alpha

        # Reconstruct G and project back onto W_q
        G_small_new = vecs @ np.diag(vals_new) @ np.linalg.inv(vecs)
        G_small_new = np.real(G_small_new)

        wq_new_h = wk @ np.linalg.pinv(wk.T @ wk) @ G_small_new
        W_q_new[:, h*D_HEAD:(h+1)*D_HEAD] = wq_new_h

    W_new = W.copy()
    W_new[:, :D_MODEL] = W_q_new
    model.transformer.h[layer].attn.c_attn.weight.data = torch.tensor(W_new, dtype=torch.float32).to(DEVICE)
    return W_q_new

def freeze_qk_circuit(model, layer):
    """Locks the QK circuit at its new geometry so the OV circuit must adapt."""
    attn = model.transformer.h[layer].attn
    def hook(grad):
        grad_copy = grad.clone()
        grad_copy[:, :2*D_MODEL] = 0.0 # Freeze W_q and W_k slices
        return grad_copy
    return attn.c_attn.weight.register_hook(hook)

# =============================================================================
# EVALUATION METRICS (WikiText-2 Test / LAMBADA / HellaSwag)
# =============================================================================

def measure_wt2_ppl(model, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    stride, max_len, nlls = 512, 1024, []
    model.eval()
    with torch.no_grad():
        for b in range(0, enc.size(0) - max_len, stride):
            inp = enc[b:b+max_len].unsqueeze(0).to(DEVICE)
            tgt = inp.clone(); tgt[:, :stride] = -100
            nlls.append(model(input_ids=inp, labels=tgt).loss.item() * (max_len - stride))
    return round(math.exp(sum(nlls) / (len(nlls) * (max_len - stride))), 4)

def measure_lambada_acc(model, tokenizer):
    try: ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except: ds = load_dataset("lambada", split="test")
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
                logits = outputs.logits[0, ctx_len-1:-1, :]
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
    print(f"    WT2 PPL:       {wt2}")
    print(f"    LAMBADA Acc:   {lmda}%")
    print(f"    HellaSwag Acc: {hswag}%")
    return {"wt2_ppl": wt2, "lambada_acc": lmda, "hellaswag_acc": hswag}

# =============================================================================
# DATA STREAMING (OpenWebText - Honest OOD Healing)
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
    """Loads OpenWebText for healing to ensure zero data leakage with WT2."""
    print("  Loading OpenWebText (Streaming) for honest healing...")
    try:
        ds = load_dataset("openwebtext", split="train", streaming=True)
        next(iter(ds)) # Validation peek
        return TokenDataset(ds, tokenizer)
    except Exception as e:
        print(f"  OpenWebText fail ({e}). Falling back to WikiText-103...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        return TokenDataset(ds, tokenizer)

# =============================================================================
# REFINEMENT ENGINE
# =============================================================================

def run_refinement(tokenizer, baseline, layers, alpha, label):
    print("\n" + "="*65); print(f"  RUNNING: {label}"); print(f"  Layers: {layers} | Alpha: {alpha}"); print("="*65)

    model = fresh_model()
    wq_snapshots = {l: inject_boosted_g_negative(model, l, alpha) for l in layers}

    # 1. Pre-Healing Baseline
    pre_ft = evaluate(model, tokenizer, label=f"{label}_pre_ft")

    # 2. Freeze Navigational Circuits
    model.transformer.wte.weight.requires_grad = False
    model.transformer.wpe.weight.requires_grad = False
    hooks = [freeze_qk_circuit(model, l) for l in layers]

    # 3. Heal on Out-of-Distribution Data
    loader = DataLoader(load_ood_corpus(tokenizer), batch_size=DEFAULT_BATCH_SIZE)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DEFAULT_LR)

    model.train()
    for step, (inp, tgt) in enumerate(tqdm(loader, total=DEFAULT_FT_STEPS, desc="  Healing")):
        if step >= DEFAULT_FT_STEPS: break
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        optim.zero_grad(); model(input_ids=inp, labels=tgt).loss.backward(); optim.step()

        # Hard Geometric Lock: Re-inject to prevent any optimizer drift in frozen QK
        with torch.no_grad():
            for l, snap in wq_snapshots.items():
                W = model.transformer.h[l].attn.c_attn.weight.data.cpu().numpy().copy()
                W[:, :D_MODEL] = snap
                model.transformer.h[l].attn.c_attn.weight.data = torch.tensor(W, dtype=torch.float32).to(DEVICE)

    for h in hooks: h.remove()
    post_ft = evaluate(model, tokenizer, label=f"{label}_post_ft")
    del model; gc.collect()

    # Print comparison vs original model
    print(f"\n  {label} HONEST SUMMARY:")
    print(f"  WT2 PPL:     {baseline['wt2_ppl']} (Orig) -> {post_ft['wt2_ppl']} (Healed)")
    print(f"  LAMBADA Acc: {baseline['lambada_acc']}% (Orig) -> {post_ft['lambada_acc']}% (Healed)")
    print(f"  HellaSwag:   {baseline['hellaswag_acc']}% (Orig) -> {post_ft['hellaswag_acc']}% (Healed)")

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n  Measuring Pure Baseline...")
    bm = fresh_model(); baseline = evaluate(bm, tokenizer, label="baseline"); del bm; gc.collect()

    # Leg 1: The Gentle Scalpel
    run_refinement(tokenizer, baseline, [1], 1.25, "Gentle_Scalpel_L1")

    # Leg 2: The Distributed Filter
    run_refinement(tokenizer, baseline, [1, 2, 3], 1.2, "Distributed_Filter_L123")

if __name__ == "__main__":
    main()