"""
benchmark.py  —  give it model paths, get benchmark results.

Usage:
    python benchmark.py
"""
import sys, os, json, subprocess
from datetime import datetime

MODELS = {
    "baseline":              None, # loads gpt2 from HuggingFace
    "wv_polar_q":            "phase3_results/20260328_233945/wv_polar_q",
    "wv_pca_proc":           "phase3_results/20260328_233945/wv_pca_proc",
    "wv_pca_raw":            "phase3_results/20260328_233945/wv_pca_raw",
    "wv_random_null":        "phase3_results/20260328_233945/wv_random_null",
    "g_pmi_activation":      "phase3_results/20260328_233945/g_pmi_activation",
    "g_pmi_activation_proc": "phase3_results/20260328_233945/g_pmi_activation_proc",
    "g_pmi_layer":           "phase3_results/20260328_233945/g_pmi_layer",
    "g_pmi_layer_proc":      "phase3_results/20260328_233945/g_pmi_layer_proc",
    "g_random_null":         "phase3_results/20260328_233945/g_random_null",
}

TASKS   = "lambada_openai,arc_challenge,hellaswag,piqa,winogrande"

DEVICE  = "cuda"
OUT_DIR = "phase4_results"

# =============================================================================
# RUN
# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)
results = {}

for name, path in MODELS.items():
    model_args = "pretrained=gpt2" if path is None else f"pretrained={path},dtype=float32"
    out_file   = f"{OUT_DIR}/{name}.json"

    print(f"\n--- {name} ---")
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model",       "hf",
        "--model_args",  model_args,
        "--tasks",       TASKS,
        "--device",      DEVICE,
        "--batch_size",  "auto",
        "--output_path", out_file,
    ]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"  FAILED (code={ret})")
        results[name] = None
    else:
        results[name] = out_file

# =============================================================================
# TABLE
# =============================================================================

def get(path, task, metric):
    if path is None or not os.path.exists(path):
        return float("nan")
    m = json.load(open(path)).get("results", {}).get(task, {})
    v = m.get(metric) or m.get("acc,none")
    return round(v * 100, 1) if v is not None else float("nan")

def ppl(path, task):
    if path is None or not os.path.exists(path):
        return float("nan")
    m = json.load(open(path)).get("results", {}).get(task, {})
    v = m.get("perplexity,none") or m.get("word_perplexity,none")
    return round(v, 2) if v is not None else float("nan")

print(f"\n{'='*95}")
print(f"  {'Model':<30} {'LMDA-PPL':>9} {'LMDA-Acc':>9} {'ARC-C':>7} {'HSwag':>7} {'PIQA':>7} {'Wino':>7}")
print(f"  {'─'*93}")

rows = {}
for name, out_file in results.items():
    lp  = ppl(out_file, "lambada_openai")
    la  = get(out_file, "lambada_openai", "acc,none")
    arc = get(out_file, "arc_challenge",  "acc_norm,none")
    hs  = get(out_file, "hellaswag",      "acc_norm,none")
    pi  = get(out_file, "piqa",           "acc_norm,none")
    wi  = get(out_file, "winogrande",     "acc,none")
    rows[name] = (lp, la, arc, hs, pi, wi)
    print(f"  {name:<30} {lp:>9.2f} {la:>8.1f}% {arc:>6.1f}% {hs:>6.1f}% {pi:>6.1f}% {wi:>6.1f}%")

# Delta vs baseline
if "baseline" in rows:
    b = rows["baseline"]
    print(f"\n  {'─'*93}")
    print(f"  Delta vs baseline")
    print(f"  {'─'*93}")
    for name, v in rows.items():
        if name == "baseline":
            continue
        print(f"  {name:<30} {v[0]-b[0]:>+9.2f} {v[1]-b[1]:>+8.1f}% "
              f"{v[2]-b[2]:>+6.1f}% {v[3]-b[3]:>+6.1f}% "
              f"{v[4]-b[4]:>+6.1f}% {v[5]-b[5]:>+6.1f}%")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump(rows, open(f"{OUT_DIR}/summary_{ts}.json", "w"), indent=2)
print(f"\n  Saved: {OUT_DIR}/summary_{ts}.json")