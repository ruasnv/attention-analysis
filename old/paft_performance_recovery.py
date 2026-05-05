"""
paft_performance_recovery.py
=======================================================================
Restores the comparison table by evaluating existing saved models.
Tasks: Python, Legal, Medical, English_Web.
=======================================================================
"""

import os
import json
import torch
import safetensors.torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel

# Import your custom project modules
from paft_core import convert_model_to_paft
from data_and_eval import get_dataloader, evaluate_ppl, evaluate_loss, evaluate_accuracy
# Note: Ensure fuse_paft_weights is in your paft_core or available here
from paft_stability_test import fuse_paft_weights

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
TASKS = ["python", "legal", "medical", "english_web"]

MODELS_TO_CHECK = [
    ("Baseline", None),
    ("Standard-FT", "standard_ft"),
    ("Pure-PAFT", "pure_paft"),
    ("Hybrid-PAFT", "hybrid_paft"),
    ("LoRA", "lora")
]


def load_saved_model(task, arm_label, arm_type, tokenizer):
    """Surgically loads models based on their training architecture."""
    if arm_label == "Baseline":
        return GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    path = f"./models/{task}_{arm_type}"
    if not os.path.exists(path):
        print(f"  [!] Missing model: {path}")
        return None

    if arm_label == "Standard-FT":
        return GPT2LMHeadModel.from_pretrained(path).to(DEVICE)

    elif arm_label == "LoRA":
        base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        return PeftModel.from_pretrained(base, path).to(DEVICE)

    elif "PAFT" in arm_label:
        base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        mode = "pure" if "Pure" in arm_label else "hybrid"
        model = convert_model_to_paft(base, mode=mode)

        # Handle different save formats
        sf_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(sf_path):
            sd = safetensors.torch.load_file(sf_path)
        else:
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)

        model.load_state_dict(sd, strict=False)
        model.to(DEVICE)
        # Fusing weights ensures we are testing the "Final" deployed geometry
        return fuse_paft_weights(model)


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    all_recovered_results = {}

    for task in TASKS:
        print(f"\n{'=' * 50}\nRECOVERING RESULTS: {task.upper()}\n{'=' * 50}")
        test_loader = get_dataloader(task, tokenizer, split="test", batch_size=BATCH_SIZE)

        is_accuracy_task = (task == "medical")
        metric_name = "Accuracy" if is_accuracy_task else "Target_Loss"
        task_results = {}

        for arm_label, arm_type in MODELS_TO_CHECK:
            print(f"  -> Evaluating {arm_label}...")
            model = load_saved_model(task, arm_label, arm_type, tokenizer)

            if model is None:
                task_results[arm_label] = {"WT2_PPL": "N/A", metric_name: "N/A"}
                continue

            # Calculate Metrics
            ppl = evaluate_ppl(model, tokenizer, DEVICE)
            if is_accuracy_task:
                perf = evaluate_accuracy(model, tokenizer, task, DEVICE)
            else:
                perf = evaluate_loss(model, test_loader, DEVICE)

            task_results[arm_label] = {
                "WT2_PPL": round(float(ppl), 4),
                metric_name: round(float(perf), 4)
            }

            del model;
            torch.cuda.empty_cache()

        all_recovered_results[task] = task_results

        # Immediate Print for this task
        print(f"\nResults for {task.upper()}:")
        print(f"{'Arm':<15} | {'WT2 PPL':<10} | {metric_name:<10}")
        print("-" * 40)
        for arm, scores in task_results.items():
            print(f"{arm:<15} | {str(scores['WT2_PPL']):<10} | {str(scores[metric_name]):<10}")

    # Final backup save
    with open("expert_results.json", "w") as f:
        json.dump(all_recovered_results, f, indent=4)
    print("\n[Complete] All results saved to expert_results.json")


if __name__ == "__main__":
    main()