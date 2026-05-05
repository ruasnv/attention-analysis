"""
evaluate_checkpoints.py
Loads all saved step checkpoints and evaluates them,
producing a learning curve JSON for each task and arm.
"""

import os
import json
import torch
import safetensors.torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel
from paft_core import convert_model_to_paft
from data_and_eval import get_dataloader, evaluate_ppl, evaluate_loss, evaluate_accuracy

MODEL_NAME = "gpt2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
STEPS      = [250, 500, 1000]

TASKS = ["python", "legal", "medical", "english_web"]

ARMS = [
    ("Standard-FT",  "standard_ft",  "standard"),
    ("Pure-PAFT",    "pure_paft",    "paft_pure"),
    ("Hybrid-PAFT",  "hybrid_paft",  "paft_hybrid"),
    ("LoRA",         "lora",         "lora"),
]

def load_model(task, folder, arm_type, step):
    path = f"./models/{task}_{folder}_step{step}"
    if not os.path.exists(path):
        return None

    base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    if arm_type == "lora":
        return PeftModel.from_pretrained(base, path).to(DEVICE)

    elif arm_type.startswith("paft"):
        mode  = "pure" if "pure" in arm_type else "hybrid"
        model = convert_model_to_paft(base, mode=mode)
        sf    = os.path.join(path, "model.safetensors")
        bin_  = os.path.join(path, "pytorch_model.bin")
        sd    = safetensors.torch.load_file(sf) if os.path.exists(sf) \
                else torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        return model.to(DEVICE)

    else:
        return GPT2LMHeadModel.from_pretrained(path).to(DEVICE)


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    curve_results = {}

    for task in TASKS:
        print(f"\n{'='*50}\nTASK: {task.upper()}\n{'='*50}")
        is_accuracy = (task == "medical")
        metric_key  = "Accuracy" if is_accuracy else "Target_Loss"
        curve_results[task] = {}

        # Evaluate baseline once per task
        baseline = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
        curve_results[task]["Baseline"] = {
            "WT2_PPL":  evaluate_ppl(baseline, tokenizer, DEVICE),
            metric_key: evaluate_accuracy(baseline, tokenizer, task, DEVICE)
                        if is_accuracy else
                        evaluate_loss(baseline,
                                      get_dataloader(task, tokenizer,
                                                     split="test",
                                                     batch_size=BATCH_SIZE),
                                      DEVICE)
        }
        del baseline; torch.cuda.empty_cache()

        for label, folder, arm_type in ARMS:
            print(f"\n  Arm: {label}")
            curve_results[task][label] = {}

            for step in STEPS:
                print(f"    Step {step}...", end=" ")
                model = load_model(task, folder, arm_type, step)

                if model is None:
                    print("NOT FOUND")
                    curve_results[task][label][f"step_{step}"] = None
                    continue

                ppl = evaluate_ppl(model, tokenizer, DEVICE)

                if is_accuracy:
                    metric = evaluate_accuracy(model, tokenizer, task, DEVICE)
                else:
                    fresh  = get_dataloader(task, tokenizer,
                                            split="test", batch_size=BATCH_SIZE)
                    metric = evaluate_loss(model, fresh, DEVICE)

                curve_results[task][label][f"step_{step}"] = {
                    "WT2_PPL":  round(float(ppl),    4),
                    metric_key: round(float(metric),  4),
                }
                print(f"PPL={ppl:.4f} | {metric_key}={metric:.4f}")

                del model; torch.cuda.empty_cache()

    # Save
    with open("checkpoint_curves.json", "w") as f:
        json.dump(curve_results, f, indent=4)
    print("\n\nSaved to checkpoint_curves.json")

    # Print summary table per task
    for task in TASKS:
        metric_key = "Accuracy" if task == "medical" else "Target_Loss"
        print(f"\n{'='*70}")
        print(f"  {task.upper()} — Learning Curves")
        print(f"{'='*70}")
        print(f"{'Arm':<15} | {'Metric':<8} | {'step_250':<10} | {'step_500':<10} | {'step_1000':<10}")
        print("─" * 70)
        for label, data in curve_results[task].items():
            if label == "Baseline":
                bline = data.get(metric_key, "N/A")
                print(f"{'Baseline':<15} | {metric_key:<8} | {'─':<10} | {'─':<10} | {str(bline):<10}")
                continue
            vals = [
                str(data.get(f"step_{s}", {}).get(metric_key, "N/A"))
                for s in STEPS
            ]
            print(f"{label:<15} | {metric_key:<8} | {vals[0]:<10} | {vals[1]:<10} | {vals[2]:<10}")
        print("=" * 70)


if __name__ == "__main__":
    main()