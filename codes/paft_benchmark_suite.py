"""
paft_benchmark_suite.py
=======================================================================
Expanded Zero-Shot Evaluation Suite using EleutherAI lm-eval harness.
Tests: Context Retention, Factual Recall, Commonsense Reasoning.
=======================================================================
"""

import os
import json
import torch
import safetensors.torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM

from paft_core import convert_model_to_paft

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
BATCH_SIZE = 8

MODELS_TO_TEST = {
    "Baseline": None,
    "LoRA": "./models/english_web_lora",
    "Pure-PAFT": "./models/english_web_pure_paft",
    "Hybrid-PAFT": "./models/english_web_hybrid_paft"
}

# The expanded cognitive profile suite
TASKS = ["lambada_openai", "winogrande", "piqa", "arc_easy", "hellaswag"]

def load_model_for_eval(name, path):
    print(f"\nLoading {name} into memory...")
    if name == "Baseline":
        return GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    elif name == "LoRA":
        base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        return PeftModel.from_pretrained(base, path).to(DEVICE)

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
        return model.to(DEVICE)

def extract_best_metric(task_results):
    """Safely extracts accuracy whether it's stored as 'acc,none' or 'acc_norm,none'"""
    if 'acc_norm,none' in task_results:
        return round(task_results['acc_norm,none'] * 100, 2)
    elif 'acc,none' in task_results:
        return round(task_results['acc,none'] * 100, 2)
    return "N/A"

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    final_scores = {task: {} for task in TASKS}

    for name, path in MODELS_TO_TEST.items():
        print(f"\n{'='*60}\nRUNNING BENCHMARKS: {name}\n{'='*60}")

        try:
            raw_model = load_model_for_eval(name, path)
            hf_lm = HFLM(pretrained=raw_model, tokenizer=tokenizer, batch_size=BATCH_SIZE)

            results = lm_eval.simple_evaluate(
                model=hf_lm,
                tasks=TASKS,
                num_fewshot=0,
                device=DEVICE
            )

            # Extract and store metrics
            for task in TASKS:
                task_metrics = results['results'].get(task, {})
                score = extract_best_metric(task_metrics)
                final_scores[task][name] = score
                print(f"  -> {task.upper()}: {score}%")

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            for task in TASKS:
                final_scores[task][name] = "ERROR"

        del raw_model
        torch.cuda.empty_cache()

    # Print Final Master Table
    print("\n\n" + "="*95)
    header = f"{'Experiment Arm':<18} | " + " | ".join([f"{t[:10]:<10}" for t in TASKS])
    print(header)
    print("-" * 95)

    for name in MODELS_TO_TEST.keys():
        row_str = f"{name:<18} | "
        scores = [f"{str(final_scores[t].get(name, 'N/A')):<10}" for t in TASKS]
        row_str += " | ".join(scores)
        print(row_str)
    print("="*95)

    # Save to disk
    with open("../results/english_evaluation.json", "w") as f:
        json.dump(final_scores, f, indent=4)
    print("\nResults saved to zero_shot_generalization.json")

if __name__ == "__main__":
    main()