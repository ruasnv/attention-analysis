"""
run_paft_experiment.py (Triple-Task Production Version)
=======================================================================
Benchmarks: Baseline, Standard-FT, Pure-PAFT, Hybrid-PAFT, LoRA.
Tasks: Python (Loss), Legal (Loss), Medical (Accuracy).
=======================================================================
"""

import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Import our custom modules
from paft_core import convert_model_to_paft
from data_and_eval import get_dataloader, evaluate_ppl, evaluate_loss, print_comparison_table, evaluate_accuracy

# Hyperparameters
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-5
STEPS = 1000
BATCH_SIZE = 4
SEQ_LEN = 512
# Save checkpoints at 250, 500, 1000 steps
checkpoint_steps = [250, 500, 1000]

def train_arm(model, dataloader, steps=1000, lr=2e-5, label="Experiment"):
    print(f"\n  Starting Training: {label}...")
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    current_step = 0
    pbar = tqdm(total=steps, desc=f"  [{label}]")

    while current_step < steps:
        for batch in dataloader:
            if current_step >= steps: break
            input_ids, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            current_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    pbar.close()
    return model

def train_arm_with_checkpoints(model, dataloader, steps=1000, lr=2e-5, label="", save_path=""):
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    current_step = 0
    pbar = tqdm(total=steps, desc=f"  [{label}]")

    while current_step < steps:
        for batch in dataloader:
            if current_step >= steps: break
            input_ids, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            current_step += 1
            pbar.update(1)

            # Save intermediate checkpoints
            if current_step in checkpoint_steps:
                ckpt_path = f"{save_path}_step{current_step}"
                model.save_pretrained(ckpt_path)
                print(f"\n  [Checkpoint saved: {ckpt_path}]")

    pbar.close()
    return model

def main():
    os.makedirs("./models", exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tasks = ["english_web", "python", "legal", "medical"]
    all_task_results = {}

    for task in tasks:
        print(f"\n{'=' * 45}\nSTARTING EXPERIMENT: {task.upper()}\n{'=' * 45}")

        train_loader = get_dataloader(task, tokenizer, split="train", batch_size=BATCH_SIZE)
        # NOTE: No test_loader here anymore — created fresh per evaluation below

        is_accuracy_task = (task == "medical")
        metric_key = "Accuracy" if is_accuracy_task else "Target_Loss"

        def get_current_metric(m):
            if is_accuracy_task:
                return evaluate_accuracy(m, tokenizer, task, DEVICE)
            else:
                # Fresh loader every call — prevents iterator exhaustion
                fresh_loader = get_dataloader(task, tokenizer, split="test", batch_size=BATCH_SIZE)
                return evaluate_loss(m, fresh_loader, DEVICE)

        results = {}

        # --- ARM 1: BASELINE ---
        print(f"\n[{task.upper()}] Evaluating Baseline...")
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
        results["Baseline"] = {
            "WT2_PPL": evaluate_ppl(model, tokenizer, DEVICE),
            metric_key: get_current_metric(model),
            "Model_Path": "N/A"
        }
        del model; torch.cuda.empty_cache()

        # --- ARMS 2-5 ---
        training_types = [
            ("Standard-FT", "full"),
            ("Pure-PAFT", "pure"),
            ("Hybrid-PAFT", "hybrid"),
            ("LoRA", "lora")
        ]

        for arm_label, arm_type in training_types:
            print(f"\n[{task.upper()}] Running {arm_label}...")
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

            if arm_type in ["pure", "hybrid"]:
                model = convert_model_to_paft(model, mode=arm_type)
            elif arm_type == "lora":
                lora_config = LoraConfig(
                    r=8, lora_alpha=32, target_modules=["c_attn"],
                    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", fan_in_fan_out=True
                )
                model = get_peft_model(model, lora_config)

            model.to(DEVICE)

            # save_path defined BEFORE training so checkpoints use it
            save_path = f"./models/{task}_{arm_label.lower().replace('-', '_')}"
            model = train_arm_with_checkpoints(
                model, train_loader, steps=STEPS, lr=LR,
                label=f"{arm_label}-{task}", save_path=save_path
            )

            model.save_pretrained(save_path)

            results[arm_label] = {
                "WT2_PPL": evaluate_ppl(model, tokenizer, DEVICE),
                metric_key: get_current_metric(model),
                "Model_Path": save_path
            }
            del model; torch.cuda.empty_cache()

        all_task_results[task] = results
        print_comparison_table(results, task_name=task)

    with open("triple_expert_results.json", "w") as f:
        json.dump(all_task_results, f, indent=4)
    print("\nExperiment Complete. Results saved to triple_expert_results.json")

if __name__ == "__main__":
    main()