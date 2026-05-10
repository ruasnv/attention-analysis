"""
data_and_eval.py
=======================================================================
Refined for Triple-Task Benchmarking: Python, Legal, Medical.
Fixed: DataLoader consistency and dynamic table logging.
=======================================================================
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


# =============================================================================
# DATA STREAMING UTILITIES
# =============================================================================

class TokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len=512, text_key="text"):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key

    def __iter__(self):
        buffer = []
        for ex in self.ds:
            text = ex[self.text_key]
            if not text: continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(chunk), torch.tensor(chunk)


def get_dataloader(dataset_name, tokenizer, split="train", batch_size=4, seq_len=512):
    if dataset_name == "python":
        ds = load_dataset("code_search_net", "python", split=split, streaming=True)
        return DataLoader(TokenDataset(ds, tokenizer, seq_len, text_key="whole_func_string"), batch_size=batch_size)
    elif dataset_name == "legal":
        ds = load_dataset("lex_glue", "scotus", split=split, streaming=True)
        return DataLoader(TokenDataset(ds, tokenizer, seq_len, text_key="text"), batch_size=batch_size)
    elif dataset_name == "medical":
        # Load the only available split ("train") and split it 80/20 on the fly
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        ds = ds.train_test_split(test_size=0.2, seed=42)[split]
        return DataLoader(TokenDataset(ds, tokenizer, seq_len, text_key="question"), batch_size=batch_size)
    elif dataset_name == "english_web":
        # OpenWebText is massive, so streaming is mandatory.
        ds = load_dataset("openwebtext", split="train", streaming=True)

        # Create a synthetic train/test split on the stream
        if split == "train":
            # Use the first portion for training
            pass
        else:
            # Skip the first 50,000 examples to create an unseen test set
            ds = ds.skip(50000)

        return DataLoader(TokenDataset(ds, tokenizer, seq_len, text_key="text"), batch_size=batch_size)


def evaluate_accuracy(model, tokenizer, dataset_name="medical", device="cuda", max_samples=100):
    model.eval()
    correct, total = 0, 0
    if dataset_name == "medical":
        # Load the same 20% test split to evaluate accuracy
        full_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        ds = full_ds.train_test_split(test_size=0.2, seed=42)["test"]

        # Make sure we don't exceed max_samples if the test set is smaller
        actual_samples = min(max_samples, len(ds))
        ds = ds.select(range(actual_samples))

        choices = [" yes", " no", " maybe"]
        choice_ids = [tokenizer.encode(c)[0] for c in choices]
        for item in ds:
            context = " ".join(item['context']['contexts'])
            prompt = f"Context: {context}\nQuestion: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                choice_logits = logits[:, -1, choice_ids]
                prediction = torch.argmax(choice_logits, dim=-1).item()
                gt = {"yes": 0, "no": 1, "maybe": 2}[item['final_decision']]
                if prediction == gt: correct += 1
                total += 1
    return round(correct / total, 4) if total > 0 else 0


def evaluate_ppl(model, tokenizer, device="cuda"):
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(ds["text"])
    encodings = tokenizer(test_text, return_tensors="pt")

    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len: break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return round(ppl.item(), 4)


def evaluate_loss(model, dataloader, device="cuda", max_steps=100):
    model.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
            steps += 1
            if steps >= max_steps: break

    return round(total_loss / steps, 4)


# =============================================================================
# LOGGING UTILITY
# =============================================================================
def print_comparison_table(results_dict, task_name="task"):
    # Dynamically determine the label
    metric_label = "Accuracy" if task_name == "medical" else "Target Loss"
    metric_key = "Accuracy" if task_name == "medical" else "Target_Loss"

    header = f"{'Experiment Arm':<25} | {'WT2 PPL (English)':<20} | {f'{metric_label} ({task_name})':<20}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for arm, metrics in results_dict.items():
        ppl = metrics.get("WT2_PPL", "N/A")
        val = metrics.get(metric_key, "N/A")
        print(f"{arm:<25} | {ppl:<20} | {val:<20}")
    print("=" * len(header) + "\n")