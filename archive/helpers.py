import json
import os
import numpy as np
from pathlib import Path


def hdr(s):
    print(f"\n{'='*65}\n  {s}\n{'='*65}")


def serialise(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialise(v) for v in obj]
    return obj


def save_json_results(all_results, model_name, prefix, timestamp, results_dir="results"):
    """
    Sanitizes name, saves results to JSON, and prints file size.
    """
    # 1. Ensure directory exists
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Sanitize model name for Windows/Unix file systems
    safe_name = model_name.replace("/", "_")

    # 3. Construct the full path
    file_name = f"{prefix}_{safe_name}_{timestamp}.json"
    file_path = out_dir / file_name

    # 4. Save the data (Assumes 'serialise' is available in your scope)
    # If serialise is in another file, you'll need to import it here.
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serialise(all_results), f, indent=2)

    # 5. Calculate size and report
    size_kb = file_path.stat().st_size / 1024
    print(f"\n  ✓ Results saved → {file_path} ({size_kb:.1f} KB)")

    return str(file_path)

