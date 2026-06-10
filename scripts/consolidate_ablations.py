"""
consolidate_ablations.py — Aggregate ablation result JSONs into a CSV.

Reads from experiments/ablation_results/ and outputs experiments/ablation_results.csv.
Mirrors the structure of consolidate_results.py but for ablation experiments.

Usage:
    python scripts/consolidate_ablations.py \\
        --results_dir experiments/ablation_results \\
        --output experiments/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_result(path: Path) -> dict | None:
    """Load one result JSON; return None if invalid."""
    try:
        doc = json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] Could not parse {path.name}: {e}")
        return None
    if "test_metrics" not in doc:
        return None
    return doc


def flatten_result(doc: dict) -> dict:
    """Flatten a result doc into a CSV row dict, including ablation-specific fields."""
    cfg = doc.get("config", {})
    exp = cfg.get("experiment", {})
    method_params = cfg.get("method_params", {})
    test_m = doc.get("test_metrics", {})
    per_class_f1 = test_m.get("per_class_f1", {})

    row = {
        "experiment_name": doc.get("experiment_name", ""),
        "method": exp.get("method", ""),
        "dataset": exp.get("dataset", ""),
        "architecture": exp.get("architecture", ""),
        "seed": exp.get("seed", ""),
        # Ablation-specific fields
        "ablation_variant": method_params.get("ablation_variant", exp.get("method", "")),
        "ablation_type": _infer_ablation_type(exp, method_params),
        "update_every": method_params.get("update_every", ""),
        "eta": method_params.get("eta", ""),
        "alpha": str(method_params.get("alpha", "")),
        "sensitivity_param": method_params.get("sensitivity_param", ""),
        "sensitivity_value": method_params.get("sensitivity_value", ""),
        # Metrics
        "macro_f1": test_m.get("macro_f1", ""),
        "balanced_accuracy": test_m.get("balanced_accuracy", ""),
        "g_mean": test_m.get("g_mean", ""),
    }

    # per-class F1 columns (up to 10 classes)
    for i in range(10):
        key = f"per_class_f1_{i}"
        row[key] = per_class_f1.get(i, per_class_f1.get(str(i), ""))

    row["duration_seconds"] = doc.get("duration_seconds", "")
    row["timestamp"] = doc.get("timestamp", "")
    return row


def _infer_ablation_type(exp: dict, method_params: dict) -> str:
    """Infer the ablation type from config fields."""
    sens_param = method_params.get("sensitivity_param", "")
    if sens_param == "update_every":
        return "sensitivity_update_freq"
    if sens_param == "eta":
        return "sensitivity_eta"
    variant = method_params.get("ablation_variant", exp.get("method", ""))
    if variant in ("adacal_no_loss_traj", "adacal_no_grad_norm", "adacal_no_f1_gap"):
        return "component"
    if variant == "adacal":
        return "full"
    return "unknown"


def build_pivot(rows: list[dict]) -> None:
    """Print a summary pivot: variant × dataset → mean macro_f1."""
    from collections import defaultdict

    data: dict[tuple, list[float]] = defaultdict(list)
    variants: set[str] = set()
    datasets: set[str] = set()

    for row in rows:
        if row["macro_f1"] == "":
            continue
        v = row.get("ablation_variant") or row["method"]
        d = row["dataset"]
        variants.add(v)
        datasets.add(d)
        data[(v, d)].append(float(row["macro_f1"]))

    variants_sorted = sorted(variants)
    datasets_sorted = sorted(datasets)

    if not variants_sorted or not datasets_sorted:
        print("No data to pivot.")
        return

    col_w = 16
    header = f"{'variant':<36}" + "".join(f"{d:<{col_w}}" for d in datasets_sorted)
    print("\n" + "=" * len(header))
    print("Pivot: ablation_variant × dataset → mean macro_f1")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for v in variants_sorted:
        row_str = f"{v:<36}"
        for d in datasets_sorted:
            vals = data.get((v, d), [])
            if vals:
                row_str += f"{sum(vals)/len(vals):.4f}{'':<{col_w - 6}}"
            else:
                row_str += f"{'—':<{col_w}}"
        print(row_str)
    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate ablation experiment results into CSV."
    )
    parser.add_argument("--results_dir", default="experiments/ablation_results",
                        help="Directory containing ablation result JSON files.")
    parser.add_argument("--output", default="experiments/ablation_results.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(results_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.stem.endswith("_FAILED")]

    if not json_files:
        print(f"No valid result files found in {results_dir}")
        return

    rows = []
    for p in json_files:
        doc = load_result(p)
        if doc is not None:
            rows.append(flatten_result(doc))

    if not rows:
        print(f"No valid results loaded from {results_dir}")
        return

    # Determine all columns
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")
    build_pivot(rows)


if __name__ == "__main__":
    main()
