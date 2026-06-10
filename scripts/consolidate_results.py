"""
consolidate_results.py — Aggregate all experiment result JSONs into a CSV and pivot table.

Usage:
    python scripts/consolidate_results.py \\
        --results_dir experiments/results \\
        --output experiments/raw_results.csv
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
    # skip FAILED markers
    if "test_metrics" not in doc:
        return None
    return doc


def flatten_result(doc: dict) -> dict:
    """Flatten a result doc into a CSV row dict."""
    cfg = doc.get("config", {})
    exp = cfg.get("experiment", {})
    test_m = doc.get("test_metrics", {})
    per_class_f1 = test_m.get("per_class_f1", {})

    row = {
        "experiment_name": doc.get("experiment_name", ""),
        "method": exp.get("method", ""),
        "dataset": exp.get("dataset", ""),
        "architecture": exp.get("architecture", ""),
        "seed": exp.get("seed", ""),
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


def build_pivot(rows: list[dict]) -> None:
    """Print a pivot table: dataset × method → mean macro_f1."""
    from collections import defaultdict

    # Accumulate macro_f1 per (dataset, method)
    data: dict[tuple, list[float]] = defaultdict(list)
    datasets = set()
    methods = set()

    for row in rows:
        if row["macro_f1"] == "":
            continue
        ds = row["dataset"]
        m = row["method"]
        datasets.add(ds)
        methods.add(m)
        data[(ds, m)].append(float(row["macro_f1"]))

    datasets = sorted(datasets)
    methods = sorted(methods)

    if not datasets or not methods:
        print("No data to pivot.")
        return

    # Header
    col_w = 14
    header = f"{'dataset':<24}" + "".join(f"{m:<{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print("Pivot: dataset × method → mean macro_f1 (avg over seeds × architectures)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for ds in datasets:
        row_str = f"{ds:<24}"
        for m in methods:
            vals = data.get((ds, m), [])
            if vals:
                row_str += f"{sum(vals)/len(vals):.4f}{'':<{col_w - 6}}"
            else:
                row_str += f"{'—':<{col_w}}"
        print(row_str)
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Consolidate experiment results into CSV.")
    parser.add_argument("--results_dir", default="experiments/results")
    parser.add_argument("--output", default="experiments/raw_results.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(results_dir.glob("*.json"))
    # Exclude *_FAILED.json
    json_files = [f for f in json_files if not f.stem.endswith("_FAILED")]

    rows = []
    for p in json_files:
        doc = load_result(p)
        if doc is not None:
            rows.append(flatten_result(doc))

    if not rows:
        print(f"No valid result files found in {results_dir}")
        return

    # Determine all columns (union, preserving order)
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
