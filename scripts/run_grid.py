"""
run_grid.py — Sequential grid runner with filtering and resume support.

Usage:
    python scripts/run_grid.py \\
        --config_dir configs/experiments \\
        --filter_method adacal focal ce \\
        --filter_dataset ucr_forda \\
        --skip_completed \\
        --max_parallel 1
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---- ensure project root on sys.path ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_name(stem: str) -> dict | None:
    """Parse experiment name: {method}_{dataset}_{arch}_s{seed}."""
    # stem format: ce_stock_lstm_s0
    parts = stem.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    prefix, seed_str = parts
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    # prefix: ce_stock_lstm — split into method_dataset_arch
    # method may contain underscores (e.g., weighted_ce, class_balanced)
    # arch is always last token before seed
    tokens = prefix.split("_")
    if len(tokens) < 3:
        return None
    # architectures are fixed identifiers
    ARCHS = {"lstm", "inception_time", "patchtst"}
    # scan from the right for arch
    arch_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        candidate = tokens[i]
        # inception_time spans two tokens
        if i > 0 and f"{tokens[i-1]}_{tokens[i]}" in ARCHS:
            arch_idx = i - 1
            arch = f"{tokens[i-1]}_{tokens[i]}"
            break
        if candidate in ARCHS:
            arch_idx = i
            arch = candidate
            break
    if arch_idx is None:
        return None
    remaining = tokens[:arch_idx]
    # datasets
    DATASETS = {"stock", "ucr_forda", "ucr_electricdevices", "ecg_mitbih"}
    # scan remaining from right for dataset
    dataset_idx = None
    for i in range(len(remaining) - 1, -1, -1):
        # dataset may be 1 token (stock) or 2 (ucr_forda) or 3 (ucr_electricdevices/ecg_mitbih)
        for width in range(3, 0, -1):
            if i - width + 1 >= 0:
                candidate = "_".join(remaining[i - width + 1: i + 1])
                if candidate in DATASETS:
                    dataset_idx = i
                    dataset = candidate
                    method_tokens = remaining[: i - width + 1]
                    break
        if dataset_idx is not None:
            break
    if dataset_idx is None:
        return None
    method = "_".join(method_tokens)
    return {"method": method, "dataset": dataset, "arch": arch, "seed": seed}


def collect_configs(
    config_dir: Path,
    filter_method: list[str] | None,
    filter_dataset: list[str] | None,
    filter_arch: list[str] | None,
) -> list[Path]:
    """Collect and filter YAML config files."""
    yamls = sorted(config_dir.glob("*.yaml"))
    result = []
    for p in yamls:
        parsed = _parse_name(p.stem)
        if parsed is None:
            # fallback: try loading yaml
            try:
                cfg = _load_yaml(p)
                exp = cfg.get("experiment", {})
                parsed = {
                    "method": exp.get("method", ""),
                    "dataset": exp.get("dataset", ""),
                    "arch": exp.get("architecture", ""),
                    "seed": exp.get("seed", 0),
                }
            except Exception:
                continue
        if filter_method and parsed["method"] not in filter_method:
            continue
        if filter_dataset and parsed["dataset"] not in filter_dataset:
            continue
        if filter_arch and parsed["arch"] not in filter_arch:
            continue
        result.append(p)
    return result


def results_exist(config_path: Path, results_dir: Path) -> bool:
    """Return True if result JSON already exists for this config."""
    stem = config_path.stem
    json_path = results_dir / f"{stem}.json"
    failed_path = results_dir / f"{stem}_FAILED.json"
    return json_path.exists() or failed_path.exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sequential grid runner.")
    parser.add_argument("--config_dir", default="configs/experiments")
    parser.add_argument("--filter_method", nargs="+", help="Only run these methods")
    parser.add_argument("--filter_dataset", nargs="+", help="Only run these datasets")
    parser.add_argument("--filter_arch", nargs="+", help="Only run these architectures")
    parser.add_argument("--skip_completed", action="store_true", help="Skip experiments with existing results")
    parser.add_argument("--max_parallel", type=int, default=1, help="(Currently only 1 is supported)")
    parser.add_argument("--results_dir", default="experiments/results")
    parser.add_argument("--dry_run", action="store_true", help="Pass --dry_run to each experiment")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = collect_configs(
        config_dir,
        filter_method=args.filter_method,
        filter_dataset=args.filter_dataset,
        filter_arch=args.filter_arch,
    )

    if not configs:
        print("No configs matched the filters.")
        sys.exit(0)

    # Filter already-completed
    if args.skip_completed:
        before = len(configs)
        configs = [c for c in configs if not results_exist(c, results_dir)]
        print(f"[skip_completed] {before - len(configs)} already done; {len(configs)} remaining.")

    print(f"Running {len(configs)} experiments (max_parallel={args.max_parallel})\n")

    # CSV log
    log_path = results_dir.parent / "run_log.csv"
    log_fields = ["config_file", "status", "duration", "macro_f1", "timestamp"]
    log_file = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_path.stat().st_size == 0:
        writer.writeheader()

    python_exe = str(Path(sys.executable))
    runner_script = str(Path(__file__).resolve().parent / "run_experiment.py")

    n_completed = 0
    n_failed = 0
    best_f1 = -1.0
    best_name = ""

    iterator = tqdm(configs, unit="exp") if HAS_TQDM else configs

    for cfg_path in iterator:
        cmd = [python_exe, runner_script, "--config", str(cfg_path)]
        if args.dry_run:
            cmd.append("--dry_run")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False)
        duration = round(time.time() - t0, 2)

        status = "completed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            n_failed += 1
        else:
            n_completed += 1

        # Try to read macro_f1 from result JSON
        macro_f1 = ""
        json_path = results_dir / f"{cfg_path.stem}.json"
        if json_path.exists():
            try:
                import json
                doc = json.loads(json_path.read_text())
                macro_f1 = doc.get("test_metrics", {}).get("macro_f1", "")
                if macro_f1 != "" and float(macro_f1) > best_f1:
                    best_f1 = float(macro_f1)
                    best_name = cfg_path.stem
            except Exception:
                pass

        writer.writerow({
            "config_file": cfg_path.name,
            "status": status,
            "duration": duration,
            "macro_f1": macro_f1,
            "timestamp": datetime.utcnow().isoformat(),
        })
        log_file.flush()

    log_file.close()

    print(f"\n{'='*50}")
    print(f"  Completed : {n_completed}")
    print(f"  Failed    : {n_failed}")
    if best_f1 >= 0:
        print(f"  Best F1   : {best_f1:.4f}  ({best_name})")
    print(f"  Log       : {log_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
