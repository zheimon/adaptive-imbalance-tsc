"""
generate_configs.py — Generate the full 4×3×9×3=324 experiment config files.

Usage:
    python scripts/generate_configs.py --output_dir configs/experiments
    python scripts/generate_configs.py --output_dir configs/experiments --filter_method adacal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

DATASETS = ["stock", "ucr_forda", "ucr_electricdevices", "ecg_mitbih"]
ARCHITECTURES = ["lstm", "inception_time", "patchtst"]
METHODS = [
    "ce", "weighted_ce", "focal", "class_balanced",
    "ldam", "logit_adj", "balanced_softmax", "icmlt", "adacal",
]
SEEDS = [0, 1, 2]

# Dataset-specific overrides
DATASET_SEQ_LEN = {
    "stock": 60,
    "ucr_forda": 500,
    "ucr_electricdevices": 96,
    "ecg_mitbih": 280,
}
DATASET_N_CLASSES = {
    "stock": 4,
    "ucr_forda": 2,
    "ucr_electricdevices": 7,
    "ecg_mitbih": 5,
}


def generate_config(method: str, dataset: str, arch: str, seed: int) -> dict:
    """Build a single experiment config dict."""
    exp_name = f"{method}_{dataset}_{arch}_s{seed}"
    return {
        "experiment": {
            "name": exp_name,
            "method": method,
            "dataset": dataset,
            "architecture": arch,
            "seed": seed,
        },
        "data": {
            "seq_len": DATASET_SEQ_LEN[dataset],
            "n_classes": DATASET_N_CLASSES[dataset],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 324 experiment configs.")
    parser.add_argument("--output_dir", default="configs/experiments", help="Directory to write configs")
    parser.add_argument("--filter_method", nargs="+", help="Only generate configs for these methods")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = args.filter_method if args.filter_method else METHODS

    count = 0
    for dataset in DATASETS:
        for arch in ARCHITECTURES:
            for method in methods:
                for seed in SEEDS:
                    cfg = generate_config(method, dataset, arch, seed)
                    exp_name = cfg["experiment"]["name"]
                    out_path = output_dir / f"{exp_name}.yaml"
                    with open(out_path, "w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    count += 1

    total_possible = len(DATASETS) * len(ARCHITECTURES) * len(METHODS) * len(SEEDS)
    if args.filter_method:
        print(f"Generated {count} configs to {output_dir}/ (filtered; full grid would be {total_possible})")
    else:
        print(f"Generated {count} configs to {output_dir}/")


if __name__ == "__main__":
    main()
