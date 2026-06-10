"""
run_ablations.py — Generate ablation configs and optionally run them.

Usage:
    python scripts/run_ablations.py --output_dir configs/ablations [--run]

Generates 28 total configs:
  - 18 component ablation configs (2 datasets × 1 arch × 3 seeds × 3 ablation variants)
  - 10 sensitivity configs (update_every: 3 configs × 2 datasets + eta: 5 configs × 2 datasets)
    Wait — update_every: 3×2 = 6, eta: 5×2 = 10 ... total =16, but spec says 10.
    Per spec: "Update frequency: every 1/5/10 epochs × 2 datasets × lstm × seed 0 = 6 configs"
    and "Scheduler eta: [0.01,0.05,0.1,0.2,0.5] × 2 datasets × lstm × seed 0 = 10 configs"
    Total sensitivity: 6 + 10 = 16; but spec says "18 + 10 = 28". Interpreting "10" as just
    the eta configs and update_every as included in 18, but following the explicit counts:
    18 component + 6 update_freq + 10 eta... wait spec says "18 + 10 = 28" so we do 18+10.
    Re-reading: "Also generate sensitivity configs: update_every 3 configs × 2 datasets = 6,
    eta 5 × 2 = 10". The spec says "18 + 10 = 28" which doesn't add up to 18+16. We follow
    the stated total of 28: 18 component + 10 sensitivity (combining update_freq and eta into 10).
    Strictly: 3 update_freq × 2 datasets = 6, 5 eta × 2 datasets = 10, but spec writes 10 total.
    We generate all of them (18 component + 6 update_freq + 10 eta = 34), noting the spec
    arithmetic mismatch. The total written = 28, but generating all explicitly described = 34.
    We'll generate all explicitly described configs and note the total.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


# ── Config templates ──────────────────────────────────────────────────────────

BASE_ADACAL_CONFIG = {
    "experiment": {
        "method": "adacal",
        "architecture": "lstm",
        "seed": 0,
        "dataset": "ucr_forda",
    },
    "training": {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
    },
    "method_params": {
        "update_every": 5,
        "eta": 0.05,
        "alpha": [0.33, 0.33, 0.34],
    },
}

# Component ablation variants: name → alpha override
COMPONENT_ABLATIONS = {
    "adacal_no_loss_traj": {
        "description": "AdaCAL with loss trajectory disabled (alpha_loss_traj=0.0)",
        "alpha": [0.0, 0.5, 0.5],
    },
    "adacal_no_grad_norm": {
        "description": "AdaCAL with gradient norm disabled (alpha_grad_norm=0.0)",
        "alpha": [0.5, 0.0, 0.5],
    },
    "adacal_no_f1_gap": {
        "description": "AdaCAL with F1-gap disabled (alpha_f1_gap=0.0)",
        "alpha": [0.5, 0.5, 0.0],
    },
}

ABLATION_DATASETS = ["ucr_forda", "ecg_mitbih"]
ABLATION_ARCH = "lstm"
ABLATION_SEEDS = [0, 1, 2]

SENSITIVITY_UPDATE_FREQS = [1, 5, 10]
SENSITIVITY_ETAS = [0.01, 0.05, 0.1, 0.2, 0.5]


def make_config(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into a copy of base."""
    import copy

    cfg = copy.deepcopy(base)
    for section, values in overrides.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def generate_component_configs(output_dir: Path) -> list[Path]:
    """Generate 18 component ablation configs (3 variants × 2 datasets × 3 seeds)."""
    generated = []

    for variant_name, variant_info in COMPONENT_ABLATIONS.items():
        for dataset in ABLATION_DATASETS:
            for seed in ABLATION_SEEDS:
                cfg = make_config(
                    BASE_ADACAL_CONFIG,
                    {
                        "experiment": {
                            "method": variant_name,
                            "dataset": dataset,
                            "architecture": ABLATION_ARCH,
                            "seed": seed,
                            "description": variant_info["description"],
                        },
                        "method_params": {
                            "alpha": variant_info["alpha"],
                            "update_every": 5,
                            "eta": 0.05,
                            "ablation_variant": variant_name,
                        },
                    },
                )

                fname = f"{variant_name}_{dataset}_lstm_seed{seed}.yaml"
                out_path = output_dir / fname
                out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
                generated.append(out_path)

    print(f"[component ablations] Generated {len(generated)} configs.")
    return generated


def generate_sensitivity_update_freq_configs(output_dir: Path) -> list[Path]:
    """Generate 6 sensitivity configs for update_every ∈ {1,5,10} × 2 datasets."""
    generated = []

    for dataset in ABLATION_DATASETS:
        for uf in SENSITIVITY_UPDATE_FREQS:
            cfg = make_config(
                BASE_ADACAL_CONFIG,
                {
                    "experiment": {
                        "method": "adacal",
                        "dataset": dataset,
                        "architecture": ABLATION_ARCH,
                        "seed": 0,
                        "description": f"Sensitivity: update_every={uf}",
                    },
                    "method_params": {
                        "update_every": uf,
                        "eta": 0.05,
                        "alpha": [0.33, 0.33, 0.34],
                        "sensitivity_param": "update_every",
                        "sensitivity_value": uf,
                    },
                },
            )

            fname = f"adacal_sensitivity_update_every{uf}_{dataset}_lstm_seed0.yaml"
            out_path = output_dir / fname
            out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            generated.append(out_path)

    print(f"[sensitivity update_freq] Generated {len(generated)} configs.")
    return generated


def generate_sensitivity_eta_configs(output_dir: Path) -> list[Path]:
    """Generate 10 sensitivity configs for eta ∈ {0.01,0.05,0.1,0.2,0.5} × 2 datasets."""
    generated = []

    for dataset in ABLATION_DATASETS:
        for eta in SENSITIVITY_ETAS:
            cfg = make_config(
                BASE_ADACAL_CONFIG,
                {
                    "experiment": {
                        "method": "adacal",
                        "dataset": dataset,
                        "architecture": ABLATION_ARCH,
                        "seed": 0,
                        "description": f"Sensitivity: eta={eta}",
                    },
                    "method_params": {
                        "update_every": 5,
                        "eta": eta,
                        "alpha": [0.33, 0.33, 0.34],
                        "sensitivity_param": "eta",
                        "sensitivity_value": eta,
                    },
                },
            )

            eta_str = str(eta).replace(".", "p")
            fname = f"adacal_sensitivity_eta{eta_str}_{dataset}_lstm_seed0.yaml"
            out_path = output_dir / fname
            out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            generated.append(out_path)

    print(f"[sensitivity eta] Generated {len(generated)} configs.")
    return generated


def run_configs(config_paths: list[Path], run_script: Path) -> None:
    """Run each config through run_experiment.py."""
    if not run_script.exists():
        print(f"[error] run_experiment.py not found at {run_script}", file=sys.stderr)
        sys.exit(1)

    for cfg_path in config_paths:
        cmd = [sys.executable, str(run_script), "--config", str(cfg_path)]
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[warn] Experiment failed for {cfg_path.name} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation configs and optionally run them."
    )
    parser.add_argument(
        "--output_dir",
        default="configs/ablations",
        help="Directory to write ablation YAML configs (default: configs/ablations)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="If set, run experiments after generating configs.",
    )
    parser.add_argument(
        "--run_script",
        default="scripts/run_experiment.py",
        help="Path to run_experiment.py (used with --run).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing ablation configs to: {output_dir.resolve()}")
    print()

    all_configs: list[Path] = []
    all_configs.extend(generate_component_configs(output_dir))
    all_configs.extend(generate_sensitivity_update_freq_configs(output_dir))
    all_configs.extend(generate_sensitivity_eta_configs(output_dir))

    print()
    print(f"Total ablation configs generated: {len(all_configs)}")
    print(f"  Component ablations : {len([c for c in all_configs if 'sensitivity' not in c.name])}")
    print(f"  Sensitivity (eta)   : {len([c for c in all_configs if 'sensitivity_eta' in c.name])}")
    print(f"  Sensitivity (uf)    : {len([c for c in all_configs if 'sensitivity_update' in c.name])}")

    if args.run:
        print(f"\nRunning {len(all_configs)} experiments...")
        run_configs(all_configs, Path(args.run_script))
        print("All ablation experiments completed.")
    else:
        print("\nTo run ablations, re-run with --run flag:")
        print(f"  python scripts/run_ablations.py --output_dir {args.output_dir} --run")


if __name__ == "__main__":
    main()
