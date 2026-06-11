#!/usr/bin/env python3
"""Generate ablation configs and optionally run them.

Usage:
    python scripts/run_ablations.py --output_dir configs/ablations [--run]
"""
import argparse, os, yaml, subprocess, sys
from pathlib import Path

DATASETS = ['ucr_forda', 'ecg_mitbih']
ARCH = 'lstm'
SEEDS = [0, 1, 2]

COMPONENT_ABLATIONS = {
    'adacal_no_loss_traj': {'eta': 0.1, 'alpha': [0.0, 0.5, 0.5]},
    'adacal_no_grad_norm': {'eta': 0.1, 'alpha': [0.5, 0.0, 0.5]},
    'adacal_no_f1_gap':    {'eta': 0.1, 'alpha': [0.5, 0.5, 0.0]},
}

UPDATE_EVERY_VALUES = [1, 5, 10]
ETA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]

DATASET_SEQLEN = {'ucr_forda': 500, 'ecg_mitbih': 280}
DATASET_NCLASSES = {'ucr_forda': 2, 'ecg_mitbih': 5}

def make_config(variant_name, dataset, arch, seed, method_params_override):
    name = f"{variant_name}_{dataset}_{arch}_s{seed}"
    return {
        'experiment': {'name': name, 'method': 'adacal', 'dataset': dataset,
                       'architecture': arch, 'seed': seed},
        'data': {'root': 'data/processed', 'seq_len': DATASET_SEQLEN[dataset],
                 'n_classes': DATASET_NCLASSES[dataset]},
        'method_params': method_params_override,
        'training': {'epochs': 100, 'batch_size': 64, 'lr': 1e-3,
                     'weight_decay': 1e-4, 'early_stop_patience': 20},
        'output': {'results_dir': 'experiments/ablation_results',
                   'checkpoint_dir': 'experiments/ablation_checkpoints'},
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='configs/ablations')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configs_generated = []

    # Component ablations
    for variant, mp in COMPONENT_ABLATIONS.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                cfg = make_config(variant, dataset, ARCH, seed, mp)
                path = Path(args.output_dir) / f"{cfg['experiment']['name']}.yaml"
                with open(path, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                configs_generated.append(str(path))

    # Update frequency sensitivity
    for ue in UPDATE_EVERY_VALUES:
        for dataset in DATASETS:
            mp = {'eta': 0.1, 'alpha': [0.4, 0.3, 0.3], 'update_every': ue}
            cfg = make_config(f'adacal_ue{ue}', dataset, ARCH, 0, mp)
            path = Path(args.output_dir) / f"{cfg['experiment']['name']}.yaml"
            with open(path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
            configs_generated.append(str(path))

    # Eta sensitivity
    for eta in ETA_VALUES:
        for dataset in DATASETS:
            mp = {'eta': eta, 'alpha': [0.4, 0.3, 0.3]}
            eta_str = str(eta).replace('.', 'p')
            cfg = make_config(f'adacal_eta{eta_str}', dataset, ARCH, 0, mp)
            path = Path(args.output_dir) / f"{cfg['experiment']['name']}.yaml"
            with open(path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
            configs_generated.append(str(path))

    print(f"Generated {len(configs_generated)} ablation configs to {args.output_dir}/")

    if args.run:
        for cfg_path in configs_generated:
            print(f"\nRunning: {cfg_path}")
            ret = subprocess.run([sys.executable, 'scripts/run_experiment.py', '--config', cfg_path])
            if ret.returncode != 0:
                print(f"  FAILED: {cfg_path}")

if __name__ == '__main__':
    main()
