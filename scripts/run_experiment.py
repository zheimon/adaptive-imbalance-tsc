"""
run_experiment.py — Full experiment runner for adaptive-imbalance-tsc.

Usage:
    python scripts/run_experiment.py --config configs/experiments/ce_forda_lstm_s0.yaml
    python scripts/run_experiment.py --config configs/experiments/ce_forda_lstm_s0.yaml --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

# ---- ensure project root on sys.path ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (returns new dict)."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    """Load YAML config, merging with base schema defaults."""
    schema_path = PROJECT_ROOT / "configs" / "experiment_schema.yaml"
    with open(schema_path) as f:
        base = yaml.safe_load(f)
    with open(path) as f:
        override = yaml.safe_load(f)
    return _deep_merge(base, override)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, tuple[int, int]] = {
    # dataset_key: (n_classes, default_seq_len)
    "stock": (4, 60),
    "ucr_forda": (2, 500),
    "ucr_electricdevices": (7, 96),
    "ecg_mitbih": (5, 280),
}


def _get_npz_path(cfg: dict) -> Path:
    data_root = PROJECT_ROOT / cfg["data"]["root"]
    dataset_key = cfg["experiment"]["dataset"]
    seed = cfg["experiment"]["seed"]
    return data_root / f"{dataset_key}_s{seed}.npz"


def _make_synthetic_dataset(n_classes: int, seq_len: int, seed: int, n_samples: int = 400) -> tuple:
    """Create a tiny synthetic dataset for dry-run / offline fallback."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 1, seq_len)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    return X, y


def load_dataset(cfg: dict) -> dict:
    """Load or create dataset splits.

    Returns dict with 'train', 'val', 'test' each having 'X', 'y'.
    Raises RuntimeError on network failure (caller writes _FAILED.json).
    """
    dataset_key = cfg["experiment"]["dataset"]
    seed = cfg["experiment"]["seed"]
    seq_len = cfg["data"]["seq_len"]
    n_classes, _ = DATASET_REGISTRY.get(dataset_key, (2, seq_len))

    npz_path = _get_npz_path(cfg)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if npz_path.exists():
        print(f"[data] Loading cached splits from {npz_path}")
        data = np.load(npz_path)
        return {
            split: {"X": data[f"{split}_X"], "y": data[f"{split}_y"]}
            for split in ("train", "val", "test")
        }

    # Try to load real dataset
    try:
        splits = _load_real_dataset(cfg, dataset_key, seq_len)
    except Exception as exc:
        raise RuntimeError(
            f"Dataset '{dataset_key}' failed to load: {exc}"
        ) from exc

    # Cache to .npz
    save_dict = {}
    for split, d in splits.items():
        save_dict[f"{split}_X"] = d["X"]
        save_dict[f"{split}_y"] = d["y"]
    np.savez_compressed(npz_path, **save_dict)
    print(f"[data] Saved splits to {npz_path}")
    return splits


def _load_real_dataset(cfg: dict, dataset_key: str, seq_len: int) -> dict:
    """Attempt to load a real dataset via src.data modules."""
    # Lazy-import dataset loaders — they may not exist yet for all datasets
    if dataset_key == "ucr_forda":
        from src.data.ucr import FordADataset  # type: ignore
        ds = FordADataset()
    elif dataset_key == "ucr_electricdevices":
        from src.data.ucr import ElectricDevicesDataset  # type: ignore
        ds = ElectricDevicesDataset()
    elif dataset_key == "ecg_mitbih":
        from src.data.ecg import MitBihDataset  # type: ignore
        ds = MitBihDataset()
    elif dataset_key == "stock":
        stock_csv = cfg["data"].get("stock_csv")
        if stock_csv is None:
            raise ValueError("data.stock_csv must be set for dataset=stock")
        from src.data.stock import StockDataset  # type: ignore
        ds = StockDataset(stock_csv, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return ds.get_splits(random_state=cfg["experiment"]["seed"])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict, n_classes: int, in_channels: int) -> torch.nn.Module:
    arch = cfg["experiment"]["architecture"]
    mc = cfg["model"]
    seq_len = cfg["data"]["seq_len"]

    if arch == "lstm":
        from src.models.lstm import LSTMClassifier
        return LSTMClassifier(
            in_channels=in_channels,
            n_classes=n_classes,
            hidden_dim=mc["hidden_dim"],
        )
    elif arch == "inception_time":
        from src.models.inception_time import InceptionTime
        return InceptionTime(
            in_channels=in_channels,
            n_classes=n_classes,
            nb_filters=mc["nb_filters"],
            depth=mc["depth"],
        )
    elif arch == "patchtst":
        from src.models.patchtst import PatchTST
        return PatchTST(
            in_channels=in_channels,
            n_classes=n_classes,
            seq_len=seq_len,
            patch_len=mc["patch_len"],
            stride=mc["stride"],
            d_model=mc["d_model"],
            n_heads=mc["n_heads"],
            n_layers=mc["n_layers"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ---------------------------------------------------------------------------
# Method builder
# ---------------------------------------------------------------------------

def build_method(cfg: dict, n_classes: int, class_counts: np.ndarray | None = None):
    method_key = cfg["experiment"]["method"]
    mp = cfg.get("method_params", {})

    if method_key == "ce":
        from src.methods.cross_entropy import VanillaCE
        return VanillaCE()

    elif method_key == "weighted_ce":
        from src.methods.weighted_ce import WeightedCE
        if class_counts is not None:
            total = class_counts.sum()
            weights = total / (n_classes * class_counts.astype(np.float32))
        else:
            weights = np.ones(n_classes, dtype=np.float32)
        return WeightedCE(class_weights=weights)

    elif method_key == "focal":
        from src.methods.focal_loss import FocalLoss
        return FocalLoss(gamma=float(mp.get("gamma", 2.0)))

    elif method_key == "class_balanced":
        from src.methods.class_balanced import ClassBalancedLoss
        counts = class_counts if class_counts is not None else np.ones(n_classes)
        return ClassBalancedLoss(
            samples_per_class=counts,
            beta=float(mp.get("beta", 0.9999)),
            gamma=float(mp.get("gamma", 2.0)),
        )

    elif method_key == "ldam":
        from src.methods.ldam import LDAMLoss
        counts = class_counts if class_counts is not None else np.ones(n_classes)
        return LDAMLoss(
            cls_num_list=counts,
            max_m=float(mp.get("max_m", 0.5)),
            s=float(mp.get("s", 30)),
            drw_epoch=int(mp.get("drw_epoch", 160)),
        )

    elif method_key == "logit_adj":
        from src.methods.logit_adjustment import LogitAdjustment
        counts = class_counts if class_counts is not None else np.ones(n_classes)
        return LogitAdjustment(
            cls_num_list=counts,
            tau=float(mp.get("tau", 1.0)),
        )

    elif method_key == "balanced_softmax":
        from src.methods.balanced_softmax import BalancedSoftmax
        counts = class_counts if class_counts is not None else np.ones(n_classes)
        return BalancedSoftmax(cls_num_list=counts)

    elif method_key == "icmlt":
        from src.methods.icmlt_baseline import ICMLTBaseline
        counts = class_counts if class_counts is not None else np.ones(n_classes)
        return ICMLTBaseline(cls_num_list=counts)

    elif method_key == "adacal":
        from src.methods.adaptive_hybrid import AdaCAL
        alpha_raw = mp.get("alpha", [0.4, 0.3, 0.3])
        alpha = tuple(float(a) for a in alpha_raw)
        return AdaCAL(
            n_classes=n_classes,
            eta=float(mp.get("eta", 0.1)),
            alpha=alpha,
            k=int(mp.get("k", 5)),
        )

    else:
        raise ValueError(f"Unknown method: {method_key}")


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(splits: dict, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _make(split_data, shuffle: bool) -> DataLoader:
        X = torch.tensor(split_data["X"], dtype=torch.float32)
        y = torch.tensor(split_data["y"], dtype=torch.long)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    return (
        _make(splits["train"], shuffle=True),
        _make(splits["val"], shuffle=False),
        _make(splits["test"], shuffle=False),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test(model, loader, device) -> dict:
    from src.evaluation.metrics import macro_f1, balanced_accuracy, g_mean, per_class_prf
    from sklearn.metrics import confusion_matrix
    import torch.nn.functional as F

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    prf_df = per_class_prf(y_true, y_pred)
    per_class_f1 = {int(row["class"]): float(row["f1"]) for _, row in prf_df.iterrows()}
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "g_mean": g_mean(y_true, y_pred),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run one experiment end-to-end.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--dry_run", action="store_true", help="Instantiate but do not train; print summary and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    exp_name = exp["name"]

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Method     : {exp['method']}")
    print(f"Dataset    : {exp['dataset']}")
    print(f"Arch       : {exp['architecture']}")
    print(f"Seed       : {exp['seed']}")
    if args.dry_run:
        print("[dry_run]  Training will be SKIPPED")
    print(f"{'='*60}\n")

    set_seeds(exp["seed"])

    dataset_key = exp["dataset"]
    n_classes, default_seq_len = DATASET_REGISTRY.get(dataset_key, (2, 128))
    # respect config seq_len override
    seq_len = cfg["data"].get("seq_len", default_seq_len)
    cfg["data"]["seq_len"] = seq_len

    # ---- Dry run: synthetic data, no training ----
    if args.dry_run:
        print(f"[dry_run] Creating synthetic dataset: n_classes={n_classes}, seq_len={seq_len}")
        X_syn, y_syn = _make_synthetic_dataset(n_classes, seq_len, exp["seed"])
        n = len(X_syn)
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        splits = {
            "train": {"X": X_syn[:n_train], "y": y_syn[:n_train]},
            "val": {"X": X_syn[n_train:n_train + n_val], "y": y_syn[n_train:n_train + n_val]},
            "test": {"X": X_syn[n_train + n_val:], "y": y_syn[n_train + n_val:]},
        }
    else:
        # ---- Real dataset loading with fault tolerance ----
        results_dir = Path(cfg["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        failed_path = results_dir / f"{exp_name}_FAILED.json"
        try:
            splits = load_dataset(cfg)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            failed_path.write_text(json.dumps({
                "experiment_name": exp_name,
                "error": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            }, indent=2))
            print(f"[ERROR] Wrote failure marker to {failed_path}")
            sys.exit(1)

    in_channels = splits["train"]["X"].shape[1]
    class_counts = np.bincount(splits["train"]["y"], minlength=n_classes)

    # ---- Build model ----
    model = build_model(cfg, n_classes=n_classes, in_channels=in_channels)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model.name} — total params: {total_params:,}, trainable: {trainable_params:,}")

    # ---- Build method ----
    method = build_method(cfg, n_classes=n_classes, class_counts=class_counts)
    print(f"[method] {method.name}")

    if args.dry_run:
        print("\n[dry_run] Config summary:")
        print(f"  batch_size       : {cfg['training']['batch_size']}")
        print(f"  epochs           : {cfg['training']['epochs']}")
        print(f"  lr               : {cfg['training']['lr']}")
        print(f"  early_stop       : {cfg['training']['early_stop_patience']}")
        print(f"  results_dir      : {cfg['output']['results_dir']}")
        print(f"  checkpoint_dir   : {cfg['output']['checkpoint_dir']}")
        print("\n[dry_run] Complete — no training performed.")
        return

    # ---- Build DataLoaders ----
    batch_size = cfg["training"]["batch_size"]
    train_loader, val_loader, test_loader = build_loaders(splits, batch_size)
    print(f"[data] train={len(splits['train']['y'])}, val={len(splits['val']['y'])}, test={len(splits['test']['y'])}")

    # ---- Build Trainer ----
    from src.training.trainer import Trainer

    trainer_cfg = {
        "training": {
            "epochs": cfg["training"]["epochs"],
            "patience": cfg["training"]["early_stop_patience"],
        },
        "wandb": cfg.get("wandb", {}),
        "run_name": exp_name,
        "output": cfg["output"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    trainer = Trainer(model=model, method=method, config=trainer_cfg, device=device)

    # ---- W&B init ----
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "adaptive-imbalance-tsc"),
                entity=wandb_cfg.get("entity"),
                name=exp_name,
                config=cfg,
            )
        except Exception as e:
            print(f"[wandb] Failed to init: {e}")

    # ---- Train ----
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    duration = time.time() - t0

    # ---- Test evaluation ----
    print("\n[eval] Running test evaluation...")
    test_metrics = evaluate_test(model, test_loader, device)

    # ---- Save results ----
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{exp_name}.json"

    result_doc = {
        "experiment_name": exp_name,
        "config": cfg,
        "train_metrics": {
            "epoch_losses": history["train_loss"],
            "epoch_val_f1": history["val_macro_f1"],
        },
        "test_metrics": test_metrics,
        "duration_seconds": round(duration, 2),
        "hostname": socket.gethostname(),
        "timestamp": datetime.utcnow().isoformat(),
    }

    result_path.write_text(json.dumps(result_doc, indent=2))
    print(f"\n[results] Saved to {result_path}")

    # ---- W&B finish ----
    if wandb_cfg.get("enabled", False):
        try:
            import wandb
            wandb.log(test_metrics)
            wandb.finish()
        except Exception:
            pass

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"{'RESULTS':^60}")
    print(f"{'='*60}")
    print(f"  macro_f1          : {test_metrics['macro_f1']:.4f}")
    print(f"  balanced_accuracy : {test_metrics['balanced_accuracy']:.4f}")
    print(f"  g_mean            : {test_metrics['g_mean']:.4f}")
    print(f"  duration          : {duration:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
