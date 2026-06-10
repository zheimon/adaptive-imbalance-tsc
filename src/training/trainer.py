"""Trainer for class-imbalanced time-series classification."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import macro_f1, balanced_accuracy, per_class_prf


class Trainer:
    """Training loop for imbalance-aware TSC methods.

    Parameters
    ----------
    model : nn.Module
    method : ImbalanceMethod
        Must implement compute_loss(logits, targets).
    config : dict
        Expected keys:
          config['training']['epochs']: int
          config['training'].get('patience', None): int or None (early stopping)
          config.get('wandb', {}).get('enabled', False): bool
          config.get('run_name', 'run'): str
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        method,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.method = method
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        training_cfg = config.get("training", {})
        self.epochs: int = training_cfg.get("epochs", 10)
        self.patience: Optional[int] = training_cfg.get("patience", None)
        self.run_name: str = config.get("run_name", "run")

        self._wandb_enabled: bool = config.get("wandb", {}).get("enabled", False)

        # Checkpoint directory
        self._ckpt_dir = Path("experiments/checkpoints")
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks=None,
    ) -> dict:
        """Train the model.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
        callbacks : list, optional
            Objects with .on_epoch_end(epoch, signals) method.

        Returns
        -------
        dict with keys 'train_loss', 'val_loss', 'val_macro_f1', 'val_balanced_acc'
            each as list of length n_epochs (actual, respecting early stopping).
        """
        model = self.model
        method = self.method
        device = self.device

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "val_balanced_acc": [],
        }

        best_f1 = -1.0
        no_improve = 0

        for epoch in range(self.epochs):
            # Notify method of epoch if it supports DRW (LDAM)
            if hasattr(method, "set_epoch"):
                method.set_epoch(epoch)

            # ---- Training ----
            model.train()
            train_loss_sum = 0.0
            train_n = 0

            # Per-class accumulators
            n_classes = self._infer_n_classes(train_loader)
            class_loss_sum = np.zeros(n_classes, dtype=np.float64)
            class_counts = np.zeros(n_classes, dtype=np.int64)

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = method.compute_loss(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(y_batch)
                train_n += len(y_batch)

                # Accumulate per-class losses (no_grad for efficiency)
                with torch.no_grad():
                    import torch.nn.functional as F
                    per_sample_loss = F.cross_entropy(
                        logits, y_batch, reduction="none"
                    )
                    for c in range(n_classes):
                        mask = y_batch == c
                        if mask.any():
                            class_loss_sum[c] += per_sample_loss[mask].sum().item()
                            class_counts[c] += mask.sum().item()

            scheduler.step()

            train_loss_avg = train_loss_sum / max(train_n, 1)

            # Per-class mean loss (avoid div by zero)
            per_class_loss = np.where(
                class_counts > 0,
                class_loss_sum / np.maximum(class_counts, 1),
                0.0,
            )

            # ---- Validation ----
            val_metrics = self.evaluate(val_loader)
            per_class_val_f1 = val_metrics.get(
                "per_class_f1", np.zeros(n_classes)
            )

            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_macro_f1"].append(val_metrics["val_macro_f1"])
            history["val_balanced_acc"].append(val_metrics["val_balanced_acc"])

            # ---- Callbacks ----
            signals = {
                "per_class_loss": per_class_loss,
                "per_class_val_f1": per_class_val_f1,
                "per_class_grad_norm": np.ones(n_classes),  # placeholder
            }
            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, signals)

            # ---- W&B logging ----
            if self._wandb_enabled:
                try:
                    import wandb
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss_avg,
                            "val_loss": val_metrics["val_loss"],
                            "val_macro_f1": val_metrics["val_macro_f1"],
                            "val_balanced_acc": val_metrics["val_balanced_acc"],
                        }
                    )
                except ImportError:
                    pass

            # ---- Best checkpoint ----
            current_f1 = val_metrics["val_macro_f1"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                no_improve = 0
                ckpt_path = self._ckpt_dir / f"{self.run_name}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_macro_f1": best_f1,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1

            # ---- Early stopping ----
            if self.patience is not None and no_improve >= self.patience:
                break

        return history

    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate the model on a DataLoader.

        Returns
        -------
        dict with keys:
            'val_loss': float
            'val_macro_f1': float
            'val_balanced_acc': float
            'per_class_prf': pd.DataFrame
            'per_class_f1': np.ndarray (K,)
        """
        model = self.model
        device = self.device

        model.eval()
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        loss_sum = 0.0
        n_total = 0

        import torch.nn.functional as F

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = F.cross_entropy(logits, y_batch, reduction="sum")
                loss_sum += loss.item()
                n_total += len(y_batch)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        prf_df = per_class_prf(y_true, y_pred)

        n_classes = int(y_true.max()) + 1
        per_class_f1 = np.zeros(n_classes, dtype=np.float64)
        for _, row in prf_df.iterrows():
            c = int(row["class"])
            if 0 <= c < n_classes:
                per_class_f1[c] = float(row["f1"])

        return {
            "val_loss": loss_sum / max(n_total, 1),
            "val_macro_f1": macro_f1(y_true, y_pred),
            "val_balanced_acc": balanced_accuracy(y_true, y_pred),
            "per_class_prf": prf_df,
            "per_class_f1": per_class_f1,
        }

    def _infer_n_classes(self, loader: DataLoader) -> int:
        """Infer number of classes from loader dataset."""
        dataset = loader.dataset
        if hasattr(dataset, "n_classes"):
            return int(dataset.n_classes)
        if hasattr(dataset, "tensors"):
            labels = dataset.tensors[1]
            return int(labels.max().item()) + 1
        # Fallback: scan one batch
        for _, y in loader:
            return int(y.max().item()) + 1
        return 2
