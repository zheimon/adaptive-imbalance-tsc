"""
Abstract base class for all datasets used in the adaptive imbalanced
time-series classification (AAAI) research pipeline.

Every dataset must implement:
  - name (property)      : unique string identifier
  - load()               : returns (X, y) with X shaped (N, C, T)
  - All inherited helpers are concrete once load() is defined.
"""

from __future__ import annotations

import abc
from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class BaseDataset(abc.ABC):
    """Abstract base for time-series classification datasets.

    Subclasses must implement :meth:`load` and the :attr:`name` property.
    All other public methods are concrete and rely on ``load()``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, filesystem-safe identifier for the dataset."""

    @abc.abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where X has shape (N, C, T) and y has shape (N,).

        X dtype should be float32, y dtype should be int64.
        """

    # ------------------------------------------------------------------
    # Derived properties (require load())
    # ------------------------------------------------------------------

    @property
    def n_classes(self) -> int:
        """Number of distinct class labels."""
        _, y = self.load()
        return int(np.unique(y).size)

    @property
    def seq_len(self) -> int:
        """Length T of each time series."""
        X, _ = self.load()
        return int(X.shape[2])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_class_distribution(self) -> Dict[int, int]:
        """Return mapping {class_label: count}.

        Returns
        -------
        dict[int, int]
            Keys are integer class labels present in ``y``; values are the
            number of samples belonging to that class.
        """
        _, y = self.load()
        labels, counts = np.unique(y, return_counts=True)
        return {int(lbl): int(cnt) for lbl, cnt in zip(labels, counts)}

    def imbalance_ratio(self) -> float:
        """Return max_class_count / min_class_count.

        A perfectly balanced dataset returns 1.0.
        """
        dist = self.get_class_distribution()
        counts = list(dist.values())
        return float(max(counts) / min(counts))

    def get_splits(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """Return stratified train / val / test index splits.

        The splits satisfy the 80/10/10 default and are guaranteed to be
        disjoint (asserted via set intersection of indices).

        Parameters
        ----------
        val_ratio:
            Fraction of total data reserved for validation (default 0.1).
        test_ratio:
            Fraction of total data reserved for testing (default 0.1).
        random_state:
            Random seed for reproducibility.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
        a dict ``{"X": ndarray, "y": ndarray, "indices": ndarray}``.
        """
        X, y = self.load()
        n = len(y)
        all_indices = np.arange(n)

        # --- Step 1: carve out test set ---
        splitter_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        trainval_idx, test_idx = next(splitter_test.split(all_indices, y))

        # --- Step 2: carve out val set from the remaining trainval ---
        # Adjust val_ratio relative to trainval size
        val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
        splitter_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=random_state
        )
        train_rel, val_rel = next(
            splitter_val.split(trainval_idx, y[trainval_idx])
        )
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]

        # --- Assert zero overlap ---
        assert len(set(train_idx) & set(val_idx)) == 0, (
            "train and val index sets overlap!"
        )
        assert len(set(train_idx) & set(test_idx)) == 0, (
            "train and test index sets overlap!"
        )
        assert len(set(val_idx) & set(test_idx)) == 0, (
            "val and test index sets overlap!"
        )

        def _pack(idx: np.ndarray) -> dict:
            return {"X": X[idx], "y": y[idx], "indices": idx}

        return {
            "train": _pack(train_idx),
            "val": _pack(val_idx),
            "test": _pack(test_idx),
        }
