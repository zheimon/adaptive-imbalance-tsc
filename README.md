# Adaptive Imbalance TSC

**Adaptive Methods for Class-Imbalanced Time-Series Classification**

*Submitted to AAAI 2026*

## Abstract

Class imbalance is pervasive in real-world time-series classification (TSC) — ECG arrhythmia
detection, industrial fault diagnosis, and financial signal recognition all exhibit severe
between-class frequency disparities that cause standard cross-entropy-trained models to
collapse toward majority classes. While imbalance methods have been extensively studied for
image data, their efficacy on time-series remains poorly characterized across architectures
and imbalance regimes. We systematically benchmark eight established imbalance methods
(Focal Loss, LDAM-DRW, Class-Balanced Loss, Logit Adjustment, Balanced Softmax, weighted
CE, SMOTE, vanilla CE) across three architectures (LSTM, InceptionTime, PatchTST) and four
datasets (stock signals, UCR FordA, UCR ElectricDevices, MIT-BIH Arrhythmia), then propose
**AdaptiveITSC**, a method that dynamically adjusts per-class margins and re-weights
gradient contributions based on online class-difficulty estimates. AdaptiveITSC achieves
state-of-the-art macro-F1 on all four benchmarks while remaining architecture-agnostic.

## Structure

```
src/
  data/         # Dataset loaders, augmentation, imbalance simulation
  models/       # ConvNet backbone and variants
  methods/      # Loss functions, resampling, adaptive strategies
  training/     # Trainer, device utils, callbacks
  evaluation/   # Metrics, calibration, per-class analysis
configs/        # YAML experiment configs
scripts/        # CLI entry points (train.py, eval.py, sweep.py)
tests/          # Unit & integration tests
experiments/    # Run outputs (gitignored), analysis notebooks
paper/          # LaTeX source
notebooks/      # Exploratory analysis
docs/           # Method documentation
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
python scripts/train.py --config configs/base.yaml
```
