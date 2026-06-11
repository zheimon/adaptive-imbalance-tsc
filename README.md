# AdaCAL — Adaptive Convergence-Aware Loss for Class-Imbalanced Time-Series Classification

**Author:** Anish Kumar Thakur  
**Affiliation:** Department of Computer Science and Engineering, ABV-IIITM Gwalior  
**Venue:** AAAI 2026 (under review)

---

## Overview

Real-world time-series data is rarely balanced. An ECG captures thousands of normal heartbeats for every dangerous arrhythmia; a factory sensor logs days of normal operation before a fault appears. Standard classifiers trained on such data learn to predict the majority class and quietly fail on the cases that matter most.

**AdaCAL** (Adaptive Convergence-Aware Loss) fixes this by monitoring three online signals per class every epoch:

| Signal | What it measures |
|--------|-----------------|
| Loss Plateau Detector | Is the class loss stuck? |
| Gradient Norm Monitor | Is the class gradient stagnating? |
| Validation F1 Gap Tracker | Is the class underperforming vs. the best class? |

These signals drive an exponential multiplicative weight update — classes that are struggling get upweighted dynamically, not just based on their static frequency in the dataset.

---

## Results (mock — update after running experiments)

| Method | Stock | FordA | ElecDevices | MIT-BIH |
|--------|-------|-------|-------------|---------|
| Vanilla CE | 0.612 | 0.781 | 0.551 | 0.718 |
| LDAM-DRW | 0.661 | **0.812** | 0.591 | 0.752 |
| **AdaCAL (ours)** | **0.694** | 0.809 | **0.627** | **0.779** |

Macro-F1, mean over 3 seeds × 3 architectures. AdaCAL is best on 3/4 datasets (Wilcoxon p < 0.05).

---

## Project Structure

```
adaptive-imbalance-tsc/
├── src/
│   ├── data/               # Dataset loaders (Stock, UCR FordA, ElectricDevices, MIT-BIH ECG)
│   ├── models/             # LSTM, InceptionTime, PatchTST
│   ├── methods/            # 9 imbalance methods incl. AdaCAL
│   │   ├── adaptive_hybrid.py      ← AdaCAL (main contribution)
│   │   ├── adaptive_loss_trajectory.py
│   │   ├── adaptive_gradient_norm.py
│   │   ├── adaptive_f1_gap.py
│   │   ├── focal_loss.py
│   │   ├── ldam.py
│   │   ├── class_balanced.py
│   │   └── ...
│   ├── training/           # Trainer loop, callbacks, device utils
│   └── evaluation/         # Metrics (macro-F1, G-mean, Wilcoxon tests)
├── configs/
│   ├── base.yaml                   # Default hyperparameters
│   ├── experiment_schema.yaml      # Full config schema
│   └── experiments/                # 324 auto-generated experiment configs
├── scripts/
│   ├── train.py                    # Single training run
│   ├── run_experiment.py           # End-to-end experiment runner
│   ├── generate_configs.py         # Generate 324-config grid
│   ├── run_grid.py                 # Sequential grid runner
│   ├── prepare_data.py             # Preprocess datasets → .npz
│   ├── run_ablations.py            # Generate 34 ablation configs
│   └── consolidate_results.py      # Merge results → CSV
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_grid_runner_colab.ipynb  # Google Colab runner
│   ├── 03_grid_runner_kaggle.ipynb # Kaggle P100 runner
│   ├── 04_main_results.ipynb       # Headline results table + LaTeX export
│   ├── 05_statistical_significance.ipynb
│   ├── 06_failure_analysis.ipynb
│   ├── 07_ablation_results.ipynb
│   └── 08_convergence_curves.ipynb
├── paper/
│   ├── main.tex                    # AAAI 2026 LaTeX source (8 pages)
│   ├── references.bib              # 14 BibTeX entries
│   ├── aaai2026.sty / .bst         # Official AAAI 2026 style files
│   └── figures/                    # 4 publication-quality vector figures
├── docs/
│   ├── METHOD.md                   # Full AdaCAL math derivation
│   └── DATASETS.md                 # Dataset provenance, licenses, preprocessing
└── tests/                          # 33 unit tests (all passing)
```

---

## Setup

```bash
git clone https://github.com/zheimon/adaptive-imbalance-tsc.git
cd adaptive-imbalance-tsc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Requirements:** Python 3.11, PyTorch 2.x (CUDA or MPS), aeon, wfdb

---

## Prepare Datasets

```bash
# UCR datasets (auto-downloads via aeon)
python scripts/prepare_data.py --datasets ucr_forda ucr_electricdevices --output_dir data/processed

# MIT-BIH ECG (auto-downloads from PhysioNet — no login needed)
python scripts/prepare_data.py --datasets ecg_mitbih --output_dir data/processed

# Stock (supply your own OHLCV CSV)
python scripts/prepare_data.py --datasets stock --stock_csv your_data.csv --output_dir data/processed
```

---

## Run Experiments

**Single experiment:**
```bash
python scripts/run_experiment.py --config configs/experiments/adacal_ecg_mitbih_lstm_s0.yaml
```

**Full grid (324 configs):**
```bash
# Generate configs first (already done — 324 files in configs/experiments/)
python scripts/run_grid.py --config_dir configs/experiments --filter_method adacal --skip_completed
```

**On Kaggle (free P100):** open `notebooks/03_grid_runner_kaggle.ipynb`  
**On Colab:** open `notebooks/02_grid_runner_colab.ipynb`

---

## Reproduce Results

After running experiments:
```bash
# Consolidate all results
python scripts/consolidate_results.py --results_dir experiments/results --output experiments/raw_results.csv

# Run analysis notebooks in order
jupyter notebook notebooks/04_main_results.ipynb
jupyter notebook notebooks/05_statistical_significance.ipynb
jupyter notebook notebooks/07_ablation_results.ipynb
```

LaTeX tables auto-export to `paper/tables/` and are ready to paste into `paper/main.tex`.

---

## Run Tests

```bash
pytest tests/ -v
# Expected: 33 passed
```

---

## Paper

The full AAAI 2026 paper is in `paper/`. To compile:
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Citation

```bibtex
@inproceedings{thakur2026adacal,
  title     = {AdaCAL: Adaptive Convergence-Aware Loss for Class-Imbalanced Time-Series Classification},
  author    = {Thakur, Anish Kumar},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

MIT License. Dataset licenses vary — see `docs/DATASETS.md` for full details.
