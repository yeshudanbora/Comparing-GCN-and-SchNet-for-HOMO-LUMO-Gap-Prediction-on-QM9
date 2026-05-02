# Comparing-GCN-and-SchNet-for-HOMO-LUMO-Gap-Prediction-on-QM9

Course mini-project for **24-788 Introduction to Deep Learning (Spring 2026)**, Carnegie Mellon University.

This repository compares a topology-only Graph Convolutional Network (GCN) baseline against the geometry-aware SchNet architecture on the QM9 HOMO–LUMO gap regression task, and analyzes how each model's performance scales with training-set size.

## Headline results

Trained on the standard 110k / 10k / 10k random split of QM9 (seed 42), reporting test MAE in meV:

| Model | Test MAE (meV) | R² | Parameters | Train time |
|---|---|---|---|---|
| GCN (baseline) | 177.8 | 0.961 | 51,201 | ~58 min |
| SchNet (variant) | **51.7** | **0.996** | 455,809 | ~92 min |

SchNet achieves a 3.4× reduction in test MAE over the GCN baseline.

## Repository contents

```
.
├── qm9_gcn_vs_schnet.ipynb     # Main notebook (training + evaluation + figures)
├── report.tex                  # LaTeX report source
├── report.pdf                  # Compiled report
├── references.bib              # Bibliography
├── neurips_2023.sty            # NeurIPS 2023 style file (provided by course)
├── figures/                    # Figures used in the report
│   ├── training_curves.pdf
│   └── learning_curves.pdf
├── checkpoints/                # Trained model weights
│   ├── gcn_full.pt             # GCN trained on 110k molecules
│   ├── schnet_full.pt          # SchNet trained on 110k molecules
│   ├── gcn_n5000.pt            # learning-curve runs ↓
│   ├── gcn_n20000.pt
│   ├── gcn_n50000.pt
│   ├── schnet_n5000.pt
│   ├── schnet_n20000.pt
│   └── schnet_n50000.pt
├── logs/                       # Training histories (JSON)
│   ├── gcn_full.json
│   ├── schnet_full.json
│   └── learning_curves.json
└── README.md
```

## Environment setup

The project was developed and run on **Kaggle Notebooks** with the **GPU T4 ×2** accelerator (only one T4 is used at a time). Total training wall time across all runs was about 4 hours.

To run the notebook elsewhere, you need:

- Python ≥ 3.10
- PyTorch ≥ 2.5 with CUDA support
- PyTorch Geometric and the `torch_scatter` / `torch_sparse` extension wheels matching your torch + CUDA version
- `matplotlib`, `numpy`

The first cell of the notebook installs PyG and its extensions automatically using the wheel index at `https://data.pyg.org/whl/`. On Kaggle or Google Colab with a GPU runtime, just open the notebook and run all cells.

The QM9 dataset (~300 MB) is downloaded automatically by `torch_geometric.datasets.QM9` the first time the notebook runs into `./qm9_project/data/QM9/`.

### One adapted dependency: SchNet's radius-graph builder

PyTorch Geometric's `SchNet` implementation calls `torch_cluster.radius_graph` to build interaction edges. Because `torch_cluster` is an optional dependency that requires a wheel matching the exact torch + CUDA version (which was awkward in our Kaggle environment), the notebook replaces this builder with a pure-PyTorch equivalent (`_BatchedRadiusGraph` in cell 9) that computes pairwise distances directly. For QM9 molecules (≤29 atoms each), this is functionally equivalent and runs at comparable speed; the only feature it does not enforce is `max_num_neighbors`, which never binds for QM9. This change is the only deviation from the canonical PyG SchNet implementation.

## How to reproduce the report results

### Option A — full retraining (≈ 4 hours total on a T4)

1. Open `qm9_gcn_vs_schnet.ipynb` on Kaggle (or Colab) with a GPU runtime.
2. Run all cells top-to-bottom. The notebook is structured as:
   - **Section 1–4:** environment, data loading, model definitions, training loop
   - **Section 5:** trains GCN on 110k molecules (~58 min), then SchNet on 110k molecules (~92 min). Produces Table 1 of the report and `figures/training_curves.pdf`.
   - **Section 6:** retrains both models on 5k, 20k, 50k molecule subsets (~70 min). Produces Table 2 and `figures/learning_curves.pdf`.
   - **Section 7:** reproducibility script (see below).
3. Saved outputs land in `qm9_project/checkpoints/`, `qm9_project/figures/`, and `qm9_project/logs/`.

### Option B — reproduce key metrics without retraining (recommended for graders)

Run only **Section 7** of `qm9_gcn_vs_schnet.ipynb`, after running Sections 1, 2, 3, and 4 to set up data and model definitions. The `reproduce()` function loads `checkpoints/gcn_full.pt` and `checkpoints/schnet_full.pt` and re-evaluates them on the test split. This runs in well under a minute (no training) and prints:

```
GCN TEST   : {'mae_eV': 0.1778, 'mae_meV': 177.77, 'rmse_eV': 0.2545, 'r2': 0.9612}
SchNet TEST: {'mae_eV': 0.0517, 'mae_meV': 51.66, 'rmse_eV': 0.0803, 'r2': 0.9961}
```

These are the headline numbers reported in Table 1 of the report.

To regenerate the figures from saved logs without retraining: re-run cell 16 (uses `gcn_run` / `schnet_run` history dicts — to recover these from disk, load `logs/gcn_full.json` and `logs/schnet_full.json`) and cell 19 (uses `lc_results` — load from `logs/learning_curves.json`).

## Reproducibility notes

- **Random seed:** 42, fixed for the data split, dataloader shuffle, and model weight init.
- **Target normalization:** targets are scaled to zero mean / unit variance over the training set during training; all reported metrics are computed on the original eV scale.
- **Hardware:** Kaggle T4 GPU. On the same hardware, numbers should reproduce to within ~1 meV across re-runs of the same code.
- **Differences across hardware:** because cuDNN/CUDA non-determinism affects floating-point reductions, exact bit-level reproduction across different GPUs is not guaranteed. Numbers in the same ballpark (within a few meV) should reproduce on any modern NVIDIA GPU.

## Citations and external code

- **PyTorch** (Paszke et al., 2019)
- **PyTorch Geometric** (Fey & Lenssen, 2019) — used for QM9 dataset loader, `GCNConv`, and `SchNet` implementations
- **GCN** baseline architecture follows Kipf & Welling (2017)
- **SchNet** variant architecture follows Schütt et al. (2017, 2018)
- **QM9** dataset from Ramakrishnan et al. (2014), drawn from the GDB-17 chemical universe (Ruddigkeit et al., 2012)

Full citation details are in `references.bib` and the report's References section.

## AI tool use

Claude was used as a debugging assistant during environment setup (resolving a CUDA architecture mismatch on Kaggle's P100 GPU and a missing `torch_cluster` dependency for SchNet's radius-graph builder), and to assist with the LaTeX report scaffolding. All experimental design choices, training runs, results, and analysis are my own. See the Collaboration Statement in the report for full details.
