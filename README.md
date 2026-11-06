# Supplementary Material: Memorization Is Folding

This supplementary material accompanies the paper *Memorization Is Folding:
Topological Signatures of Noisy-Label Learning* (NeurIPS 2026 submission).
It contains the implementation, raw experimental result files, and instructions
needed to reproduce the main results.

## Contents

- `code/`        — Python implementations of training, persistent-homology
                   computation, and analyses
- `results/`     — CSV result files for all experiments reported in the paper
- `figures/`     — (Plotting scripts can be generated from CSVs in `results/`)
- `REPRODUCE.md` — Step-by-step instructions for reproducing each main result
- `requirements.txt` — Python dependencies

## Quick reproduction (no GPU needed for analysis)

The result CSVs are already provided. Most aggregate values reported in the
paper can be reproduced by re-running the analysis scripts in `code/` against
the CSVs in `results/`. See `REPRODUCE.md` for specific commands.

To recompute results from scratch (requires GPU for training experiments),
see the training instructions in `REPRODUCE.md`.

## Software requirements

- Python 3.9+
- PyTorch 2.0+
- ripser (for persistent homology)
- numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels

Install via `pip install -r requirements.txt`.

## Hardware

Training experiments were run on NVIDIA A100 80GB GPUs.
Persistent-homology computation runs on CPU (≤ 0.1s per checkpoint at n=300).

## Data

All datasets used are public: MNIST, FashionMNIST, CIFAR-10/100, AG News,
Tiny-ImageNet, ImageNette, CIFAR-10N. Download instructions are in
`REPRODUCE.md`. We do not redistribute any dataset.

## Note on figure regeneration

All numbers in the paper figures are computed from the CSV result files in
`results/`. The figure-generation scripts are derived from the analysis
scripts in `code/` (e.g., `analyse_classpair_10seed.py`, `exp4_partial_correlation.py`).

## License

The code in this submission is released under the MIT License (see `LICENSE`).
Result CSVs are provided for reproducibility.
