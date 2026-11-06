# Reproduction Instructions

## Step 0: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 1: Reproduce analysis from provided result CSVs (CPU, minutes)

All result CSVs are in `results/`. To regenerate the analyses reported in the paper:

```bash
cd code

# Class-pair ablation (10 seeds) -- main paper Figure 9
python analyse_classpair_10seed.py

# Per-class topology analysis
python per_class_topology.py

# Partial correlation analysis (Figure 4)
python exp4_partial_correlation.py

# PH summaries
python analyze_ph_results.py
```

Each script prints the aggregated statistics that appear in the paper.

## Step 2: Recompute persistent homology from saved activations (CPU, ~10 min per condition)

This step requires the activation tensors which are too large for the
supplementary archive (>10 GB). They will be released separately upon
de-anonymization.

```bash
python code/compute_ph_new.py --activation_path <path_to_activations> --n_sub 300
```

## Step 3: Retrain from scratch (GPU, hours per condition)

To reproduce a single training run with PH extraction:

```bash
# CIFAR-10 / ResNet-18 / 50% label noise / seed 42
python code/per_class_v2.py --dataset cifar10 --arch resnet18 \
    --noise symmetric --shuffle_frac 0.5 --seed 42 --epochs 50

# CIFAR-10N (realistic noise)
python code/per_class_v2.py --dataset cifar10 --arch resnet18 \
    --noise cifar10n_aggre --seed 42 --epochs 50

# Grokking experiment
python code/train_grokking.py --task modular_addition --p 113 --seed 42
```

Each architecture's hyperparameters match Table 3 of the main paper.

## Compute time estimates

- One ResNet-18 / CIFAR-10 training run: ~2-3 GPU-hours
- One ViT-Tiny training run: ~3-4 GPU-hours
- One DistilBERT / AG News run: ~4 GPU-hours
- Tiny-ImageNet ResNet-18 run: ~6 GPU-hours
- Total reported in paper: ~2,800 GPU-hours

## Result files

| CSV file | Experiment | Section |
|----------|-----------|---------|
| `results_phase_transition/*_metrics.csv` | Per-seed training metrics | All |
| `results_phase_transition/independence_test.csv` | Independence test (rho values) | Sec. 3 |
| `results_phase_transition/ph_phase_transition.csv` | Phi across all conditions | Sec. 3, Fig. 1 |
| `results_classpair_10seed/*/results.csv` | Class-pair ablation (10 seeds) | Sec. 4, Fig. 9 |
| `results_realistic_noise/*_metrics.csv` | Asymmetric / instance / CIFAR-10N noise | Sec. 6 |
| `results_routing_v2/*.csv` | Routing analysis | Appendix |
| `results_methods/*_metrics.csv` | Method comparisons (dropout, label smoothing, etc.) | Sec. 7 |
| `results_largescale/*_metrics.csv` | ResNet-50 large-scale runs | Appendix |
| `results_phase_transition/distilbert_perclass.csv` | DistilBERT per-class PH | Sec. 6 |
| `results_phase_transition/synthetic_2d_results.csv` | Synthetic 2D experiment | Sec. 4 |
| `results_phase_transition/cka_comparison.csv` | CKA comparison | Appendix |
| `results_phase_transition/rtd_comparison.csv` | RTD comparison | Appendix |
| `results_phase_transition/intrinsic_dim_comparison.csv` | Intrinsic dimension | Appendix |
| `results_phase_transition/neural_collapse_results.csv` | Neural collapse analysis | Appendix |
