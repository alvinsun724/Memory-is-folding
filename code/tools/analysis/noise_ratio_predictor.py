#!/usr/bin/env python3
"""
CTIS — Zero-Shot Noise Ratio Predictor
Uses epoch 10 total persistence to predict label noise ratio.
No clean reference data needed — purely from training dynamics.

Usage: python3 noise_ratio_predictor.py --input ~/results/ph_results_REAL.csv --outdir ~/project/paper/figures/
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

HOME = os.path.expanduser('~')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(HOME, 'results', 'ph_results_REAL.csv'))
    parser.add_argument('--phase_input', default=os.path.join(HOME, 'results_phase_transition', 'ph_phase_transition.csv'))
    parser.add_argument('--outdir', default=os.path.join(HOME, 'project', 'paper', 'figures'))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Try phase transition data first (more epochs), fall back to original
    if os.path.exists(args.phase_input):
        df = pd.read_csv(args.phase_input)
        print(f"Using phase transition data: {len(df)} rows")
    elif os.path.exists(args.input):
        df = pd.read_csv(args.input)
        print(f"Using original data: {len(df)} rows")
    else:
        print("No data found")
        return

    # Use epoch 10 data
    ep10 = df[df['epoch'] == 10].copy()
    if len(ep10) == 0:
        # Try closest available epoch
        available = sorted(df['epoch'].unique())
        closest = min(available, key=lambda x: abs(x - 10))
        ep10 = df[df['epoch'] == closest].copy()
        print(f"Epoch 10 not found, using epoch {closest}")

    print(f"Epoch 10 data: {len(ep10)} samples")

    # Normalize persistence within each dataset/architecture
    ep10['persistence_norm'] = 0.0
    for (ds, arch), group in ep10.groupby(['dataset', 'architecture']):
        max_p = group['total_persistence'].max()
        if max_p > 0:
            ep10.loc[group.index, 'persistence_norm'] = group['total_persistence'] / max_p

    # Train a simple predictor: persistence_norm → shuffle_frac
    X = ep10[['persistence_norm']].values
    y = ep10['shuffle_frac'].values

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    rho, pval = spearmanr(y, y_pred)

    print(f"\nNoise Ratio Predictor (Linear, from epoch 10 persistence):")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Spearman ρ = {rho:.4f} (p={pval:.2e})")
    print(f"  Formula: noise_ratio = {reg.coef_[0]:.4f} * normalized_persistence + {reg.intercept_:.4f}")

    # Per-dataset performance
    print(f"\n{'Dataset/Arch':<25} {'N':<6} {'R²':<8} {'RMSE':<8} {'ρ':<8}")
    print("-" * 55)
    for (ds, arch), group in ep10.groupby(['dataset', 'architecture']):
        if len(group) < 4:
            continue
        Xi = group[['persistence_norm']].values
        yi = group['shuffle_frac'].values
        yi_pred = reg.predict(Xi)
        r2i = r2_score(yi, yi_pred)
        rmsei = np.sqrt(mean_squared_error(yi, yi_pred))
        ri, _ = spearmanr(yi, yi_pred)
        print(f"{ds}/{arch:<20} {len(group):<6} {r2i:<8.4f} {rmsei:<8.4f} {ri:<8.4f}")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: actual vs predicted noise
    ax = axes[0]
    for (ds, arch), group in ep10.groupby(['dataset', 'architecture']):
        Xi = group[['persistence_norm']].values
        yi = group['shuffle_frac'].values
        yi_pred = reg.predict(Xi)
        ax.scatter(yi, yi_pred, alpha=0.5, s=30, label=f'{ds}/{arch}')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Perfect')
    ax.set_xlabel('Actual Noise Ratio')
    ax.set_ylabel('Predicted Noise Ratio')
    ax.set_title(f'Zero-Shot Noise Prediction (R²={r2:.3f}, RMSE={rmse:.3f})')
    ax.legend(fontsize=7)

    # Persistence vs noise ratio per dataset
    ax = axes[1]
    for (ds, arch), group in ep10.groupby(['dataset', 'architecture']):
        means = group.groupby('shuffle_frac')['total_persistence'].mean()
        ax.plot(means.index, means.values, 'o-', label=f'{ds}/{arch}', markersize=6)
    
    ax.set_xlabel('Label Noise Ratio')
    ax.set_ylabel('Total Persistence (epoch 10)')
    ax.set_title('Persistence vs Noise Level')
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(args.outdir, 'noise_ratio_predictor.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: {path}")
    print("DONE")


if __name__ == '__main__':
    main()
