#!/usr/bin/env python3
"""
Topological Crossover Analysis

Core finding: Clean learning SIMPLIFIES representation topology (Phi decreases).
Noisy memorization COMPLEXIFIES it (Phi increases). The crossover point where
noisy Phi exceeds clean Phi marks the precise learning-to-memorization transition.

This is the central figure of the paper.

Usage: python3 crossover_analysis.py --input ~/results_phase_transition/ph_phase_transition.csv --outdir ~/project/paper/figures/
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

HOME = os.path.expanduser('~')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {0.0: '#2ca02c', 0.25: '#1f77b4', 0.5: '#ff7f0e', 1.0: '#d62728'}
LABELS = {0.0: 'Clean (0%)', 0.25: '25% noise', 0.5: '50% noise', 1.0: '100% noise'}


def find_crossover(epochs, clean_vals, noisy_vals):
    """Find the epoch where noisy Phi first exceeds clean Phi."""
    for i in range(len(epochs)):
        if noisy_vals[i] > clean_vals[i] and i > 0:
            # Linear interpolation for more precise crossover
            e0, e1 = epochs[i-1], epochs[i]
            c0, c1 = clean_vals[i-1], clean_vals[i]
            n0, n1 = noisy_vals[i-1], noisy_vals[i]
            # Solve: c0 + (c1-c0)*t = n0 + (n1-n0)*t
            dc = c1 - c0
            dn = n1 - n0
            if abs((dn - dc)) > 1e-10:
                t = (c0 - n0) / ((dn - dc))
                crossover_epoch = e0 + t * (e1 - e0)
                crossover_phi = c0 + t * dc
                return crossover_epoch, crossover_phi
            return epochs[i], clean_vals[i]
    return None, None


def plot_main_crossover_figure(df, outdir):
    """The core figure: CNN vs ViT crossover comparison."""
    # Select CIFAR-10 ResNet and ViT for direct comparison
    datasets_to_plot = [
        ('cifar10', 'resnet18', 'CIFAR-10 / ResNet-18 (CNN)'),
        ('cifar10', 'vit_tiny', 'CIFAR-10 / ViT-Tiny'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    crossover_results = []
    
    for idx, (ds, arch, title) in enumerate(datasets_to_plot):
        ax = axes[idx]
        subset = df[(df['dataset'] == ds) & (df['architecture'] == arch)]
        
        if len(subset) == 0:
            ax.set_title(f'{title} (no data)')
            continue
        
        shuffles = sorted(subset['shuffle_frac'].unique())
        clean_data = None
        
        for sf in shuffles:
            sf_data = subset[subset['shuffle_frac'] == sf]
            means = sf_data.groupby('epoch')['total_persistence'].mean().sort_index()
            stds = sf_data.groupby('epoch')['total_persistence'].std().sort_index()
            epochs = means.index.values
            vals = means.values
            
            color = COLORS.get(sf, '#333')
            label = LABELS.get(sf, f'{sf}')
            
            ax.plot(epochs, vals, 'o-', color=color, label=label,
                    markersize=4, linewidth=2)
            ax.fill_between(epochs, vals - stds.values, vals + stds.values,
                           alpha=0.1, color=color)
            
            if sf == 0.0:
                clean_data = (epochs, vals)
            elif sf in [0.25, 0.5] and clean_data is not None:
                # Find crossover
                ce, cv = find_crossover(list(epochs), list(clean_data[1]), list(vals))
                if ce is not None:
                    ax.axvline(x=ce, color=color, linestyle='--', alpha=0.5, linewidth=1)
                    ax.plot(ce, cv, '*', color=color, markersize=15, zorder=5,
                           markeredgecolor='black', markeredgewidth=0.5)
                    ax.annotate(f'Crossover\nt*={ce:.0f}',
                               xy=(ce, cv), xytext=(ce + 3, cv * 1.1),
                               fontsize=8, color=color,
                               arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
                    crossover_results.append({
                        'dataset': ds, 'architecture': arch,
                        'noise_level': sf, 'crossover_epoch': ce,
                        'crossover_phi': cv
                    })
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Total Persistence Φ(t)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
    
    fig.suptitle('The Topological Crossover: Learning vs Memorization',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig_crossover_main.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    
    return crossover_results


def plot_all_datasets_crossover(df, outdir):
    """Extended figure: crossover analysis for all datasets."""
    dataset_groups = list(df.groupby(['dataset', 'architecture']))
    n = len(dataset_groups)
    
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    
    all_crossovers = []
    
    for idx, ((ds, arch), group) in enumerate(dataset_groups):
        ax = axes[idx]
        shuffles = sorted(group['shuffle_frac'].unique())
        clean_data = None
        
        for sf in shuffles:
            sf_data = group[group['shuffle_frac'] == sf]
            means = sf_data.groupby('epoch')['total_persistence'].mean().sort_index()
            stds = sf_data.groupby('epoch')['total_persistence'].std().sort_index()
            epochs = means.index.values
            vals = means.values
            
            color = COLORS.get(sf, '#333')
            label = LABELS.get(sf, f'{sf}')
            
            ax.plot(epochs, vals, 'o-', color=color, label=label,
                    markersize=3, linewidth=1.5)
            ax.fill_between(epochs, vals - stds.values, vals + stds.values,
                           alpha=0.1, color=color)
            
            if sf == 0.0:
                clean_data = (epochs, vals)
            elif clean_data is not None:
                ce, cv = find_crossover(list(epochs), list(clean_data[1]), list(vals))
                if ce is not None:
                    ax.axvline(x=ce, color=color, linestyle='--', alpha=0.4)
                    ax.plot(ce, cv, '*', color=color, markersize=12, zorder=5)
                    all_crossovers.append({
                        'dataset': ds, 'architecture': arch,
                        'noise_level': sf, 'crossover_epoch': round(ce, 1)
                    })
        
        # Determine pattern
        if clean_data is not None:
            peak = max(clean_data[1])
            final = clean_data[1][-1]
            retention = final / (peak + 1e-10)
            if retention > 0.8:
                pattern = 'PERSISTENT'
            elif retention > 0.3:
                pattern = 'FADING'
            else:
                pattern = 'COLLAPSED'
        else:
            pattern = '?'
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Φ(t)')
        ax.set_title(f'{ds}/{arch} [{pattern}]', fontsize=11)
        ax.legend(fontsize=7, loc='best')
    
    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Topological Phase Transition Across Datasets & Architectures',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig_crossover_all_datasets.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    
    return all_crossovers


def plot_direction_analysis(df, outdir):
    """Analyze Phi slope direction: positive (complexifying) vs negative (simplifying)."""
    results = []
    
    for (ds, arch, sf), group in df.groupby(['dataset', 'architecture', 'shuffle_frac']):
        means = group.groupby('epoch')['total_persistence'].mean().sort_index()
        epochs = means.index.values.astype(float)
        vals = means.values
        
        if len(epochs) < 3:
            continue
        
        # Compute slope over different windows
        # Early slope (epoch 1-10)
        early = means[means.index <= 10]
        if len(early) >= 2:
            early_slope = np.polyfit(early.index.values.astype(float), early.values, 1)[0]
        else:
            early_slope = 0
        
        # Late slope (epoch 25-50)
        late = means[means.index >= 25]
        if len(late) >= 2:
            late_slope = np.polyfit(late.index.values.astype(float), late.values, 1)[0]
        else:
            late_slope = 0
        
        # Overall slope
        overall_slope = np.polyfit(epochs, vals, 1)[0]
        
        results.append({
            'dataset': ds, 'architecture': arch, 'shuffle_frac': sf,
            'early_slope': round(early_slope, 2),
            'late_slope': round(late_slope, 2),
            'overall_slope': round(overall_slope, 2),
            'early_direction': 'UP' if early_slope > 1 else 'DOWN' if early_slope < -1 else 'FLAT',
            'late_direction': 'UP' if late_slope > 1 else 'DOWN' if late_slope < -1 else 'FLAT',
        })
    
    rdf = pd.DataFrame(results)
    
    print("\n=== Phi Direction Analysis ===")
    print(f"{'Dataset/Arch':<25} {'Shuffle':<8} {'Early':<8} {'Late':<8} {'Early Dir':<10} {'Late Dir':<10}")
    print("-" * 69)
    for _, r in rdf.iterrows():
        print(f"{r['dataset']}/{r['architecture']:<20} {r['shuffle_frac']:<8} "
              f"{r['early_slope']:<8} {r['late_slope']:<8} "
              f"{r['early_direction']:<10} {r['late_direction']:<10}")
    
    # Key insight: does clean go DOWN and noisy go UP?
    print("\n=== KEY INSIGHT: Learning vs Memorization Direction ===")
    for (ds, arch), group in rdf.groupby(['dataset', 'architecture']):
        clean = group[group['shuffle_frac'] == 0.0]
        noisy = group[group['shuffle_frac'] == 0.5]
        if len(clean) == 0 or len(noisy) == 0:
            continue
        
        clean_late = clean.iloc[0]['late_direction']
        noisy_late = noisy.iloc[0]['late_direction']
        
        if clean_late == 'DOWN' and noisy_late == 'UP':
            verdict = "CROSSOVER CONFIRMED"
        elif clean_late == 'UP' and noisy_late == 'UP':
            verdict = "BOTH COMPLEXIFYING (ViT pattern)"
        elif clean_late == 'DOWN' and noisy_late == 'DOWN':
            verdict = "BOTH SIMPLIFYING"
        else:
            verdict = f"Mixed ({clean_late}/{noisy_late})"
        
        print(f"  {ds}/{arch}: clean_late={clean_late}, noisy_late={noisy_late} → {verdict}")
    
    # Save
    csv_path = os.path.join(outdir, 'crossover_direction_analysis.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    return rdf


def print_crossover_summary(crossovers):
    """Print summary of all crossover points."""
    if not crossovers:
        print("\nNo crossover points found.")
        return
    
    print("\n=== CROSSOVER SUMMARY ===")
    print(f"{'Dataset/Arch':<25} {'Noise':<8} {'Crossover Epoch':<15}")
    print("-" * 48)
    for c in crossovers:
        print(f"{c['dataset']}/{c['architecture']:<20} {c['noise_level']:<8} {c['crossover_epoch']:<15.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(HOME, 'results_phase_transition', 'ph_phase_transition.csv'))
    parser.add_argument('--outdir', default=os.path.join(HOME, 'project', 'paper', 'figures'))
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Datasets: {df.groupby(['dataset','architecture']).ngroups}")
    print(f"Epochs: {sorted(df['epoch'].unique())}")
    print()
    
    # Core figure: CNN vs ViT crossover
    crossovers_main = plot_main_crossover_figure(df, args.outdir)
    
    # All datasets figure
    crossovers_all = plot_all_datasets_crossover(df, args.outdir)
    
    # Direction analysis
    direction_df = plot_direction_analysis(df, args.outdir)
    
    # Summary
    all_crossovers = crossovers_main + crossovers_all
    # Deduplicate
    seen = set()
    unique_crossovers = []
    for c in all_crossovers:
        key = (c['dataset'], c['architecture'], c['noise_level'])
        if key not in seen:
            seen.add(key)
            unique_crossovers.append(c)
    
    print_crossover_summary(unique_crossovers)
    
    # Save crossover data
    if unique_crossovers:
        cdf = pd.DataFrame(unique_crossovers)
        csv_path = os.path.join(args.outdir, 'crossover_points.csv')
        cdf.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    print("\nDONE")


if __name__ == '__main__':
    main()
