#!/usr/bin/env python3
"""
CTIS — PH Sensitivity Analysis
Tests whether topological findings are robust to:
1. Different subsample sizes (100, 200, 300, 500)
2. Different random seeds for subsampling

Usage: python3 ph_sensitivity.py --act_dir ~/results/activations --outdir ~/project/paper/figures/
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HOME = os.path.expanduser('~')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Check if ripser is available
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def compute_ph(activations, subsample_size, seed=42):
    """Compute persistent homology with given subsample size."""
    if not HAS_RIPSER:
        return None, None, None
    
    if len(activations) > subsample_size:
        idx = np.random.RandomState(seed).choice(len(activations), subsample_size, replace=False)
        activations = activations[idx]
    
    result = ripser(activations, maxdim=1)
    diagrams = result['dgms']
    
    total_persistence = 0
    b0 = 0
    b1 = 0
    
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        persistence = np.sum(finite[:, 1] - finite[:, 0])
        total_persistence += persistence
        if dim == 0:
            b0 = len(finite)
        elif dim == 1:
            b1 = len(finite)
    
    return total_persistence, b0, b1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_dir', default=os.path.join(HOME, 'results', 'activations'))
    parser.add_argument('--outdir', default=os.path.join(HOME, 'project', 'paper', 'figures'))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not HAS_RIPSER:
        print("ERROR: ripser not installed. Run this script in topo_env:")
        print("  source ~/project/topo_env/bin/activate")
        print("  python3 ph_sensitivity.py")
        return

    # Get a representative subset of activation files
    act_files = [f for f in os.listdir(args.act_dir) if f.endswith('.npy') and 'labels' not in f]
    
    if not act_files:
        print(f"No activation files found in {args.act_dir}")
        return

    # Sample up to 20 files for sensitivity analysis
    np.random.seed(42)
    if len(act_files) > 20:
        act_files = list(np.random.choice(act_files, 20, replace=False))
    
    print(f"Testing sensitivity on {len(act_files)} activation files")

    subsample_sizes = [50, 100, 200, 300, 500]
    subsample_seeds = [42, 123, 456]

    results = []
    for fname in act_files:
        fpath = os.path.join(args.act_dir, fname)
        try:
            act = np.load(fpath, allow_pickle=True)
            if act.dtype == object:
                continue
        except:
            continue

        # Parse metadata from filename
        parts = fname.replace('.npy', '').split('_')
        
        for ss in subsample_sizes:
            if ss > len(act):
                continue
            for rs in subsample_seeds:
                tp, b0, b1 = compute_ph(act, ss, rs)
                if tp is not None:
                    results.append({
                        'file': fname, 'subsample_size': ss, 'subsample_seed': rs,
                        'total_persistence': tp, 'b0': b0, 'b1': b1,
                        'n_samples': len(act)
                    })

    if not results:
        print("No results computed")
        return

    rdf = pd.DataFrame(results)
    print(f"Computed {len(rdf)} PH values")

    # Analysis 1: How stable is PH across subsample sizes?
    print("\n=== Subsample Size Sensitivity ===")
    print(f"{'Size':<8} {'Mean TP':<12} {'Std TP':<12} {'CV':<8}")
    print("-" * 40)
    for ss in subsample_sizes:
        subset = rdf[rdf['subsample_size'] == ss]
        if len(subset) == 0:
            continue
        mean_tp = subset['total_persistence'].mean()
        std_tp = subset['total_persistence'].std()
        cv = std_tp / mean_tp if mean_tp > 0 else 0
        print(f"{ss:<8} {mean_tp:<12.2f} {std_tp:<12.2f} {cv:<8.4f}")

    # Analysis 2: Does the ranking of files stay the same across sizes?
    print("\n=== Ranking Stability (Spearman between subsample sizes) ===")
    reference_size = 300
    ref_data = rdf[(rdf['subsample_size'] == reference_size) & (rdf['subsample_seed'] == 42)]
    
    if len(ref_data) > 0:
        for ss in subsample_sizes:
            if ss == reference_size:
                continue
            comp_data = rdf[(rdf['subsample_size'] == ss) & (rdf['subsample_seed'] == 42)]
            merged = ref_data.merge(comp_data, on='file', suffixes=('_ref', '_comp'))
            if len(merged) >= 5:
                rho, p = spearmanr(merged['total_persistence_ref'], merged['total_persistence_comp'])
                print(f"  n={reference_size} vs n={ss}: rho={rho:.4f}, p={p:.2e}")

    # Analysis 3: Seed stability at fixed subsample size
    print(f"\n=== Seed Stability (at subsample={reference_size}) ===")
    ref_ss = rdf[rdf['subsample_size'] == reference_size]
    if len(ref_ss) > 0:
        seed_cv = ref_ss.groupby('file')['total_persistence'].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        print(f"  Mean CV across seeds: {seed_cv.mean():.4f}")
        print(f"  Max CV: {seed_cv.max():.4f}")
        print(f"  Files with CV > 0.1: {(seed_cv > 0.1).sum()}/{len(seed_cv)}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot: TP distribution by subsample size
    ax = axes[0]
    data_by_size = [rdf[rdf['subsample_size'] == ss]['total_persistence'].values for ss in subsample_sizes if len(rdf[rdf['subsample_size'] == ss]) > 0]
    valid_sizes = [ss for ss in subsample_sizes if len(rdf[rdf['subsample_size'] == ss]) > 0]
    ax.boxplot(data_by_size, labels=[str(s) for s in valid_sizes])
    ax.set_xlabel('Subsample Size')
    ax.set_ylabel('Total Persistence')
    ax.set_title('PH Stability Across Subsample Sizes')

    # Scatter: n=300 vs n=100
    ax = axes[1]
    if reference_size in rdf['subsample_size'].values and 100 in rdf['subsample_size'].values:
        ref = rdf[(rdf['subsample_size'] == reference_size) & (rdf['subsample_seed'] == 42)].set_index('file')
        comp = rdf[(rdf['subsample_size'] == 100) & (rdf['subsample_seed'] == 42)].set_index('file')
        common = ref.index.intersection(comp.index)
        if len(common) > 0:
            ax.scatter(ref.loc[common, 'total_persistence'], comp.loc[common, 'total_persistence'], alpha=0.6)
            lims = [0, max(ref.loc[common, 'total_persistence'].max(), comp.loc[common, 'total_persistence'].max())]
            ax.plot(lims, lims, 'k--', linewidth=0.5)
            rho, _ = spearmanr(ref.loc[common, 'total_persistence'], comp.loc[common, 'total_persistence'])
            ax.set_xlabel(f'Total Persistence (n={reference_size})')
            ax.set_ylabel('Total Persistence (n=100)')
            ax.set_title(f'Ranking Stability (ρ={rho:.3f})')

    fig.tight_layout()
    path = os.path.join(args.outdir, 'ph_sensitivity.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: {path}")

    csv_path = os.path.join(args.outdir, 'ph_sensitivity.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print("\nDONE")


if __name__ == '__main__':
    main()
