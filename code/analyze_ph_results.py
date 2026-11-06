#!/usr/bin/env python3
"""
analyze_ph_results.py - Statistical analysis of persistent homology results

Analyzes PH metrics (total persistence, Betti numbers) across:
- Different label shuffle ratios
- Different datasets/architectures
- Different training epochs

Outputs:
- Summary statistics per (dataset, architecture, shuffle_ratio, epoch)
- Correlation between shuffle ratio and PH metrics
- Visualization-ready CSV files
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
import json

def load_ph_results(filepath):
    """Load PH results CSV"""
    df = pd.read_csv(filepath)
    return df

def compute_group_statistics(df, group_cols):
    """Compute mean, std, count for each group"""
    stats_dict = {}
    grouped = df.groupby(group_cols)
    
    for name, group in grouped:
        stats_dict[str(name)] = {
            'total_persistence': {
                'mean': group['total_persistence'].mean(),
                'std': group['total_persistence'].std(),
                'count': len(group)
            },
            'b0': {
                'mean': group['b0'].mean(),
                'std': group['b0'].std(),
                'count': len(group)
            },
            'b1': {
                'mean': group['b1'].mean(),
                'std': group['b1'].std(),
                'count': len(group)
            },
            'accuracy': {
                'mean': group.get('accuracy', pd.Series([np.nan])).mean(),
                'std': group.get('accuracy', pd.Series([np.nan])).std()
            }
        }
    return stats_dict

def test_shuffle_correlation(df, dataset, arch, epoch):
    """Test correlation between shuffle ratio and PH metrics"""
    subset = df[(df['dataset'] == dataset) & 
                (df['architecture'] == arch) & 
                (df['epoch'] == epoch)]
    
    if len(subset) < 6:  # Need enough samples for correlation
        return None
    
    shuffle_ratios = subset['shuffle_frac'].values
    tp = subset['total_persistence'].values
    b0 = subset['b0'].values
    b1 = subset['b1'].values
    
    results = {}
    for metric, values in [('total_persistence', tp), ('b0', b0), ('b1', b1)]:
        if np.std(values) > 0:
            corr, pval = stats.spearmanr(shuffle_ratios, values)
            results[metric] = {
                'spearman_rho': corr,
                'p_value': pval,
                'significant': pval < 0.05
            }
    
    return results

def anova_shuffle_effect(df, dataset, arch, epoch):
    """Test if shuffle ratio has significant effect on PH metrics (ANOVA)"""
    subset = df[(df['dataset'] == dataset) & 
                (df['architecture'] == arch) & 
                (df['epoch'] == epoch)]
    
    if len(subset) < 8:
        return None
    
    # Group by shuffle ratio
    grouped = subset.groupby('shuffle_frac')
    groups = [group['total_persistence'].values for _, group in grouped]
    
    if len(groups) < 2:
        return None
    
    # One-way ANOVA
    f_stat, pval = stats.f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': pval,
        'significant': pval < 0.05,
        'n_groups': len(groups),
        'group_sizes': {k: len(v) for k, v in grouped}
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze PH results')
    parser.add_argument('--input', type=str, default=os.path.expanduser('~/results/ph_results_ALL.csv'),
                        help='Input PH results CSV')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/results'),
                        help='Output directory for analysis results')
    args = parser.parse_args()
    
    print(f"Loading PH results from {args.input}")
    df = load_ph_results(args.input)
    print(f"Loaded {len(df)} PH computations")
    
    # Overall summary
    print("\n=== Dataset Summary ===")
    summary = df.groupby(['dataset', 'architecture']).agg({
        'total_persistence': ['mean', 'std', 'min', 'max'],
        'b0': ['mean', 'std'],
        'b1': ['mean', 'std'],
        'shuffle_frac': 'count'
    }).round(3)
    print(summary)
    
    # Per-epoch analysis
    print("\n=== Per-Epoch Analysis ===")
    for epoch in sorted(df['epoch'].unique()):
        epoch_df = df[df['epoch'] == epoch]
        print(f"\nEpoch {epoch}:")
        
        for (dataset, arch), group in epoch_df.groupby(['dataset', 'architecture']):
            print(f"  {dataset}/{arch}: {len(group)} samples")
            
            # Correlation with shuffle
            corr_results = test_shuffle_correlation(df, dataset, arch, epoch)
            if corr_results:
                for metric, res in corr_results.items():
                    if res:
                        sig = "✓" if res['significant'] else " "
                        print(f"    {sig} {metric}: ρ={res['spearman_rho']:.3f}, p={res['p_value']:.4f}")
            
            # ANOVA test
            anova_results = anova_shuffle_effect(df, dataset, arch, epoch)
            if anova_results:
                sig = "✓" if anova_results['significant'] else " "
                print(f"    {sig} ANOVA: F={anova_results['f_statistic']:.3f}, p={anova_results['p_value']:.4f}")
    
    # Save detailed statistics
    output_file = os.path.join(args.output_dir, 'ph_analysis_summary.json')
    
    analysis_results = {
        'total_samples': len(df),
        'datasets': df['dataset'].unique().tolist(),
        'architectures': df['architecture'].unique().tolist(),
        'epochs': sorted(df['epoch'].unique().tolist()),
        'per_group_stats': {},
        'correlation_tests': [],
        'anova_tests': []
    }
    
    # Compute per-group statistics
    for epoch in sorted(df['epoch'].unique()):
        for (dataset, arch), group in df.groupby(['dataset', 'architecture']):
            epoch_group = group[group['epoch'] == epoch]
            if len(epoch_group) > 0:
                key = f"{dataset}_{arch}_epoch{epoch}"
                analysis_results['per_group_stats'][key] = {
                    'n_samples': len(epoch_group),
                    'total_persistence': {
                        'mean': float(epoch_group['total_persistence'].mean()),
                        'std': float(epoch_group['total_persistence'].std())
                    },
                    'b0': {
                        'mean': float(epoch_group['b0'].mean()),
                        'std': float(epoch_group['b0'].std())
                    },
                    'b1': {
                        'mean': float(epoch_group['b1'].mean()),
                        'std': float(epoch_group['b1'].std())
                    }
                }
    
    # Correlation tests
    for epoch in sorted(df['epoch'].unique()):
        for dataset in df['dataset'].unique():
            for arch in df['architecture'].unique():
                corr = test_shuffle_correlation(df, dataset, arch, epoch)
                if corr:
                    analysis_results['correlation_tests'].append({
                        'dataset': dataset,
                        'architecture': arch,
                        'epoch': epoch,
                        'results': corr
                    })
    
    # ANOVA tests
    for epoch in sorted(df['epoch'].unique()):
        for dataset in df['dataset'].unique():
            for arch in df['architecture'].unique():
                anova = anova_shuffle_effect(df, dataset, arch, epoch)
                if anova:
                    # Convert numpy types to Python native types
                    anova_clean = {
                        'f_statistic': float(anova['f_statistic']),
                        'p_value': float(anova['p_value']),
                        'significant': bool(anova['significant']),
                        'n_groups': int(anova['n_groups']),
                        'group_sizes': {float(k): int(v) for k, v in anova['group_sizes'].items()}
                    }
                    analysis_results['anova_tests'].append({
                        'dataset': dataset,
                        'architecture': arch,
                        'epoch': int(epoch),
                        'results': anova_clean
                    })
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n=== Analysis saved to {output_file} ===")
    
    # Key findings summary
    print("\n=== KEY FINDINGS ===")
    significant_corrs = [t for t in analysis_results['correlation_tests'] 
                         if any(r['significant'] for r in t['results'].values() if r)]
    significant_anovas = [t for t in analysis_results['anova_tests'] 
                          if t['results']['significant']]
    
    print(f"Significant shuffle-PH correlations: {len(significant_corrs)}")
    print(f"Significant ANOVA results: {len(significant_anovas)}")
    
    if significant_corrs:
        print("\nNotable correlations (shuffle ratio → PH metric):")
        for test in significant_corrs[:5]:
            print(f"  {test['dataset']}/{test['architecture']} epoch{test['epoch']}:")
            for metric, res in test['results'].items():
                if res and res['significant']:
                    print(f"    {metric}: ρ={res['spearman_rho']:.3f} (p={res['p_value']:.4f})")

if __name__ == '__main__':
    main()
