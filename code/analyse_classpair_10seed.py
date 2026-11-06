#!/usr/bin/env python3
"""Analyse 10-seed class-pair ablation. Definitive replication test."""
import os, glob
import pandas as pd
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.expanduser("~/results_classpair_10seed")

dfs = []
for d in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
    csv = os.path.join(d, "results.csv")
    if os.path.exists(csv):
        dfs.append(pd.read_csv(csv))

if not dfs:
    print("No results found")
    exit()

df = pd.concat(dfs)
print(f"Loaded {len(df)} rows from {len(dfs)} runs\n")

print("=" * 70)
print("CLASS-PAIR ABLATION: 10 SEEDS — DEFINITIVE TEST")
print("=" * 70)

n_seeds = df.groupby('condition')['seed'].nunique()
print(f"Seeds per condition: {dict(n_seeds)}")

for ep in [10, 25, 50]:
    ep_data = df[df['epoch'] == ep]
    if len(ep_data) < 10: continue
    
    print(f"\n--- Epoch {ep} ---")
    print(f"{'Condition':<20} {'Φ mean':>10} {'Φ std':>8} {'n':>4}")
    print("-" * 48)
    
    vals = {}
    for cond in ['most_confusable', 'least_confusable']:
        sub = ep_data[ep_data['condition'] == cond]
        if len(sub) == 0: continue
        phi_mean = sub['phi'].mean()
        phi_std = sub['phi'].std()
        vals[cond] = sub['phi'].values
        print(f"{cond:<20} {phi_mean:>10.2f} {phi_std:>8.2f} {len(sub):>4}")
    
    if 'most_confusable' in vals and 'least_confusable' in vals:
        most = vals['most_confusable']
        least = vals['least_confusable']
        if len(most) >= 5 and len(least) >= 5:
            t_stat, p_val = stats.ttest_ind(most, least)
            ratio = most.mean() / least.mean() if least.mean() > 0 else 0
            pct = (ratio - 1) * 100
            print(f"\n  Most/Least ratio: {ratio:.3f} ({pct:+.1f}%)")
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
            
            if p_val < 0.05 and most.mean() > least.mean():
                print(f"  ✓ REPLICATES: confusable folds more, significant")
            elif p_val < 0.05 and most.mean() < least.mean():
                print(f"  ✗ REVERSED: least-confusable folds more (significant)")
            else:
                print(f"  ~ NOT SIGNIFICANT: no clear effect")

print(f"\n{'='*70}")
print("PASTE THIS TO CLAUDE")
print(f"{'='*70}")
