#!/usr/bin/env python3
"""Analyse class-pair ablation with 5 seeds. Run: python analyse_classpair.py"""
import os, glob
import pandas as pd
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.expanduser("~/results_classpair")

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
print("CLASS-PAIR ABLATION: CONFUSABLE vs NON-CONFUSABLE (5 seeds)")
print("=" * 70)

seeds_per_cond = df.groupby('condition')['seed'].nunique()
print(f"\nSeeds per condition: {dict(seeds_per_cond)}")

for ep in [10, 25, 50]:
    ep_data = df[df['epoch'] == ep]
    if len(ep_data) < 4:
        continue
    
    print(f"\n--- Epoch {ep} ---")
    print(f"{'Condition':<16} {'Φ mean':>10} {'Φ std':>8} {'Val Acc':>10} {'n':>4}")
    print("-" * 50)
    
    cond_phis = {}
    for cond in ['confusable', 'non_confusable']:
        sub = ep_data[ep_data['condition'] == cond]
        if len(sub) == 0: continue
        phi_mean = sub['phi'].mean()
        phi_std = sub['phi'].std()
        va = sub['val_acc'].mean()
        cond_phis[cond] = sub['phi'].values
        print(f"{cond:<16} {phi_mean:>10.2f} {phi_std:>8.2f} {va:>9.1f}% {len(sub):>4}")
    
    if 'confusable' in cond_phis and 'non_confusable' in cond_phis:
        conf = cond_phis['confusable']
        non = cond_phis['non_confusable']
        if len(conf) >= 3 and len(non) >= 3:
            t_stat, p_val = stats.ttest_ind(conf, non)
            ratio = conf.mean() / non.mean() if non.mean() > 0 else 0
            pct = (ratio - 1) * 100
            print(f"  ratio: {ratio:.3f} ({pct:+.1f}% more folding in confusable)")
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"  → Statistically significant")

# Per-seed breakdown
print(f"\n{'='*70}")
print("PER-SEED VALUES AT EPOCH 10")
print(f"{'='*70}")
ep10 = df[df['epoch'] == 10]
for cond in ['confusable', 'non_confusable']:
    sub = ep10[ep10['condition'] == cond].sort_values('seed')
    print(f"\n  {cond}:")
    for _, r in sub.iterrows():
        print(f"    seed {int(r['seed'])}: Φ={r['phi']:.1f}")

print(f"\n{'='*70}")
print("PASTE THIS TO CLAUDE")
print(f"{'='*70}")
