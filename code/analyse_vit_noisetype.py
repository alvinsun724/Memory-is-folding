#!/usr/bin/env python3
"""Analyse ViT noise-type results. Run: python analyse_vit_noisetype.py"""
import os, glob
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.expanduser("~/results_vit_noisetype")

dfs = []
for d in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
    csv = os.path.join(d, "results.csv")
    if os.path.exists(csv):
        dfs.append(pd.read_csv(csv))

if not dfs:
    print("No results found"); exit()

df = pd.concat(dfs)
print(f"Loaded {len(df)} rows from {len(dfs)} runs\n")

# Need to also load clean baseline from existing ViT results
# For clean reference, we use the Φ value at matching epoch from clean ViT runs

print("=" * 70)
print("ViT NOISE-TYPE TEST: ASYMMETRIC vs INSTANCE-DEPENDENT")
print("=" * 70)

for ep in [10, 25, 50]:
    ep_data = df[df['epoch'] == ep]
    if len(ep_data) == 0: continue
    
    print(f"\n--- Epoch {ep} ---")
    print(f"{'Type':<12} {'Rate':>6} {'Per-class Φ':>14} {'Global Φ':>12} {'Test Acc':>10}")
    print("-" * 58)
    for nt in ['asymmetric', 'instance']:
        for rate in sorted(ep_data['rate'].unique()):
            sub = ep_data[(ep_data['noise_type'] == nt) & (ep_data['rate'] == rate)]
            if len(sub) == 0: continue
            pc = sub['perclass_phi'].mean()
            g = sub['global_phi'].mean()
            ta = sub['test_acc'].mean()
            pc_std = sub['perclass_phi'].std()
            print(f"{nt:<12} {rate:>6.2f} {pc:>10.1f}±{pc_std:>3.1f} {g:>12.1f} {ta:>9.1f}%")

# Key finding: does asymmetric fold less than instance-dep?
print(f"\n{'='*70}")
print("KEY FINDING: ASYMMETRIC vs INSTANCE-DEPENDENT")
print(f"{'='*70}")
print("If folding is noise-type-specific, asymmetric should fold LESS than instance-dep")
print("at matched noise rates.\n")

for ep in [10, 25, 50]:
    ep_data = df[df['epoch'] == ep]
    if len(ep_data) == 0: continue
    
    print(f"--- Epoch {ep} ---")
    for rate in sorted(ep_data['rate'].unique()):
        asym = ep_data[(ep_data['noise_type'] == 'asymmetric') & (ep_data['rate'] == rate)]['perclass_phi'].mean()
        inst = ep_data[(ep_data['noise_type'] == 'instance') & (ep_data['rate'] == rate)]['perclass_phi'].mean()
        if not np.isnan(asym) and not np.isnan(inst):
            ratio = inst / asym if asym > 0 else float('inf')
            print(f"  rate={rate}: asymmetric Φ={asym:.1f}, instance-dep Φ={inst:.1f}, ratio={ratio:.2f}x")
            if inst > asym * 1.2:
                print(f"    → Instance-dep folds MORE than asymmetric ✓ (folding thesis holds)")
            elif asym > inst * 1.2:
                print(f"    → Asymmetric folds more (unexpected)")
            else:
                print(f"    → Similar folding (thesis does not clearly replicate on ViT)")

print(f"\n{'='*70}")
print("PASTE THIS TO CLAUDE")
print(f"{'='*70}")
