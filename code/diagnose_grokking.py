#!/usr/bin/env python3
"""
Diagnostic: try multiple reasonable definitions of Δϕ for grokking and report
which (if any) reproduces the paper's "10/12 unfold, mean -43.1" claim.

Run on Hydra:
  python diagnose_grokking.py
"""
import os, re, glob
import pandas as pd
import numpy as np

HOME = os.path.expanduser("~")
GROK_DIR = os.path.join(HOME, "project", "results_grokking")
pat = re.compile(r"grokking_(add|mul)_p(\d+)_seed(\d+)\.csv")

# Load all
all_dfs = {}
for csv_path in sorted(glob.glob(os.path.join(GROK_DIR, "grokking_*.csv"))):
    m = pat.match(os.path.basename(csv_path))
    if not m: continue
    config = f'{m.group(1)}/p{m.group(2)}/s{m.group(3)}'
    df = pd.read_csv(csv_path).sort_values('epoch').reset_index(drop=True)
    all_dfs[config] = df

print(f"Loaded {len(all_dfs)} configs\n")

def evaluate_definition(name, fn_compute_delta):
    """fn_compute_delta(df) -> delta or NaN."""
    deltas = {cfg: fn_compute_delta(df) for cfg, df in all_dfs.items()}
    valid = {k: v for k, v in deltas.items() if not (v is None or (isinstance(v, float) and np.isnan(v)))}
    
    n = len(valid)
    if n == 0:
        print(f"{name}: no valid runs")
        return
    vals = np.array(list(valid.values()))
    n_unfold = (vals < -2).sum()
    n_flat = (np.abs(vals) <= 2).sum()
    n_fold = (vals > 2).sum()
    mean_unfold = vals[vals < -2].mean() if n_unfold > 0 else float('nan')
    mean_all = vals.mean()
    
    print(f"{name}")
    print(f"  unfold/total = {n_unfold}/{n}, flat = {n_flat}, fold = {n_fold}")
    print(f"  mean (unfolders): {mean_unfold:.1f}")
    print(f"  mean (all):       {mean_all:.1f}")
    print()

# Definition 1: post-grok minus pre-grok using phi_train (what we did before)
def def1(df):
    pre = (df['train_acc'] >= 0.99) & (df['test_acc'] < 0.5)
    post = (df['train_acc'] >= 0.99) & (df['test_acc'] >= 0.95)
    if not pre.any() or not post.any(): return float('nan')
    pre_phi = df.loc[pre[pre].index[-1], 'phi_train']
    post_phi = df.loc[post[post].index[-1], 'phi_train']
    return post_phi - pre_phi

# Definition 2: same but phi_test
def def2(df):
    pre = (df['train_acc'] >= 0.99) & (df['test_acc'] < 0.5)
    post = (df['train_acc'] >= 0.99) & (df['test_acc'] >= 0.95)
    if not pre.any() or not post.any(): return float('nan')
    pre_phi = df.loc[pre[pre].index[-1], 'phi_test']
    post_phi = df.loc[post[post].index[-1], 'phi_test']
    return post_phi - pre_phi

# Definition 3: peak−final (anywhere)
def def3(df):
    return df['phi_train'].iloc[-1] - df['phi_train'].max()

def def4(df):
    return df['phi_test'].iloc[-1] - df['phi_test'].max()

# Definition 5: post-grok phi_train - PEAK phi_train during pre-grok period
def def5(df):
    pre = (df['train_acc'] >= 0.99) & (df['test_acc'] < 0.5)
    post = (df['train_acc'] >= 0.99) & (df['test_acc'] >= 0.95)
    if not pre.any() or not post.any(): return float('nan')
    pre_phi = df.loc[pre, 'phi_train'].max()  # peak during memorization
    post_phi = df.loc[post[post].index[-1], 'phi_train']  # final settled
    return post_phi - pre_phi

# Definition 6: same, phi_test
def def6(df):
    pre = (df['train_acc'] >= 0.99) & (df['test_acc'] < 0.5)
    post = (df['train_acc'] >= 0.99) & (df['test_acc'] >= 0.95)
    if not pre.any() or not post.any(): return float('nan')
    pre_phi = df.loc[pre, 'phi_test'].max()
    post_phi = df.loc[post[post].index[-1], 'phi_test']
    return post_phi - pre_phi

# Definition 7: average phi during pre-grok minus average phi during post-grok
def def7(df):
    pre = (df['train_acc'] >= 0.99) & (df['test_acc'] < 0.5)
    post = (df['train_acc'] >= 0.99) & (df['test_acc'] >= 0.95)
    if not pre.any() or not post.any(): return float('nan')
    return df.loc[post, 'phi_train'].mean() - df.loc[pre, 'phi_train'].mean()

print("="*70)
print("Trying multiple Δϕ definitions to find which (if any) matches paper")
print("Paper claim: 10/12 unfold, mean unfolders = -43.1, mean all = -35.2")
print("="*70 + "\n")

evaluate_definition("Def1: post-pre, phi_train (LAST in each phase)", def1)
evaluate_definition("Def2: post-pre, phi_test  (LAST in each phase)", def2)
evaluate_definition("Def3: final-peak, phi_train (peak anywhere)", def3)
evaluate_definition("Def4: final-peak, phi_test  (peak anywhere)", def4)
evaluate_definition("Def5: post-PEAK, phi_train (peak during pre-grok)", def5)
evaluate_definition("Def6: post-PEAK, phi_test", def6)
evaluate_definition("Def7: mean post - mean pre, phi_train", def7)
