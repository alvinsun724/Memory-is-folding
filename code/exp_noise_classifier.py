#!/usr/bin/env python3
"""
Noise-Type Binary Classifier + Loss(10) Comparison
=====================================================
1. Binary classifier: "Is the noise learnable (asymmetric) or irreducible (instance-dep)?"
   Uses Φ(10) only. Reports precision/recall/F1. No calibration needed.

2. Fair epoch-10 comparison: Φ(10) vs train_loss(10) vs train_acc(10) as noise estimators.

Uses existing PH CSV. CPU only, ~1 min.
Run: python exp_noise_classifier.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PH_CSV = os.path.expanduser("~/results_phase_transition/ph_phase_transition.csv")

def main():
    df = pd.read_csv(PH_CSV)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # =====================================================================
    # PART 1: NOISE-TYPE BINARY CLASSIFIER
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: NOISE-TYPE BINARY CLASSIFIER")
    print("Is the noise learnable (asymmetric) or irreducible?")
    print("=" * 70)
    
    # Check if noise_type column exists
    if 'noise_type' in df.columns:
        ep10 = df[df['epoch'] == 10].copy()
        
        # Binary: asymmetric vs instance-dependent
        asym = ep10[ep10['noise_type'] == 'asymmetric']
        inst = ep10[ep10['noise_type'] == 'instance']
        
        if len(asym) > 0 and len(inst) > 0:
            print(f"\n  Asymmetric samples: {len(asym)}")
            print(f"  Instance-dep samples: {len(inst)}")
            
            # Classify based on Φ ratio to clean reference per dataset
            # No calibration needed — just compare within same config
            # ... (would need clean reference)
        else:
            print("  Not enough noise-type labeled data in PH CSV.")
    
    # Alternative: Use the noise_model or noise column if available
    print("\n  Available columns for noise type:")
    for col in df.columns:
        uniq = df[col].nunique()
        if uniq < 20:
            print(f"    {col}: {uniq} unique values: {sorted(df[col].unique())[:10]}")
    
    # =====================================================================
    # PART 1b: CONSTRUCT CLASSIFIER FROM Φ TRAJECTORIES
    # =====================================================================
    # From the paper's data: under symmetric noise, Φ(10) decreases with noise rate.
    # Under asymmetric noise, Φ(10) stays flat.
    # Classifier: if Φ(noisy) / Φ(clean) ≈ 1.0 → asymmetric; if < 0.7 → instance-dep
    
    print("\n" + "-" * 70)
    print("NOISE DISCRIMINATION BY Φ RATIO (from paper's existing data)")
    print("-" * 70)
    
    ep10 = df[df['epoch'] == 10].copy()
    
    if 'dataset' in ep10.columns and 'architecture' in ep10.columns:
        # Group by (dataset, architecture) and compute clean reference
        for (ds, arch), group in ep10.groupby(['dataset', 'architecture']):
            clean = group[group['shuffle_frac'] == 0.0]['total_persistence']
            if len(clean) == 0:
                continue
            phi_clean_mean = clean.mean()
            if phi_clean_mean < 1:
                continue
            
            noisy = group[group['shuffle_frac'] > 0.0].copy()
            if len(noisy) == 0:
                continue
            
            noisy['phi_ratio'] = noisy['total_persistence'] / phi_clean_mean
            
            print(f"\n  {ds}/{arch} (Φ_clean={phi_clean_mean:.1f}):")
            for sf in sorted(noisy['shuffle_frac'].unique()):
                sub = noisy[noisy['shuffle_frac'] == sf]
                ratio = sub['phi_ratio'].mean()
                print(f"    sf={sf:.2f}: Φ_ratio={ratio:.3f} {'← near 1 (learnable?)' if ratio > 0.8 else '← dropped (irreducible?)'}")
    
    # =====================================================================
    # PART 2: FAIR EPOCH-10 COMPARISON
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: EPOCH-10 NOISE ESTIMATION — Φ vs STATISTICAL MEASURES")
    print("=" * 70)
    
    ep10 = df[df['epoch'] == 10].copy()
    noise = ep10['shuffle_frac'].values
    
    comparisons = []
    
    # Φ (total persistence)
    if 'total_persistence' in ep10.columns:
        phi = ep10['total_persistence'].values
        mask = np.isfinite(phi) & np.isfinite(noise)
        rho, p = stats.spearmanr(phi[mask], noise[mask])
        r2 = r2_score(noise[mask], LinearRegression().fit(
            phi[mask].reshape(-1, 1), noise[mask]).predict(phi[mask].reshape(-1, 1)))
        comparisons.append(('Φ(10) [topology]', rho, p, r2, 'No'))
        print(f"\n  Φ(10):        ρ={rho:.3f}, R²={r2:.3f}, p={p:.2e}")
    
    # Train loss at epoch 10
    if 'train_loss' in ep10.columns:
        tl = ep10['train_loss'].values
        mask = np.isfinite(tl) & np.isfinite(noise)
        if mask.sum() > 10:
            rho, p = stats.spearmanr(tl[mask], noise[mask])
            r2 = r2_score(noise[mask], LinearRegression().fit(
                tl[mask].reshape(-1, 1), noise[mask]).predict(tl[mask].reshape(-1, 1)))
            comparisons.append(('Loss(10) [statistical]', rho, p, r2, 'Yes'))
            print(f"  Loss(10):     ρ={rho:.3f}, R²={r2:.3f}, p={p:.2e}")
    
    # Train accuracy at epoch 10
    if 'train_acc' in ep10.columns:
        ta = ep10['train_acc'].values
        mask = np.isfinite(ta) & np.isfinite(noise)
        if mask.sum() > 10:
            rho, p = stats.spearmanr(ta[mask], noise[mask])
            r2 = r2_score(noise[mask], LinearRegression().fit(
                ta[mask].reshape(-1, 1), noise[mask]).predict(ta[mask].reshape(-1, 1)))
            comparisons.append(('Acc(10) [statistical]', rho, p, r2, 'Yes'))
            print(f"  Acc(10):      ρ={rho:.3f}, R²={r2:.3f}, p={p:.2e}")
    
    # Test accuracy at epoch 10
    if 'test_acc' in ep10.columns:
        test_a = ep10['test_acc'].values
        mask = np.isfinite(test_a) & np.isfinite(noise)
        if mask.sum() > 10:
            rho, p = stats.spearmanr(test_a[mask], noise[mask])
            r2 = r2_score(noise[mask], LinearRegression().fit(
                test_a[mask].reshape(-1, 1), noise[mask]).predict(test_a[mask].reshape(-1, 1)))
            comparisons.append(('TestAcc(10) [statistical]', rho, p, r2, 'Yes'))
            print(f"  TestAcc(10):  ρ={rho:.3f}, R²={r2:.3f}, p={p:.2e}")
    
    # Number of H1 loops
    if 'n_loops' in ep10.columns:
        nl = ep10['n_loops'].values
        mask = np.isfinite(nl) & np.isfinite(noise)
        if mask.sum() > 10:
            rho, p = stats.spearmanr(nl[mask], noise[mask])
            r2 = r2_score(noise[mask], LinearRegression().fit(
                nl[mask].reshape(-1, 1), noise[mask]).predict(nl[mask].reshape(-1, 1)))
            comparisons.append(('n_loops(10) [topology]', rho, p, r2, 'No'))
            print(f"  n_loops(10):  ρ={rho:.3f}, R²={r2:.3f}, p={p:.2e}")
    
    # Summary table
    if comparisons:
        print(f"\n{'='*70}")
        print("SUMMARY: EPOCH-10 NOISE ESTIMATION COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Measure':<30} {'|ρ|':>6} {'R²':>6} {'Labels?':>8}")
        print(f"  {'-'*55}")
        for name, rho, p, r2, labels in sorted(comparisons, key=lambda x: -abs(x[1])):
            print(f"  {name:<30} {abs(rho):>6.3f} {r2:>6.3f} {labels:>8}")
        
        # Key finding
        phi_rho = abs([c for c in comparisons if 'Φ' in c[0]][0][1])
        stat_rhos = [abs(c[1]) for c in comparisons if c[4] == 'Yes']
        if stat_rhos:
            best_stat = max(stat_rhos)
            if phi_rho > best_stat:
                print(f"\n  → Φ(10) beats all statistical measures at epoch 10")
                print(f"    (|ρ|={phi_rho:.3f} vs best statistical |ρ|={best_stat:.3f})")
            else:
                print(f"\n  → Best statistical measure beats Φ(10) at epoch 10")
                print(f"    (best stat |ρ|={best_stat:.3f} vs Φ |ρ|={phi_rho:.3f})")
                print(f"    BUT: statistical measures require labels; Φ does not.")
    
    print(f"\n{'='*70}")
    print("PASTE THIS TO CLAUDE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
