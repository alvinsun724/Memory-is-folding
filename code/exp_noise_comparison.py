#!/usr/bin/env python3
"""
Quantitative Noise Estimator Comparison
=========================================
Compares Φ(10) against other population-level noise estimators:
- Train loss at epoch 10
- H0 persistence at epoch 10
- Number of H1 loops at epoch 10
- Activation norm (if available)

All use the SAME epoch-10 checkpoint data. No new training needed.
Reports R² and ρ for each estimator.

Run: python exp_noise_comparison.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PH_CSV = os.path.expanduser("~/results_phase_transition/ph_phase_transition.csv")

def fit_and_evaluate(X, y, name):
    """Fit linear model, return R² and Spearman ρ."""
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    mask = np.isfinite(X.ravel()) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return None
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    rho, p = stats.spearmanr(X.ravel(), y)
    
    # Per-dataset R²
    return {'name': name, 'R2': r2, 'rho': rho, 'p': p, 'n': len(X)}

def main():
    df = pd.read_csv(PH_CSV)
    print(f"Loaded {len(df)} rows")
    
    ep10 = df[df['epoch'] == 10].copy()
    print(f"Epoch 10: {len(ep10)} rows")
    
    noise = ep10['shuffle_frac'].values
    
    print("\n" + "=" * 70)
    print("QUANTITATIVE NOISE ESTIMATOR COMPARISON (EPOCH 10)")
    print("=" * 70)
    
    # All candidate features
    candidates = []
    
    # 1. Φ (total H1 persistence)
    if 'total_persistence' in ep10.columns:
        phi = ep10['total_persistence'].values
        r = fit_and_evaluate(phi, noise, 'Φ (total H1 persistence)')
        if r: candidates.append(r)
    
    # 2. Number of H1 loops
    if 'n_loops' in ep10.columns:
        nloops = ep10['n_loops'].values
        r = fit_and_evaluate(nloops, noise, 'Number of H1 loops')
        if r: candidates.append(r)
    
    # 3. H0 persistence
    if 'h0_persistence' in ep10.columns:
        h0 = ep10['h0_persistence'].values
        r = fit_and_evaluate(h0, noise, 'H0 persistence')
        if r: candidates.append(r)
    
    # 4. Mean persistence (H1)
    if 'mean_persistence' in ep10.columns:
        mp = ep10['mean_persistence'].values
        r = fit_and_evaluate(mp, noise, 'Mean H1 persistence')
        if r: candidates.append(r)
    
    # 5. Max persistence
    if 'max_persistence' in ep10.columns:
        maxp = ep10['max_persistence'].values
        r = fit_and_evaluate(maxp, noise, 'Max H1 persistence')
        if r: candidates.append(r)
    
    # 6. Train loss (if available)
    if 'train_loss' in ep10.columns:
        tl = ep10['train_loss'].values
        r = fit_and_evaluate(tl, noise, 'Train loss')
        if r: candidates.append(r)
    
    # 7. Test accuracy (if available)
    if 'test_acc' in ep10.columns:
        ta = ep10['test_acc'].values
        r = fit_and_evaluate(ta, noise, 'Test accuracy')
        if r: candidates.append(r)
    
    # 8. Train accuracy
    if 'train_acc' in ep10.columns:
        tra = ep10['train_acc'].values
        r = fit_and_evaluate(tra, noise, 'Train accuracy')
        if r: candidates.append(r)
    
    # Print comparison
    print(f"\n{'Method':<30} {'R²':>8} {'ρ':>8} {'p':>12} {'n':>5}")
    print("-" * 70)
    for c in sorted(candidates, key=lambda x: -x['R2']):
        print(f"{c['name']:<30} {c['R2']:>8.3f} {c['rho']:>8.3f} {c['p']:>12.2e} {c['n']:>5}")
    
    # Per-dataset breakdown for top method vs Φ
    print(f"\n{'='*70}")
    print("PER-DATASET R² COMPARISON")
    print(f"{'='*70}")
    
    if 'dataset' in ep10.columns:
        datasets = sorted(ep10['dataset'].unique())
        print(f"\n{'Dataset':<20}", end='')
        for c in sorted(candidates, key=lambda x: -x['R2'])[:4]:
            print(f" {c['name'][:15]:>15}", end='')
        print()
        print("-" * 80)
        
        for ds in datasets:
            ds_data = ep10[ep10['dataset'] == ds]
            if len(ds_data) < 5:
                continue
            noise_ds = ds_data['shuffle_frac'].values
            print(f"{ds:<20}", end='')
            
            for c in sorted(candidates, key=lambda x: -x['R2'])[:4]:
                col_map = {
                    'Φ (total H1 persistence)': 'total_persistence',
                    'Number of H1 loops': 'n_loops',
                    'Train loss': 'train_loss',
                    'Test accuracy': 'test_acc',
                    'Train accuracy': 'train_acc',
                    'H0 persistence': 'h0_persistence',
                    'Mean H1 persistence': 'mean_persistence',
                    'Max H1 persistence': 'max_persistence',
                }
                col = col_map.get(c['name'])
                if col and col in ds_data.columns:
                    vals = ds_data[col].values
                    mask = np.isfinite(vals) & np.isfinite(noise_ds)
                    if mask.sum() >= 5:
                        r2 = r2_score(noise_ds[mask], 
                                     LinearRegression().fit(vals[mask].reshape(-1,1), 
                                                           noise_ds[mask]).predict(vals[mask].reshape(-1,1)))
                        print(f" {r2:>15.3f}", end='')
                    else:
                        print(f" {'---':>15}", end='')
                else:
                    print(f" {'N/A':>15}", end='')
            print()
    
    # Key comparison: Φ vs train loss
    print(f"\n{'='*70}")
    print("KEY COMPARISON: Φ vs TRAIN LOSS AS NOISE ESTIMATOR")
    print(f"{'='*70}")
    
    if 'train_loss' in ep10.columns and 'total_persistence' in ep10.columns:
        phi_vals = ep10['total_persistence'].values
        loss_vals = ep10['train_loss'].values
        
        mask = np.isfinite(phi_vals) & np.isfinite(loss_vals) & np.isfinite(noise)
        phi_m, loss_m, noise_m = phi_vals[mask], loss_vals[mask], noise[mask]
        
        # R² for each
        r2_phi = r2_score(noise_m, LinearRegression().fit(phi_m.reshape(-1,1), noise_m).predict(phi_m.reshape(-1,1)))
        r2_loss = r2_score(noise_m, LinearRegression().fit(loss_m.reshape(-1,1), noise_m).predict(loss_m.reshape(-1,1)))
        
        print(f"  Φ R² = {r2_phi:.3f}")
        print(f"  Train loss R² = {r2_loss:.3f}")
        print(f"  Advantage: {'Φ' if r2_phi > r2_loss else 'Train loss'} (+{abs(r2_phi - r2_loss):.3f})")
        
        # BUT: train loss requires labels, Φ does not
        print(f"\n  Critical difference: Train loss requires labels to compute.")
        print(f"  Φ requires only unlabelled activations.")
    
    print(f"\n{'='*70}")
    print("PASTE THIS TO CLAUDE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
