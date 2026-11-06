#!/usr/bin/env python3
"""
Experiment 4: Partial Correlation / Conditional Independence Test
=================================================================
Tests whether Phi carries information about noise level BEYOND what
activation variance already provides (and vice versa).

Method:
  1. Regress noise_level ~ activation_variance → residuals_A
  2. Correlate residuals_A with Phi → if significant, Phi adds info
  3. Regress noise_level ~ Phi → residuals_B  
  4. Correlate residuals_B with activation_variance → if significant, variance adds info
  5. Also compute partial correlation directly

If BOTH are significant → complementary channels (strong claim).
If only one → one subsumes the other.

Run on Hydra login node (CPU only):
  cd ~/project && python exp4_partial_correlation.py

Outputs: partial_correlation_results.csv, printed summary
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# === CONFIG ===
BASELINE_CSV = os.path.expanduser("~/results_phase_transition/baseline_comparison.csv")  
INDEPENDENCE_CSV = os.path.expanduser("~/results_phase_transition/independence_test.csv")
PH_CSV = os.path.expanduser("~/results_phase_transition/ph_phase_transition.csv")

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z.
    Uses residual method: regress x~z and y~z, correlate residuals."""
    # Regress x on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    
    # Regress y on z
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    
    # Correlate residuals
    r, p = stats.spearmanr(resid_x, resid_y)
    return r, p

def try_load(path):
    """Try loading CSV, print diagnostic if missing."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {path}: {len(df)} rows, columns: {list(df.columns)}")
        return df
    else:
        print(f"NOT FOUND: {path}")
        # List what IS there
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            files = os.listdir(parent)
            print(f"  Available in {parent}:")
            for f in sorted(files)[:20]:
                print(f"    {f}")
        return None

def main():
    print("=" * 60)
    print("EXPERIMENT 4: PARTIAL CORRELATION TEST")
    print("=" * 60)
    
    # Try loading baseline comparison (has Phi, act_var, noise level per config)
    baseline = try_load(BASELINE_CSV)
    independence = try_load(INDEPENDENCE_CSV)
    ph_data = try_load(PH_CSV)
    
    # We need a dataframe with columns: noise_level, phi, activation_variance
    # Try to construct this from available data
    
    if baseline is not None:
        print(f"\n--- Baseline CSV columns: {list(baseline.columns)} ---")
        print(baseline.head())
        
        # Identify the right column names (may vary)
        noise_col = None
        phi_col = None
        var_col = None
        norm_col = None
        
        for c in baseline.columns:
            cl = c.lower()
            if 'noise' in cl or 'shuffle' in cl or 'sf' in cl:
                noise_col = c
            if 'phi' in cl or 'total_persist' in cl or 'h1_total' in cl:
                phi_col = c
            if 'var' in cl or 'act_var' in cl:
                var_col = c
            if 'norm' in cl or 'act_norm' in cl:
                norm_col = c
        
        print(f"\nDetected columns: noise={noise_col}, phi={phi_col}, var={var_col}, norm={norm_col}")
        
        if noise_col and phi_col and (var_col or norm_col):
            stat_col = var_col or norm_col
            
            df = baseline[[noise_col, phi_col, stat_col]].dropna()
            noise = df[noise_col].values.astype(float)
            phi = df[phi_col].values.astype(float)
            stat = df[stat_col].values.astype(float)
            
            print(f"\nWorking with {len(df)} data points")
            
            # Test 1: Direct correlations
            r_phi_noise, p_phi_noise = stats.spearmanr(phi, noise)
            r_stat_noise, p_stat_noise = stats.spearmanr(stat, noise)
            r_phi_stat, p_phi_stat = stats.spearmanr(phi, stat)
            
            print(f"\n=== DIRECT CORRELATIONS ===")
            print(f"Phi vs noise:       rho={r_phi_noise:.3f}, p={p_phi_noise:.2e}")
            print(f"Stat vs noise:      rho={r_stat_noise:.3f}, p={p_stat_noise:.2e}")
            print(f"Phi vs Stat:        rho={r_phi_stat:.3f}, p={p_phi_stat:.2e}")
            
            # Test 2: Partial correlations
            r_partial_phi, p_partial_phi = partial_corr(phi, noise, stat)
            r_partial_stat, p_partial_stat = partial_corr(stat, noise, phi)
            
            print(f"\n=== PARTIAL CORRELATIONS ===")
            print(f"Phi vs noise | Stat:   r={r_partial_phi:.3f}, p={p_partial_phi:.2e}")
            print(f"  → {'SIGNIFICANT' if p_partial_phi < 0.05 else 'not significant'}: Phi {'DOES' if p_partial_phi < 0.05 else 'does NOT'} carry info beyond Stat")
            print(f"Stat vs noise | Phi:   r={r_partial_stat:.3f}, p={p_partial_stat:.2e}")
            print(f"  → {'SIGNIFICANT' if p_partial_stat < 0.05 else 'not significant'}: Stat {'DOES' if p_partial_stat < 0.05 else 'does NOT'} carry info beyond Phi")
            
            # Test 3: Residual analysis
            slope, intercept, _, _, _ = stats.linregress(stat, noise)
            resid_from_stat = noise - (slope * stat + intercept)
            r_resid, p_resid = stats.spearmanr(resid_from_stat, phi)
            
            slope2, intercept2, _, _, _ = stats.linregress(phi, noise)
            resid_from_phi = noise - (slope2 * phi + intercept2)
            r_resid2, p_resid2 = stats.spearmanr(resid_from_phi, stat)
            
            print(f"\n=== RESIDUAL ANALYSIS ===")
            print(f"Residuals(noise~Stat) vs Phi: rho={r_resid:.3f}, p={p_resid:.2e}")
            print(f"Residuals(noise~Phi) vs Stat: rho={r_resid2:.3f}, p={p_resid2:.2e}")
            
            # Interpretation
            print(f"\n=== INTERPRETATION ===")
            if p_partial_phi < 0.05 and p_partial_stat < 0.05:
                print("BOTH channels carry unique information → COMPLEMENTARY (supports paper claim)")
            elif p_partial_phi < 0.05:
                print("Only Phi carries unique info → Phi adds to Stat but not vice versa")
            elif p_partial_stat < 0.05:
                print("Only Stat carries unique info → Stat adds to Phi but not vice versa")
            else:
                print("Neither carries unique info → may be redundant (weakens paper claim)")
            
            # Save results
            results = pd.DataFrame([{
                'test': 'partial_corr_phi_given_stat',
                'r': r_partial_phi, 'p': p_partial_phi,
                'significant': p_partial_phi < 0.05,
            }, {
                'test': 'partial_corr_stat_given_phi', 
                'r': r_partial_stat, 'p': p_partial_stat,
                'significant': p_partial_stat < 0.05,
            }, {
                'test': 'residual_phi_on_stat_residuals',
                'r': r_resid, 'p': p_resid,
                'significant': p_resid < 0.05,
            }, {
                'test': 'residual_stat_on_phi_residuals',
                'r': r_resid2, 'p': p_resid2,
                'significant': p_resid2 < 0.05,
            }])
            
            outpath = os.path.expanduser("~/results_phase_transition/partial_correlation_results.csv")
            results.to_csv(outpath, index=False)
            print(f"\nSaved: {outpath}")
        else:
            print("\nCannot identify required columns. Please check CSV structure.")
            print("Need: noise_level, phi/total_persistence, activation_variance/norm")
    
    # Also try per-epoch analysis using independence_test.csv
    if independence is not None:
        print(f"\n\n{'='*60}")
        print("PER-EPOCH ANALYSIS (independence_test.csv)")
        print(f"{'='*60}")
        print(f"Columns: {list(independence.columns)}")
        print(independence.head())
        # This data may have epoch-level Phi and stat correlations
        # Adapt column names as needed

if __name__ == "__main__":
    main()
