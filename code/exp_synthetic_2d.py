#!/usr/bin/env python3
"""
Experiment A: Synthetic 2D Mechanistic Experiment
=================================================
Tests whether H1 scales monotonically with the density of label
contradictions at the decision boundary.

Setup:
  - 2 Gaussian clusters in 2D (class 0 centered at [-2,0], class 1 at [+2,0])
  - Flip labels ONLY for points within distance `d` of the boundary (x=0)
  - Flip rates: 0%, 10%, 30%, 50%, 70%, 100%
  - Train small MLP (2-64-64-2), track H1 at penultimate layer
  - 5 seeds each

If H1 scales with flip rate → "folding is caused by local contradictions"
If H1 doesn't scale → folding is something else

Run: cd ~/project && python exp_synthetic_2d.py
No GPU needed — trains in seconds per run.

Output: synthetic_2d_results.csv + printed summary
"""

import numpy as np
import os

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch not available. Run: pip install torch")
    exit(1)

try:
    from ripser import ripser
except ImportError:
    print("Ripser not available. Run: pip install ripser")
    exit(1)

from scipy import stats

np.random.seed(42)
torch.manual_seed(42)

# === CONFIG ===
N_SAMPLES = 1000  # per class
FLIP_RATES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
BOUNDARY_WIDTH = 1.0  # flip labels only for points with |x| < boundary_width
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
HIDDEN = 64
LR = 0.01
SUBSAMPLE = 300  # for PH computation (match paper)
RESULTS = []

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def penultimate(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

def make_data(flip_rate, seed):
    """Create 2D Gaussian clusters with boundary label flips."""
    rng = np.random.RandomState(seed)
    
    # Class 0: centered at (-2, 0)
    X0 = rng.randn(N_SAMPLES, 2) + np.array([-2, 0])
    y0 = np.zeros(N_SAMPLES, dtype=int)
    
    # Class 1: centered at (+2, 0)  
    X1 = rng.randn(N_SAMPLES, 2) + np.array([+2, 0])
    y1 = np.ones(N_SAMPLES, dtype=int)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # Flip labels for points near boundary (|x_0| < boundary_width)
    near_boundary = np.abs(X[:, 0]) < BOUNDARY_WIDTH
    n_near = near_boundary.sum()
    
    if flip_rate > 0 and n_near > 0:
        n_flip = int(flip_rate * n_near)
        flip_indices = rng.choice(np.where(near_boundary)[0], size=n_flip, replace=False)
        y[flip_indices] = 1 - y[flip_indices]
    
    return X, y, n_near

def compute_h1(model, X_tensor, n_sub=SUBSAMPLE):
    """Compute total H1 persistence of penultimate representations."""
    model.eval()
    with torch.no_grad():
        Z = model.penultimate(X_tensor).numpy()
    
    # Subsample
    if len(Z) > n_sub:
        idx = np.random.choice(len(Z), n_sub, replace=False)
        Z = Z[idx]
    
    # Compute PH
    result = ripser(Z, maxdim=1)
    dgm1 = result['dgms'][1]
    
    # Total H1 persistence
    if len(dgm1) == 0:
        return 0.0, 0
    
    # Filter infinite bars and near-zero
    finite = dgm1[np.isfinite(dgm1[:, 1])]
    if len(finite) == 0:
        return 0.0, 0
    
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0.01 * np.max(lifetimes)]  # persistence filter
    
    return float(np.sum(lifetimes)), len(lifetimes)

def run_one(flip_rate, seed):
    """Train one config and return H1 at multiple epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y, n_boundary = make_data(flip_rate, seed)
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    checkpoints = [10, 25, 50, 75, 100]
    results = {}
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch in checkpoints:
            phi, n_loops = compute_h1(model, X_t)
            
            # Also compute train accuracy
            model.eval()
            with torch.no_grad():
                pred = model(X_t).argmax(dim=1)
                acc = (pred == y_t).float().mean().item()
            
            results[epoch] = {
                'phi': phi,
                'n_loops': n_loops,
                'train_acc': acc,
                'loss': loss.item(),
            }
    
    return results, n_boundary

print("=" * 60)
print("EXPERIMENT A: SYNTHETIC 2D — DOES H1 SCALE WITH CONTRADICTIONS?")
print("=" * 60)

all_results = []

for fr in FLIP_RATES:
    phis_final = []
    for seed in SEEDS:
        results, n_boundary = run_one(fr, seed)
        
        for ep, data in results.items():
            all_results.append({
                'flip_rate': fr,
                'seed': seed,
                'epoch': ep,
                'phi': data['phi'],
                'n_loops': data['n_loops'],
                'train_acc': data['train_acc'],
                'loss': data['loss'],
                'n_boundary_points': n_boundary,
            })
        
        phis_final.append(results[100]['phi'])
    
    mean_phi = np.mean(phis_final)
    std_phi = np.std(phis_final)
    print(f"  Flip rate {fr:.0%}: Φ(100) = {mean_phi:.2f} ± {std_phi:.2f} (n_loops={results[100]['n_loops']}, acc={results[100]['train_acc']:.3f})")

# Save
import pandas as pd
df = pd.DataFrame(all_results)
outpath = os.path.expanduser("~/results_phase_transition/synthetic_2d_results.csv")
df.to_csv(outpath, index=False)

# Compute correlation
final = df[df['epoch'] == 100].groupby('flip_rate')['phi'].mean()
r, p = stats.spearmanr(final.index, final.values)

print(f"\n=== KEY RESULT ===")
print(f"Spearman(flip_rate, Φ): ρ = {r:.3f}, p = {p:.2e}")
if r > 0 and p < 0.05:
    print("✓ H1 SCALES WITH CONTRADICTION DENSITY → MECHANISTIC SUPPORT")
    print("  Folding is caused by local label contradictions.")
elif r > 0:
    print("~ Positive trend but not significant. Need more seeds?")
else:
    print("✗ H1 does NOT scale with contradictions. Rethink mechanism.")

# Also check trajectory
print(f"\n=== Φ TRAJECTORIES (mean across seeds) ===")
for fr in FLIP_RATES:
    sub = df[df['flip_rate'] == fr].groupby('epoch')['phi'].mean()
    vals = [f"ep{e}={sub.get(e, 0):.1f}" for e in [10, 50, 100]]
    print(f"  Flip {fr:.0%}: {', '.join(vals)}")

print(f"\nSaved: {outpath}")
print("=" * 60)
print("DONE. Paste full output to Claude.")
print("=" * 60)
