#!/usr/bin/env python3
"""
Synthetic 2D RERUN — 20 seeds, finer flip rates.
Run: cd ~/project && python exp_synthetic_2d_v2.py
No GPU. ~15 min.
"""
import numpy as np, os, torch, torch.nn as nn, torch.optim as optim
from ripser import ripser
from scipy import stats
import pandas as pd

SEEDS = list(range(20))
FLIP_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
N_SAMPLES, EPOCHS, HIDDEN, LR, BOUNDARY_WIDTH = 1000, 100, 64, 0.01, 1.0

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    def penultimate(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

def make_data(flip_rate, seed):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(N_SAMPLES, 2) + [-2, 0]
    X1 = rng.randn(N_SAMPLES, 2) + [+2, 0]
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_SAMPLES), np.ones(N_SAMPLES)]).astype(int)
    near = np.abs(X[:, 0]) < BOUNDARY_WIDTH
    if flip_rate > 0 and near.sum() > 0:
        n_flip = int(flip_rate * near.sum())
        idx = rng.choice(np.where(near)[0], n_flip, replace=False)
        y[idx] = 1 - y[idx]
    return torch.FloatTensor(X), torch.LongTensor(y)

def compute_phi(model, X_t):
    model.eval()
    with torch.no_grad():
        Z = model.penultimate(X_t).numpy()
    idx = np.random.choice(len(Z), 300, replace=False)
    dgm1 = ripser(Z[idx], maxdim=1)['dgms'][1]
    if len(dgm1) == 0: return 0.0
    fin = dgm1[np.isfinite(dgm1[:, 1])]
    if len(fin) == 0: return 0.0
    lt = fin[:, 1] - fin[:, 0]
    return float(np.sum(lt[lt > 0.01 * np.max(lt)]))

print("=" * 60)
print("SYNTHETIC 2D v2: 20 seeds, 9 flip rates")
print("=" * 60)

rows = []
for fr in FLIP_RATES:
    phis_10, phis_50, phis_100 = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        X_t, y_t = make_data(fr, seed)
        model = MLP()
        opt = optim.Adam(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()
        for ep in range(1, EPOCHS+1):
            model.train(); opt.zero_grad(); crit(model(X_t), y_t).backward(); opt.step()
            if ep in [10, 50, 100]:
                phi = compute_phi(model, X_t)
                rows.append({'flip_rate': fr, 'seed': seed, 'epoch': ep, 'phi': phi})
                if ep == 10: phis_10.append(phi)
                elif ep == 50: phis_50.append(phi)
                else: phis_100.append(phi)
    print(f"  Flip {fr:.0%}: ep10={np.mean(phis_10):.2f}±{np.std(phis_10):.2f}, ep50={np.mean(phis_50):.2f}±{np.std(phis_50):.2f}, ep100={np.mean(phis_100):.2f}±{np.std(phis_100):.2f}")

df = pd.DataFrame(rows)

# Key test: at epoch 10, does Phi scale with flip rate (up to 50%)?
ep10 = df[(df['epoch']==10) & (df['flip_rate']<=0.5)]
means_10 = ep10.groupby('flip_rate')['phi'].mean()
r10, p10 = stats.spearmanr(means_10.index, means_10.values)
print(f"\n=== EPOCH 10 (0-50% only): rho={r10:.3f}, p={p10:.4f} {'✓ SIG' if p10<0.05 else ''} ===")

# Also test full range at epoch 10
ep10_all = df[df['epoch']==10]
means_10_all = ep10_all.groupby('flip_rate')['phi'].mean()
r10a, p10a = stats.spearmanr(means_10_all.index, means_10_all.values)
print(f"=== EPOCH 10 (all rates): rho={r10a:.3f}, p={p10a:.4f} ===")

# Epoch 100 (fading)
ep100 = df[df['epoch']==100]
means_100 = ep100.groupby('flip_rate')['phi'].mean()
r100, p100 = stats.spearmanr(means_100.index, means_100.values)
print(f"=== EPOCH 100 (all rates): rho={r100:.3f}, p={p100:.4f} ===")

# Per-rate means for paper
print(f"\n=== VALUES FOR PAPER (mean ± std, n=20) ===")
for ep in [10, 50, 100]:
    vals = []
    for fr in FLIP_RATES:
        sub = df[(df['epoch']==ep) & (df['flip_rate']==fr)]
        vals.append(f"{fr:.0%}:{sub['phi'].mean():.1f}±{sub['phi'].std():.1f}")
    print(f"  Epoch {ep}: {', '.join(vals)}")

outpath = os.path.expanduser("~/results_phase_transition/synthetic_2d_v2_results.csv")
df.to_csv(outpath, index=False)
print(f"\nSaved: {outpath}")
