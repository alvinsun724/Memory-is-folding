#!/usr/bin/env python3
"""
Class-Pair Targeted Ablation
=============================
Inject 50% label noise ONLY into the 5 most-confusable vs 5 least-confusable
class pairs on CIFAR-10/ResNet-18. Compare Φ trajectories.

If most-confusable pairs → higher Φ than least-confusable → causal support
that semantic proximity drives folding.

Uses existing train_and_extract.py pipeline structure.
2 GPU jobs × ~3 hours each.

Run: python exp_classpair_ablation.py --condition most --seed 42
     python exp_classpair_ablation.py --condition least --seed 42
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from ripser import ripser
except:
    ripser = None

# From paper's verified data: most-folded pairs by H1 ratio
MOST_CONFUSABLE = [(7, 9), (1, 9), (1, 5), (6, 8), (3, 5)]  # horse/truck, auto/truck, etc.
# Least confusable: pairs with largest centroid distance and lowest H1 ratio
# Using CIFAR-10 class structure, furthest apart semantically
LEAST_CONFUSABLE = [(0, 1), (0, 8), (2, 8), (4, 6), (2, 9)]  # airplane/auto, airplane/ship, etc.

EPOCHS = 50
BATCH_SIZE = 128
LR = 0.1
CHECKPOINTS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
FLIP_RATE = 0.5  # 50% noise within selected pairs

def targeted_noise(targets, pair_list, flip_rate, seed):
    """Flip labels only between classes in the specified pairs."""
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    
    flipped = 0
    for c1, c2 in pair_list:
        # Find samples of class c1 and c2
        idx_c1 = np.where(targets == c1)[0]
        idx_c2 = np.where(targets == c2)[0]
        
        # Flip c1 → c2
        n_flip_1 = int(flip_rate * len(idx_c1))
        flip_1 = rng.choice(idx_c1, n_flip_1, replace=False)
        targets[flip_1] = c2
        
        # Flip c2 → c1
        n_flip_2 = int(flip_rate * len(idx_c2))
        flip_2 = rng.choice(idx_c2, n_flip_2, replace=False)
        targets[flip_2] = c1
        
        flipped += n_flip_1 + n_flip_2
    
    return targets.tolist(), flipped

def compute_phi(Z, n_sub=300):
    if ripser is None: return 0.0, 0
    if len(Z) > n_sub:
        idx = np.random.choice(len(Z), n_sub, replace=False)
        Z = Z[idx]
    result = ripser(Z, maxdim=1)
    dgm1 = result['dgms'][1]
    if len(dgm1) == 0: return 0.0, 0
    fin = dgm1[np.isfinite(dgm1[:, 1])]
    if len(fin) == 0: return 0.0, 0
    lt = fin[:, 1] - fin[:, 0]
    lt = lt[lt > 0.01 * np.max(lt)]
    return float(np.sum(lt)), len(lt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['most', 'least', 'clean'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.expanduser('~/results_classpair'))
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True,
                                           transform=transform_test)
    
    # Apply targeted noise
    if args.condition == 'most':
        pairs = MOST_CONFUSABLE
        trainset.targets, n_flipped = targeted_noise(trainset.targets, pairs, FLIP_RATE, args.seed)
    elif args.condition == 'least':
        pairs = LEAST_CONFUSABLE
        trainset.targets, n_flipped = targeted_noise(trainset.targets, pairs, FLIP_RATE, args.seed)
    else:
        n_flipped = 0
    
    print(f"Condition: {args.condition}, Seed: {args.seed}, Flipped: {n_flipped}/{len(trainset)}")
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # ResNet-18
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    run_dir = os.path.join(args.output_dir, f"{args.condition}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    results = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        scheduler.step()
        
        if epoch in CHECKPOINTS:
            # Extract penultimate activations
            model.eval()
            acts_list = []
            hook_out = []
            handle = model.avgpool.register_forward_hook(
                lambda m, i, o: hook_out.append(o.detach().cpu()))
            
            with torch.no_grad():
                n = 0
                for X, y in train_loader:
                    X = X.to(device)
                    _ = model(X)
                    acts_list.append(hook_out[-1].squeeze().numpy())
                    hook_out.clear()
                    n += len(X)
                    if n >= 10000: break
            handle.remove()
            
            acts = np.concatenate(acts_list)[:10000]
            phi, n_loops = compute_phi(acts)
            
            # Test accuracy
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    correct += (model(X).argmax(1) == y).sum().item()
                    total += len(y)
            
            results.append({
                'epoch': epoch, 'phi': phi, 'n_loops': n_loops,
                'test_acc': correct/total*100, 'condition': args.condition
            })
            print(f"  Epoch {epoch}: Φ={phi:.1f}, acc={correct/total*100:.1f}%")
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    print(f"\nSaved: {run_dir}/results.csv")

if __name__ == "__main__":
    main()
