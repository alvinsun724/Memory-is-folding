#!/usr/bin/env python3
"""
ViT-Tiny Noise-Type Test (Per-Class PH)
===========================================
Tests whether asymmetric Patrini noise produces NO folding while
instance-dependent noise DOES produce folding, now on ViT-Tiny.

Uses per-class PH which we showed recovers 8x more signal on ViT.

Usage: python run_vit_noisetype.py --noise_type asymmetric --rate 0.2 --seed 42
"""
import argparse, os, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from ripser import ripser

NUM_CLASSES = 10
EPOCHS = 50
BS = 128
LR = 3e-4  # AdamW for ViT
CHECKPOINTS = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

# =====================================================================
# Simple ViT for CIFAR-10 (the version that works from memory)
# =====================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class ViTTiny(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        if return_features:
            return x, self.head(x)
        return self.head(x)

# =====================================================================
# Noise application
# =====================================================================
def apply_noise(targets, noise_type, rate, seed):
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    original = targets.copy()
    
    if noise_type == 'asymmetric':
        for i in range(len(targets)):
            if rng.random() < rate:
                targets[i] = (targets[i] + 1) % NUM_CLASSES
    elif noise_type == 'instance':
        raw = torchvision.datasets.CIFAR10('~/data', train=True, download=False)
        flat = np.array(raw.data).reshape(len(raw.data), -1).astype(np.float32)
        centroids = np.zeros((NUM_CLASSES, flat.shape[1]))
        for c in range(NUM_CLASSES):
            centroids[c] = flat[original == c].mean(0)
        dists = np.array([np.linalg.norm(flat[i] - centroids[original[i]]) for i in range(len(flat))])
        dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-10)
        flip_probs = 2 * rate * dists
        for i in range(len(targets)):
            if rng.random() < flip_probs[i]:
                targets[i] = rng.randint(0, NUM_CLASSES)
    
    return targets.tolist(), float(np.mean(targets != original))

# =====================================================================
# PH
# =====================================================================
def compute_phi(Z, n_sub=300):
    if len(Z) > n_sub:
        idx = np.random.choice(len(Z), n_sub, replace=False)
        Z = Z[idx]
    if len(Z) < 20: return 0.0
    dgm = ripser(Z, maxdim=1)['dgms'][1]
    if len(dgm) == 0: return 0.0
    fin = dgm[np.isfinite(dgm[:, 1])]
    if len(fin) == 0: return 0.0
    lt = fin[:, 1] - fin[:, 0]
    lt = lt[lt > 0.01 * np.max(lt)]
    return float(np.sum(lt))

def compute_perclass_phi(acts, labels, n_sub=300):
    total_phi = 0.0
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() >= 30:
            phi_c = compute_phi(acts[mask], n_sub=min(n_sub, mask.sum()))
            total_phi += phi_c
    return total_phi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', required=True, choices=['asymmetric', 'instance'])
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')
    
    print(f"=== ViT {args.noise_type} rate={args.rate} seed={args.seed} ===")
    
    # Transforms (lighter for ViT)
    transform_train = T.Compose([
        T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = T.Compose([T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trainset = torchvision.datasets.CIFAR10('~/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10('~/data', train=False, download=True, transform=transform_test)
    
    # Apply noise
    noisy_targets, actual_rate = apply_noise(trainset.targets, args.noise_type, args.rate, args.seed)
    trainset.targets = noisy_targets
    print(f"Actual noise rate: {actual_rate:.3f}")
    
    train_loader = DataLoader(trainset, BS, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, BS, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ViTTiny().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    
    outdir = os.path.expanduser(f'~/results_vit_noisetype/{args.noise_type}_r{args.rate}_s{args.seed}')
    os.makedirs(outdir, exist_ok=True)
    
    results = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(X)
            correct += (out.argmax(1) == y).sum().item()
            total += len(X)
        sched.step()
        
        if epoch in CHECKPOINTS:
            # Extract features
            model.eval()
            feats_list, labels_list = [], []
            with torch.no_grad():
                n = 0
                for X, y in train_loader:
                    X = X.to(device)
                    feats, _ = model(X, return_features=True)
                    feats_list.append(feats.cpu().numpy())
                    labels_list.append(y.numpy())
                    n += len(X)
                    if n >= 10000: break
            acts = np.concatenate(feats_list)[:10000]
            labs = np.concatenate(labels_list)[:10000]
            
            global_phi = compute_phi(acts, n_sub=300)
            perclass_phi = compute_perclass_phi(acts, labs, n_sub=300)
            
            # Test acc
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    test_correct += (model(X).argmax(1) == y).sum().item()
                    test_total += len(y)
            
            row = {
                'epoch': epoch, 'noise_type': args.noise_type, 'rate': args.rate, 'seed': args.seed,
                'global_phi': global_phi, 'perclass_phi': perclass_phi,
                'train_acc': correct / total * 100, 'test_acc': test_correct / test_total * 100,
                'train_loss': loss_sum / total,
            }
            results.append(row)
            print(f"  ep{epoch}: global={global_phi:.1f}, perclass={perclass_phi:.1f}, "
                  f"train={row['train_acc']:.1f}%, test={row['test_acc']:.1f}%")
    
    # Save
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(outdir, 'results.csv'), index=False)
    print(f"\nSaved: {outdir}/")
    
    # Summary
    print(f"\n=== SUMMARY: ViT {args.noise_type} rate={args.rate} ===")
    for r in results:
        if r['epoch'] in [10, 25, 50]:
            print(f"  ep{r['epoch']}: perclass_Φ={r['perclass_phi']:.1f}, test={r['test_acc']:.1f}%")

if __name__ == "__main__":
    main()
