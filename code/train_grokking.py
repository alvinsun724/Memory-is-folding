#!/usr/bin/env python3
"""
EXP 9: Grokking Detection via Topological Phase Transition
Train small transformer on modular arithmetic (Power et al. 2022).
Track Phi(t) — hypothesis: Phi rises during memorization, drops during grokking.

Usage: python train_grokking.py --operation add --prime 97 --seed 42
"""
import torch
import torch.nn as nn
import numpy as np
import os, csv, time, argparse, math
from ripser import ripser

class ModularDataset(torch.utils.data.Dataset):
    def __init__(self, prime, operation, frac_train=0.5, split='train', seed=42):
        self.prime = prime
        rng = np.random.RandomState(seed)
        
        # All pairs (a, b) for a, b in [0, prime)
        pairs = [(a, b) for a in range(prime) for b in range(prime)]
        
        if operation == 'add':
            labels = [(a + b) % prime for a, b in pairs]
        elif operation == 'mul':
            labels = [(a * b) % prime for a, b in pairs]
        elif operation == 'sub':
            labels = [(a - b) % prime for a, b in pairs]
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Split
        n = len(pairs)
        indices = rng.permutation(n)
        n_train = int(n * frac_train)
        
        if split == 'train':
            idx = indices[:n_train]
        else:
            idx = indices[n_train:]
        
        self.inputs = torch.tensor([(pairs[i][0], pairs[i][1]) for i in idx], dtype=torch.long)
        self.labels = torch.tensor([labels[i] for i in idx], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.inputs[i], self.labels[i]

class GrokTransformer(nn.Module):
    """Small transformer for modular arithmetic (following Power et al.)."""
    def __init__(self, prime, d_model=128, nhead=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.embed_a = nn.Embedding(prime, d_model)
        self.embed_b = nn.Embedding(prime, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # positions: a, op, b
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, prime)
        self.d_model = d_model
    
    def forward(self, x):
        a, b = x[:, 0], x[:, 1]
        pos = torch.arange(2, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        emb = torch.stack([self.embed_a(a), self.embed_b(b)], dim=1)
        emb = emb + self.pos_embed(pos)
        out = self.transformer(emb)
        # Use mean pooling
        features = out.mean(dim=1)
        logits = self.head(features)
        return logits, features

def compute_phi(activations, subsample=300, seed=42):
    rng = np.random.RandomState(seed)
    n = len(activations)
    if subsample < n:
        idx = rng.choice(n, subsample, replace=False)
        activations = activations[idx]
    result = ripser(activations, maxdim=1)
    h1 = result['dgms'][1]
    return float(np.sum(h1[:, 1] - h1[:, 0])) if len(h1) > 0 else 0.0

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    all_features = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, features = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_features.append(features.cpu().numpy())
    return correct / total, np.concatenate(all_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, default='add', choices=['add', 'mul', 'sub'])
    parser.add_argument('--prime', type=int, default=97)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--ph_every', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/results_grokking'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {args.operation} mod {args.prime}, seed={args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_data = ModularDataset(args.prime, args.operation, frac_train=0.5, split='train', seed=args.seed)
    test_data = ModularDataset(args.prime, args.operation, frac_train=0.5, split='test', seed=args.seed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Model
    torch.manual_seed(args.seed)
    model = GrokTransformer(args.prime, d_model=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    results = []
    step = 0
    epoch = 0
    
    while step < args.steps:
        epoch += 1
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(1) == labels).sum().item()
            epoch_total += len(labels)
            step += 1
            
            if step >= args.steps:
                break
        
        train_acc = epoch_correct / epoch_total
        
        # Evaluate + PH at intervals
        if step % args.ph_every < len(train_loader) or step >= args.steps or epoch <= 5:
            test_acc, test_features = evaluate(model, test_loader, device)
            _, train_features = evaluate(model, train_loader, device)
            
            # Compute Phi on test features
            phi_test = compute_phi(test_features, subsample=300, seed=args.seed)
            phi_train = compute_phi(train_features, subsample=300, seed=args.seed)
            
            row = {
                'step': step,
                'epoch': epoch,
                'train_acc': round(train_acc, 4),
                'test_acc': round(test_acc, 4),
                'train_loss': round(epoch_loss / max(1, epoch_total // args.batch_size), 4),
                'phi_test': round(phi_test, 4),
                'phi_train': round(phi_train, 4),
                'operation': args.operation,
                'prime': args.prime,
                'seed': args.seed,
            }
            results.append(row)
            
            grok_status = "MEMORIZED" if train_acc > 0.99 and test_acc < 0.5 else \
                          "GROKKED" if train_acc > 0.99 and test_acc > 0.9 else \
                          "TRAINING"
            
            print(f"  Step {step:6d} | TrainAcc {train_acc:.3f} | TestAcc {test_acc:.3f} | "
                  f"Phi_test {phi_test:.1f} | Phi_train {phi_train:.1f} | {grok_status}")

    # Save results
    outfile = os.path.join(args.output_dir, f"grokking_{args.operation}_p{args.prime}_seed{args.seed}.csv")
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {outfile}")

    # Summary
    print(f"\n=== SUMMARY ===")
    if results:
        final = results[-1]
        print(f"  Final: train_acc={final['train_acc']}, test_acc={final['test_acc']}")
        print(f"  Final Phi: test={final['phi_test']:.1f}, train={final['phi_train']:.1f}")
        
        # Check for grokking signature in Phi
        memorized = [r for r in results if r['train_acc'] > 0.99 and r['test_acc'] < 0.5]
        grokked = [r for r in results if r['train_acc'] > 0.99 and r['test_acc'] > 0.9]
        
        if memorized:
            mem_phi = np.mean([r['phi_test'] for r in memorized])
            print(f"  Memorization phase: mean Phi_test = {mem_phi:.1f}")
        if grokked:
            grok_phi = np.mean([r['phi_test'] for r in grokked])
            print(f"  Grokking phase: mean Phi_test = {grok_phi:.1f}")
        if memorized and grokked:
            print(f"  Phi CHANGE: {grok_phi:.1f} - {mem_phi:.1f} = {grok_phi - mem_phi:.1f}")
            if grok_phi < mem_phi:
                print(f"  *** PHI DROPS DURING GROKKING — TOPOLOGICAL UNFOLDING DETECTED ***")

if __name__ == '__main__':
    main()
