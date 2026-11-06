#!/usr/bin/env python3
"""
EXP 7: Alternative PH summaries on existing activations.
Computes: total persistence, persistence entropy, max persistence,
persistence variance, number of significant features.
Shows crossover is robust to choice of summary statistic.
"""
import numpy as np
import os, glob, re, csv, time, argparse
from ripser import ripser

def parse_filename(filename):
    name = os.path.basename(filename).replace('.npy', '')
    if name.endswith('_labels'):
        return None
    info = {}
    epoch_match = re.search(r'_epoch(\d+)$', name)
    if epoch_match:
        info['epoch'] = int(epoch_match.group(1))
        name = name[:epoch_match.start()]
    else:
        return None
    seed_match = re.search(r'_seed(\d+)', name)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
        name = name[:seed_match.start()] + name[seed_match.end():]
    else:
        info['seed'] = 0
    shuffle_match = re.search(r'_shuffle([\d.]+)', name)
    if shuffle_match:
        info['shuffle_fraction'] = float(shuffle_match.group(1))
        name = name[:shuffle_match.start()]
    else:
        info['shuffle_fraction'] = 0.0
    known_archs = ['resnet101','resnet50','vit_tiny_large','vit_tiny','resnet18','distilbert','mlp']
    info['architecture'] = 'unknown'
    info['dataset'] = name
    for arch in sorted(known_archs, key=len, reverse=True):
        if f'_{arch}' in name:
            info['dataset'] = name.split(f'_{arch}')[0]
            info['architecture'] = arch
            break
    return info

def compute_all_summaries(activations, subsample=300, seed=42):
    rng = np.random.RandomState(seed)
    n = len(activations)
    if subsample < n:
        idx = rng.choice(n, subsample, replace=False)
        activations = activations[idx]
    start = time.time()
    result = ripser(activations, maxdim=1)
    elapsed = time.time() - start
    h1 = result['dgms'][1]
    if len(h1) == 0:
        return {
            'total_persistence': 0.0, 'persistence_entropy': 0.0,
            'max_persistence': 0.0, 'persistence_variance': 0.0,
            'n_significant': 0, 'mean_persistence': 0.0,
            'median_persistence': 0.0, 'n_features': 0,
            'compute_time': round(elapsed, 3),
        }
    lifetimes = h1[:, 1] - h1[:, 0]
    total = float(np.sum(lifetimes))
    # Persistence entropy
    if total > 0:
        probs = lifetimes / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
    else:
        entropy = 0.0
    median_life = float(np.median(lifetimes))
    n_significant = int(np.sum(lifetimes > median_life))
    return {
        'total_persistence': round(total, 4),
        'persistence_entropy': round(entropy, 4),
        'max_persistence': round(float(np.max(lifetimes)), 4),
        'persistence_variance': round(float(np.var(lifetimes)), 4),
        'n_significant': n_significant,
        'mean_persistence': round(float(np.mean(lifetimes)), 4),
        'median_persistence': round(median_life, 4),
        'n_features': len(h1),
        'compute_time': round(elapsed, 3),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_dir', default=os.path.expanduser('~/results_phase_transition/activations'))
    parser.add_argument('--output', default=os.path.expanduser('~/results_phase_transition/alt_ph_summaries.csv'))
    args = parser.parse_args()
    
    all_files = sorted(glob.glob(os.path.join(args.activations_dir, '*.npy')))
    act_files = [f for f in all_files if not f.endswith('_labels.npy')]
    
    # Focus on key configs: epoch 10, 25, 50 for main datasets
    target_files = []
    for f in act_files:
        info = parse_filename(f)
        if info and info['epoch'] in [10, 25, 50]:
            target_files.append((f, info))
    
    print(f"Processing {len(target_files)} files")
    results = []
    
    for i, (filepath, info) in enumerate(target_files):
        try:
            activations = np.load(filepath)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        summaries = compute_all_summaries(activations, subsample=300, seed=info['seed'])
        row = {
            'file': os.path.basename(filepath),
            'dataset': info['dataset'],
            'architecture': info['architecture'],
            'shuffle_fraction': info['shuffle_fraction'],
            'seed': info['seed'],
            'epoch': info['epoch'],
            **summaries,
        }
        results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(target_files)}]")
    
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDONE: {len(results)} rows -> {args.output}")
        
        # Compare summaries: which ones show the crossover?
        from scipy import stats
        print("\n=== DISCRIMINATION POWER AT EPOCH 10 ===")
        ep10 = [r for r in results if r['epoch'] == 10]
        if len(ep10) >= 5:
            sfs = [r['shuffle_fraction'] for r in ep10]
            for metric in ['total_persistence', 'persistence_entropy', 'max_persistence',
                          'persistence_variance', 'n_significant', 'mean_persistence']:
                vals = [r[metric] for r in ep10]
                if len(set(sfs)) >= 2:
                    rho, p = stats.spearmanr(vals, sfs)
                    print(f"  {metric:25s}: rho={rho:+.3f} (p={p:.4f})")
        
        print("\n=== DISCRIMINATION POWER AT EPOCH 50 ===")
        ep50 = [r for r in results if r['epoch'] == 50]
        if len(ep50) >= 5:
            sfs = [r['shuffle_fraction'] for r in ep50]
            for metric in ['total_persistence', 'persistence_entropy', 'max_persistence',
                          'persistence_variance', 'n_significant', 'mean_persistence']:
                vals = [r[metric] for r in ep50]
                if len(set(sfs)) >= 2:
                    rho, p = stats.spearmanr(vals, sfs)
                    print(f"  {metric:25s}: rho={rho:+.3f} (p={p:.4f})")

if __name__ == '__main__':
    main()
