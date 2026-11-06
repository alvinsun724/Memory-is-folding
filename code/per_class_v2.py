#!/usr/bin/env python3
"""
EXP 4v2: Inter-class topology analysis.
Key insight: folding is localised to BOUNDARIES between similar classes.
Compute H1 for ALL class pairs, correlate with centroid distance.
"""
import numpy as np
import os, glob, re, csv, time, argparse, itertools
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
        return None
    shuffle_match = re.search(r'_shuffle([\d.]+)', name)
    if shuffle_match:
        info['shuffle_fraction'] = float(shuffle_match.group(1))
        name = name[:shuffle_match.start()]
    else:
        return None
    known_archs = ['resnet101','resnet50','vit_tiny_large','vit_tiny','resnet18','distilbert']
    info['architecture'] = 'unknown'
    info['dataset'] = name
    for arch in sorted(known_archs, key=len, reverse=True):
        if f'_{arch}' in name:
            info['dataset'] = name.split(f'_{arch}')[0]
            info['architecture'] = arch
            break
    return info

def compute_h1(points, subsample=100, seed=42):
    if len(points) < 5:
        return 0.0, 0
    rng = np.random.RandomState(seed)
    if subsample and subsample < len(points):
        idx = rng.choice(len(points), subsample, replace=False)
        points = points[idx]
    result = ripser(points, maxdim=1)
    h1 = result['dgms'][1]
    return float(np.sum(h1[:, 1] - h1[:, 0])) if len(h1) > 0 else 0.0, len(h1)

def analyze(activations_dir, dataset, arch, seed, epoch, results):
    clean_f = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.0_seed{seed}_epoch{epoch}.npy")
    clean_l = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.0_seed{seed}_epoch{epoch}_labels.npy")
    noisy_f = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.5_seed{seed}_epoch{epoch}.npy")
    noisy_l = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.5_seed{seed}_epoch{epoch}_labels.npy")
    for f in [clean_f, clean_l, noisy_f, noisy_l]:
        if not os.path.exists(f):
            return
    clean_act = np.load(clean_f)
    clean_lab = np.load(clean_l)
    noisy_act = np.load(noisy_f)
    noisy_lab = np.load(noisy_l)
    classes = sorted(np.unique(clean_lab))
    if len(classes) > 20:
        return  # Skip 100-class datasets for pair analysis

    # Per-class analysis (single classes)
    for cls in classes:
        mask = (clean_lab == cls)
        c_h1, c_n = compute_h1(clean_act[mask], subsample=100, seed=seed)
        n_h1, n_n = compute_h1(noisy_act[mask], subsample=100, seed=seed)
        results.append({
            'dataset': dataset, 'architecture': arch, 'seed': seed,
            'epoch': epoch, 'pair_type': 'single', 'class_pair': str(int(cls)),
            'n_samples': int(np.sum(mask)),
            'clean_h1': round(c_h1, 4), 'noisy_h1': round(n_h1, 4),
            'h1_delta': round(n_h1 - c_h1, 4),
            'h1_ratio': round(n_h1 / c_h1, 4) if c_h1 > 0 else 999.0,
            'centroid_dist_clean': 0.0, 'centroid_dist_noisy': 0.0,
        })

    # Compute class centroids for distance calculation
    clean_centroids = {}
    noisy_centroids = {}
    for cls in classes:
        mask = (clean_lab == cls)
        clean_centroids[cls] = clean_act[mask].mean(axis=0)
        noisy_centroids[cls] = noisy_act[mask].mean(axis=0)

    # ALL class pairs
    for c1, c2 in itertools.combinations(classes, 2):
        mask = (clean_lab == c1) | (clean_lab == c2)
        pair_sub = min(150, int(np.sum(mask)))
        c_h1, _ = compute_h1(clean_act[mask], subsample=pair_sub, seed=seed)
        n_h1, _ = compute_h1(noisy_act[mask], subsample=pair_sub, seed=seed)
        # Centroid distance between the two classes
        cd_clean = float(np.linalg.norm(clean_centroids[c1] - clean_centroids[c2]))
        cd_noisy = float(np.linalg.norm(noisy_centroids[c1] - noisy_centroids[c2]))
        results.append({
            'dataset': dataset, 'architecture': arch, 'seed': seed,
            'epoch': epoch, 'pair_type': 'pair', 'class_pair': f'{int(c1)}+{int(c2)}',
            'n_samples': int(np.sum(mask)),
            'clean_h1': round(c_h1, 4), 'noisy_h1': round(n_h1, 4),
            'h1_delta': round(n_h1 - c_h1, 4),
            'h1_ratio': round(n_h1 / c_h1, 4) if c_h1 > 0 else 999.0,
            'centroid_dist_clean': round(cd_clean, 4),
            'centroid_dist_noisy': round(cd_noisy, 4),
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_dir', default=os.path.expanduser('~/results_phase_transition/activations'))
    parser.add_argument('--output', default=os.path.expanduser('~/results_phase_transition/interclass_topology.csv'))
    parser.add_argument('--epochs', type=str, default='10,25,50')
    args = parser.parse_args()
    epochs = [int(e) for e in args.epochs.split(',')]
    all_files = os.listdir(args.activations_dir)
    configs = set()
    for f in all_files:
        if 'shuffle0.5' in f and not f.endswith('_labels.npy'):
            info = parse_filename(f)
            if info and info['dataset'] in ['cifar10', 'mnist', 'fashionmnist']:
                configs.add((info['dataset'], info['architecture'], info['seed']))
    print(f"Configs: {len(configs)}, Epochs: {epochs}")
    results = []
    for i, (ds, arch, seed) in enumerate(sorted(configs)):
        for ep in epochs:
            print(f"[{i+1}/{len(configs)}] {ds}/{arch}/seed{seed}/ep{ep}")
            analyze(args.activations_dir, ds, arch, seed, ep, results)
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDONE: {len(results)} rows -> {args.output}")
        # Key analysis
        pairs = [r for r in results if r['pair_type'] == 'pair' and r['epoch'] == 50]
        if pairs:
            from scipy import stats
            dists = [r['centroid_dist_clean'] for r in pairs]
            deltas = [r['h1_delta'] for r in pairs]
            if len(dists) >= 5:
                rho, p = stats.spearmanr(dists, deltas)
                print(f"\n=== KEY RESULT ===")
                print(f"Correlation(centroid_distance, h1_delta) at epoch 50:")
                print(f"  Spearman rho={rho:.3f}, p={p:.4f}")
                print(f"  (Negative = closer classes fold MORE)")
            # Top 5 most-folded pairs
            pairs_sorted = sorted(pairs, key=lambda r: r['h1_delta'], reverse=True)
            print(f"\nTop 5 most-folded pairs (epoch 50):")
            for r in pairs_sorted[:5]:
                print(f"  {r['class_pair']}: delta={r['h1_delta']:.2f}, ratio={r['h1_ratio']:.2f}, centroid_dist={r['centroid_dist_clean']:.1f}")

if __name__ == '__main__':
    main()
