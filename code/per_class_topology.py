#!/usr/bin/env python3
"""
EXP 4: Per-Class Topology Analysis — Mechanistic Validation of Folding
Compute H1 per class, compare clean vs noisy training.
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
        return None
    shuffle_match = re.search(r'_shuffle([\d.]+)', name)
    if shuffle_match:
        info['shuffle_fraction'] = float(shuffle_match.group(1))
        name = name[:shuffle_match.start()]
    else:
        return None
    known_archs = ['resnet101', 'resnet50', 'vit_tiny_large', 'vit_tiny', 'resnet18', 'distilbert']
    info['architecture'] = 'unknown'
    info['dataset'] = name
    for arch in sorted(known_archs, key=len, reverse=True):
        if f'_{arch}' in name:
            info['dataset'] = name.split(f'_{arch}')[0]
            info['architecture'] = arch
            break
    return info

def compute_h1(points, subsample=None, seed=42):
    if len(points) < 5:
        return 0.0, 0, 0.0
    rng = np.random.RandomState(seed)
    if subsample and subsample < len(points):
        idx = rng.choice(len(points), subsample, replace=False)
        points = points[idx]
    start = time.time()
    result = ripser(points, maxdim=1)
    elapsed = time.time() - start
    h1 = result['dgms'][1]
    total = np.sum(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0.0
    return float(total), len(h1), elapsed

def analyze_dataset(activations_dir, dataset, arch, seed, epoch, output_rows):
    clean_act_file = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.0_seed{seed}_epoch{epoch}.npy")
    clean_lab_file = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.0_seed{seed}_epoch{epoch}_labels.npy")
    noisy_act_file = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.5_seed{seed}_epoch{epoch}.npy")
    noisy_lab_file = os.path.join(activations_dir, f"{dataset}_{arch}_shuffle0.5_seed{seed}_epoch{epoch}_labels.npy")
    for f in [clean_act_file, clean_lab_file, noisy_act_file, noisy_lab_file]:
        if not os.path.exists(f):
            return
    clean_act = np.load(clean_act_file)
    clean_lab = np.load(clean_lab_file)
    noisy_act = np.load(noisy_act_file)
    noisy_lab = np.load(noisy_lab_file)
    mislabeled = (clean_lab != noisy_lab)
    overall_noise_rate = np.mean(mislabeled)
    classes = sorted(np.unique(clean_lab))
    per_class_subsample = 50
    for cls in classes:
        clean_mask = (clean_lab == cls)
        clean_cls_act = clean_act[clean_mask]
        noisy_cls_act = noisy_act[clean_mask]
        cls_mislabeled = mislabeled[clean_mask]
        cls_noise_rate = np.mean(cls_mislabeled)
        n_cls_samples = int(np.sum(clean_mask))
        clean_h1, clean_nfeat, _ = compute_h1(clean_cls_act, subsample=per_class_subsample, seed=seed)
        noisy_h1, noisy_nfeat, _ = compute_h1(noisy_cls_act, subsample=per_class_subsample, seed=seed)
        h1_delta = noisy_h1 - clean_h1
        h1_ratio = noisy_h1 / clean_h1 if clean_h1 > 0 else 999.0
        output_rows.append({
            'dataset': dataset, 'architecture': arch, 'seed': seed, 'epoch': epoch,
            'class_id': int(cls), 'n_samples': n_cls_samples,
            'cls_noise_rate': round(cls_noise_rate, 4),
            'overall_noise_rate': round(overall_noise_rate, 4),
            'clean_h1': round(clean_h1, 4), 'noisy_h1': round(noisy_h1, 4),
            'h1_delta': round(h1_delta, 4), 'h1_ratio': round(h1_ratio, 4),
            'clean_n_features': clean_nfeat, 'noisy_n_features': noisy_nfeat,
        })
    # Confusion pairs for CIFAR-10
    if dataset == 'cifar10' and len(classes) == 10:
        for c1, c2 in [(3,5), (9,1), (2,0), (4,7)]:
            pair_mask = (clean_lab == c1) | (clean_lab == c2)
            pair_sub = min(100, int(np.sum(pair_mask)))
            clean_pair_h1, _, _ = compute_h1(clean_act[pair_mask], subsample=pair_sub, seed=seed)
            noisy_pair_h1, _, _ = compute_h1(noisy_act[pair_mask], subsample=pair_sub, seed=seed)
            output_rows.append({
                'dataset': dataset, 'architecture': arch, 'seed': seed, 'epoch': epoch,
                'class_id': f'{c1}+{c2}', 'n_samples': int(np.sum(pair_mask)),
                'cls_noise_rate': round(float(np.mean(mislabeled[pair_mask])), 4),
                'overall_noise_rate': round(overall_noise_rate, 4),
                'clean_h1': round(clean_pair_h1, 4), 'noisy_h1': round(noisy_pair_h1, 4),
                'h1_delta': round(noisy_pair_h1 - clean_pair_h1, 4),
                'h1_ratio': round(noisy_pair_h1 / clean_pair_h1, 4) if clean_pair_h1 > 0 else 999.0,
                'clean_n_features': 0, 'noisy_n_features': 0,
            })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_dir', default=os.path.expanduser('~/results_phase_transition/activations'))
    parser.add_argument('--output', default=os.path.expanduser('~/results_phase_transition/per_class_topology.csv'))
    parser.add_argument('--epochs', type=str, default='10,25,50')
    args = parser.parse_args()
    epochs = [int(e) for e in args.epochs.split(',')]
    all_files = os.listdir(args.activations_dir)
    act_files = [f for f in all_files if not f.endswith('_labels.npy') and f.endswith('.npy')]
    configs = set()
    for f in act_files:
        if 'shuffle0.5' in f:
            info = parse_filename(f)
            if info:
                configs.add((info['dataset'], info['architecture'], info['seed']))
    print(f"Found {len(configs)} configs with noisy data")
    print(f"Epochs: {epochs}")
    results = []
    for i, (dataset, arch, seed) in enumerate(sorted(configs)):
        for epoch in epochs:
            print(f"[{i+1}/{len(configs)}] {dataset}/{arch}/seed{seed}/ep{epoch}")
            analyze_dataset(args.activations_dir, dataset, arch, seed, epoch, results)
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDONE: {len(results)} rows -> {args.output}")
        single = [r for r in results if '+' not in str(r['class_id'])]
        if single:
            deltas = [r['h1_delta'] for r in single]
            print(f"\n=== KEY RESULTS ===")
            print(f"  Mean H1 delta (noisy-clean): {np.mean(deltas):.2f}")
            print(f"  % classes with higher H1 under noise: {np.mean([d>0 for d in deltas])*100:.0f}%")
            for ds in sorted(set(r['dataset'] for r in single)):
                ds_d = [r['h1_delta'] for r in single if r['dataset'] == ds]
                print(f"  {ds}: {np.mean([d>0 for d in ds_d])*100:.0f}% increased, mean_delta={np.mean(ds_d):.2f}")

if __name__ == '__main__':
    main()
