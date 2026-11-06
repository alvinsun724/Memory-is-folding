#!/usr/bin/env python3
"""Compute PH on existing corruption & pretrained activation files."""
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
    corrupt_match = re.search(r'_corrupt_(\w+)', name)
    if corrupt_match:
        info['corruption'] = corrupt_match.group(1)
        name = name[:corrupt_match.start()] + name[corrupt_match.end():]
    else:
        info['corruption'] = 'none'
    if '_pretrained' in name:
        info['pretrained'] = True
        name = name.replace('_pretrained', '')
    else:
        info['pretrained'] = False
    shuffle_match = re.search(r'_shuffle([\d.]+)', name)
    if shuffle_match:
        info['shuffle_fraction'] = float(shuffle_match.group(1))
        name = name[:shuffle_match.start()]
    else:
        info['shuffle_fraction'] = 0.0
    known_archs = ['resnet101', 'resnet50', 'vit_tiny_large', 'vit_tiny', 'resnet18']
    info['dataset'] = name
    info['architecture'] = 'unknown'
    for arch in known_archs:
        if f'_{arch}' in name:
            info['dataset'] = name.split(f'_{arch}')[0]
            info['architecture'] = arch
            break
    return info

def compute_ph(activations, subsample_size=300, seed=42):
    rng = np.random.RandomState(seed)
    n = len(activations)
    if subsample_size < n:
        idx = rng.choice(n, subsample_size, replace=False)
        subset = activations[idx]
    else:
        subset = activations
    start = time.time()
    result = ripser(subset, maxdim=1)
    elapsed = time.time() - start
    h0 = result['dgms'][0]
    h1 = result['dgms'][1]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_total = np.sum(h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else 0
    h1_total = np.sum(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0
    return {
        'total_persistence_h0': round(float(h0_total), 4),
        'total_persistence_h1': round(float(h1_total), 4),
        'n_features_h0': len(h0_finite),
        'n_features_h1': len(h1),
        'compute_time': round(elapsed, 3),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_dir', default=os.path.expanduser('~/results_largescale/activations'))
    parser.add_argument('--existing_ph', default=os.path.expanduser('~/results_largescale/ph_largescale.csv'))
    parser.add_argument('--output', default=os.path.expanduser('~/results_largescale/ph_corruption_pretrained.csv'))
    parser.add_argument('--subsample', type=int, default=300)
    args = parser.parse_args()
    all_files = sorted(glob.glob(os.path.join(args.activations_dir, '*.npy')))
    act_files = [f for f in all_files if not f.endswith('_labels.npy')]
    print(f"Total activation files: {len(act_files)}")
    target_files = [f for f in act_files if 'corrupt' in os.path.basename(f) or 'pretrained' in os.path.basename(f)]
    print(f"Corruption/pretrained to process: {len(target_files)}")
    results = []
    errors = 0
    for i, filepath in enumerate(target_files):
        basename = os.path.basename(filepath)
        info = parse_filename(basename)
        if info is None:
            continue
        try:
            activations = np.load(filepath)
        except Exception as e:
            print(f"  ERROR loading {basename}: {e}")
            errors += 1
            continue
        ph = compute_ph(activations, subsample_size=args.subsample, seed=info['seed'])
        row = {
            'file': basename,
            'dataset': info['dataset'],
            'architecture': info['architecture'],
            'shuffle_fraction': info['shuffle_fraction'],
            'seed': info['seed'],
            'epoch': info['epoch'],
            'corruption': info['corruption'],
            'pretrained': info['pretrained'],
            'n_samples': len(activations),
            'n_features_dim': activations.shape[1],
            'subsample_size': args.subsample,
            **ph,
        }
        results.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(target_files)}] {basename}: H1={ph['total_persistence_h1']:.1f} time={ph['compute_time']:.2f}s")
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDONE: {len(results)} rows -> {args.output}")
        print(f"Errors: {errors}")
        print("\n=== SUMMARY ===")
        for ctype in sorted(set(r['corruption'] for r in results)):
            subset = [r for r in results if r['corruption'] == ctype]
            h1_vals = [r['total_persistence_h1'] for r in subset]
            print(f"  {ctype}: n={len(subset)} H1_mean={np.mean(h1_vals):.1f} std={np.std(h1_vals):.1f}")
        for pt in [True, False]:
            subset = [r for r in results if r['pretrained'] == pt]
            if subset:
                h1_vals = [r['total_persistence_h1'] for r in subset]
                print(f"  {'pretrained' if pt else 'scratch'}: n={len(subset)} H1_mean={np.mean(h1_vals):.1f}")

if __name__ == '__main__':
    main()
