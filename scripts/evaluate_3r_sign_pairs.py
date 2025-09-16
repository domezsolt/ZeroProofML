import json
import argparse
import os
from typing import List, Dict, Any

import numpy as np

from zeroproof.metrics.pole_3r import compute_paired_sign_consistency_3r


def _load_test_inputs(dataset_file: str, n_test: int) -> List[List[float]]:
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    samples = data['samples']
    n_train_full = int(0.8 * len(samples))
    test_samples = samples[n_train_full:n_train_full + n_test]
    X = []
    for s in test_samples:
        X.append([float(s['theta1']), float(s['theta2']), float(s['theta3']), float(s['dx']), float(s['dy'])])
    return X


def _preds_from_result(entry: Dict[str, Any]) -> List[List[float]]:
    # MLP stores under test_metrics.predictions; others under predictions
    if 'test_metrics' in entry and 'predictions' in entry['test_metrics']:
        return entry['test_metrics']['predictions']
    return entry.get('predictions', [])


def main():
    ap = argparse.ArgumentParser(description='Evaluate 3R paired sign consistency under direction window')
    ap.add_argument('--results', required=True, help='Path to comprehensive_comparison_3r.json')
    ap.add_argument('--phi', type=float, default=60.0, help='Direction angle in degrees (atan2(dy,dx))')
    ap.add_argument('--phi_tol', type=float, default=35.0, help='Direction angle tolerance in degrees')
    ap.add_argument('--k', type=int, default=4, help='Number of pairs')
    ap.add_argument('--th_window', type=float, default=0.35, help='Window around zero for |theta_j|')
    ap.add_argument('--min_mag', type=float, default=5e-4, help='Min |dtheta_j| to count pair')
    ap.add_argument('--out', type=str, default=None, help='Optional JSON output path')
    args = ap.parse_args()

    with open(args.results, 'r') as f:
        res = json.load(f)
    dataset_file = res.get('dataset')
    n_test = res.get('n_test') or 0
    test_inputs = _load_test_inputs(dataset_file, int(n_test))

    summary: Dict[str, Any] = {
        'phi_deg': args.phi,
        'phi_tol_deg': args.phi_tol,
        'k': int(args.k),
        'th_window': float(args.th_window),
        'min_mag': float(args.min_mag),
        'models': {}
    }

    for display, key in [('MLP', 'MLP'), ('Rational+eps', 'Rational+eps'), ('TR-Basic', 'TR_Basic'), ('TR-Full', 'TR_Full')]:
        entry = res['results'].get(key)
        if not entry:
            continue
        preds = _preds_from_result(entry)
        # Ensure sizes line up; truncate if needed
        m = min(len(test_inputs), len(preds))
        tinp = test_inputs[:m]
        pp = preds[:m]
        th2 = compute_paired_sign_consistency_3r(tinp, pp, joint='theta2', phi_deg=args.phi, phi_tol_deg=args.phi_tol, th_window=args.th_window, k=args.k, min_mag=args.min_mag)
        th3 = compute_paired_sign_consistency_3r(tinp, pp, joint='theta3', phi_deg=args.phi, phi_tol_deg=args.phi_tol, th_window=args.th_window, k=args.k, min_mag=args.min_mag)
        summary['models'][display] = {
            'theta2': th2,
            'theta3': th3,
        }

    print('3R paired sign consistency (direction-windowed):')
    print(f"phi={args.phi}±{args.phi_tol} deg, k={args.k}, |theta_j|≤{args.th_window}, min|dtheta_j|={args.min_mag}")
    for name, stats in summary['models'].items():
        t2 = stats['theta2']
        t3 = stats['theta3']
        print(f"- {name:12s}  theta2: {t2['rate']*100:5.2f}% (pairs={t2['pairs']}),  theta3: {t3['rate']*100:5.2f}% (pairs={t3['pairs']})")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == '__main__':
    main()

