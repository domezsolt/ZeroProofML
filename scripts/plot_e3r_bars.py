import json
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Plot per-bucket MSE bars for 3R baselines")
    ap.add_argument('--json', required=True, help='Path to comprehensive_comparison_3r.json')
    ap.add_argument('--out', required=True, help='Output PNG path')
    ap.add_argument('--models', nargs='+', default=['MLP', 'Rational+eps', 'TR_Basic', 'TR_Full'],
                   help='Models to include (keys in results dict)')
    args = ap.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    edges = data.get('bucket_edges') or [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float('inf')]
    bucket_labels = [f"({edges[i]:.0e},{edges[i+1]:.0e}]" for i in range(len(edges)-1)]

    # Collect per-model bucket means
    series = {}
    for name in args.models:
        r = results.get(name)
        if not r:
            continue
        nb = r.get('near_pole_bucket_mse') or {}
        means = nb.get('bucket_mse') or {}
        series[name] = [means.get(lbl, None) for lbl in bucket_labels]

    # Plot grouped bars
    import matplotlib.pyplot as plt
    N = len(bucket_labels)
    M = len(series)
    model_names = list(series.keys())
    x = np.arange(N)
    width = 0.8 / max(1, M)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(model_names):
        ys = series[m]
        # Replace None with np.nan for plotting
        yv = [np.nan if v is None else float(v) for v in ys]
        ax.bar(x + i * width - (width * (M - 1) / 2), yv, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=25, ha='right')
    ax.set_ylabel('Per-bucket MSE')
    ax.set_title('E3 (3R, quick) per-bucket MSE by manipulability')
    ax.legend(frameon=False, ncol=min(4, M))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()

