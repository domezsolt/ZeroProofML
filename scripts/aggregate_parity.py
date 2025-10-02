#!/usr/bin/env python3
"""
Aggregate quick parity results (mean±std) across seeds.

Usage (from repo root):
  python scripts/aggregate_parity.py \
    --pattern 'results/robotics/quick_s*/comprehensive_comparison.json'

Optional outputs:
  --json_out aggregated_metrics.json

Notes:
  - Aggregates overall MSE, per-bucket MSE (by |det(J)| edges), and 2D pole metrics.
  - Methods aggregated by default: MLP, Rational+ε, ZeroProofML-Basic, ZeroProofML-Full.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def mean_std(xs):
    xs = list(xs)
    if not xs:
        return None, None
    if np is not None:
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(arr.std())
    # Fallback pure-Python
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var**0.5


def aggregate(pattern: str, methods):
    runs = sorted(glob.glob(pattern))
    if not runs:
        print(f"No runs found matching pattern: {pattern}")
        return None

    agg_overall = {m: [] for m in methods}
    agg_bucket = {m: defaultdict(list) for m in methods}
    agg_pole = {m: defaultdict(list) for m in methods}
    edges_cached = None

    for path in runs:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            continue
        indiv = data.get("individual_results", {})
        edges = data.get("dataset_info", {}).get("bucket_edges")
        if isinstance(edges, list):
            edges_cached = edges

            def _fmt_edge(e):
                try:
                    v = float(e)
                except Exception:
                    s = str(e).strip().lower()
                    if s in ("inf", "+inf", "infinity"):
                        v = float("inf")
                    elif s in ("-inf", "ninf", "-infinity"):
                        v = float("-inf")
                    else:
                        return str(e)
                if math.isinf(v):
                    return "inf" if v > 0 else "-inf"
                try:
                    return f"{v:.0e}"
                except Exception:
                    return str(v)

            bucket_keys = [
                f"({_fmt_edge(edges[i])},{_fmt_edge(edges[i+1])}]" for i in range(len(edges) - 1)
            ]
        else:
            # Fallback: attempt to extract keys from any method present
            bucket_keys = None

        # If TR per-bucket is missing (quick runs), reconstruct using TR result files
        try:
            # Only attempt when ZeroProof entries exist but lack buckets
            ds_file = data.get("dataset_info", {}).get("file")
            if ds_file and os.path.isfile(ds_file):
                run_dir = os.path.dirname(path)

                # Helper to compute TR buckets for a given result file
                def _compute_tr_buckets(res_file: str):
                    try:
                        with open(res_file, "r") as fh2:
                            trres = json.load(fh2)
                    except Exception:
                        return None
                    preds = trres.get("predictions") or trres.get("test_metrics", {}).get(
                        "predictions"
                    )
                    if not isinstance(preds, list) or not preds:
                        return None
                    # Load dataset samples
                    with open(ds_file, "r") as fds:
                        dset = json.load(fds)
                    samples = dset.get("samples", [])
                    if not samples:
                        return None
                    # Rebuild quick selected test indices as in compare_all.py
                    n_train_full = int(0.8 * len(samples))
                    test_idx = list(range(n_train_full, len(samples)))
                    # Edges from JSON or default
                    raw_edges = (
                        edges
                        if isinstance(edges, list)
                        else [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]
                    )

                    def _to_float(x):
                        try:
                            return float(x)
                        except Exception:
                            s = str(x).strip().lower()
                            if s in ("inf", "+inf", "infinity"):
                                return float("inf")
                            return float(x)

                    ed = [_to_float(e) for e in raw_edges]
                    import math as _m

                    # Bucketize test indices by |sin(theta2)| with boundary rules matching compare_all
                    buckets = {i: [] for i in range(len(ed) - 1)}
                    for i_idx in test_idx:
                        th2 = float(samples[i_idx]["theta2"])
                        dj = abs(_m.sin(th2))
                        for b in range(len(ed) - 1):
                            lo, hi = ed[b], ed[b + 1]
                            if ((dj >= lo) if b == 0 else (dj > lo)) and dj <= hi:
                                buckets[b].append(i_idx)
                                break
                    # Build selected_test order up to len(preds)
                    selected = []
                    # Preselect one from B0–B3 if available
                    for b in range(min(4, len(ed) - 1)):
                        if buckets.get(b):
                            selected.append(buckets[b][0])
                            buckets[b] = buckets[b][1:]
                    # Round-robin fill
                    rr_order = [b for b in range(len(ed) - 1) if buckets.get(b)]
                    ptrs = {b: 0 for b in rr_order}
                    while len(selected) < min(len(preds), len(test_idx)) and rr_order:
                        new_rr = []
                        for b in rr_order:
                            blist = buckets.get(b, [])
                            p = ptrs[b]
                            if p < len(blist):
                                selected.append(blist[p])
                                ptrs[b] = p + 1
                                if ptrs[b] < len(blist):
                                    new_rr.append(b)
                            if len(selected) >= min(len(preds), len(test_idx)):
                                break
                        rr_order = new_rr
                        if not rr_order:
                            break
                    # Fallback if still short
                    if len(selected) < min(len(preds), len(test_idx)):
                        remaining = [i for i in test_idx if i not in selected]
                        need = min(len(preds), len(test_idx)) - len(selected)
                        selected.extend(remaining[:need])

                    # Compute per-sample MSE and aggregate per bucket key string
                    def _key_for(dj):
                        for bi in range(len(ed) - 1):
                            lo, hi = ed[bi], ed[bi + 1]
                            if (dj > lo) and (dj <= hi):

                                def fmt(v):
                                    return "inf" if _m.isinf(v) else f"{v:.0e}"

                                return f"({fmt(lo)},{fmt(hi)}]"
                        return f"({ed[-2]:.0e},inf]"

                    bucket_vals = defaultdict(list)
                    for k, idx in enumerate(selected):
                        try:
                            pred = preds[k]
                            tgt = [float(samples[idx]["dtheta1"]), float(samples[idx]["dtheta2"])]
                            mse = (
                                (float(pred[0]) - tgt[0]) ** 2 + (float(pred[1]) - tgt[1]) ** 2
                            ) / 2.0
                            dj = abs(_m.sin(float(samples[idx]["theta2"])))
                            bucket_vals[_key_for(dj)].append(float(mse))
                        except Exception:
                            continue
                    # Compute means and counts
                    bucket_mse = {}
                    bucket_counts = {}

                    # Normalize keys to match compare_all format
                    def _norm_key(lo, hi):
                        def fmt(v):
                            return "inf" if _m.isinf(v) else f"{v:.0e}"

                        return f"({fmt(lo)},{fmt(hi)}]"

                    for bi in range(len(ed) - 1):
                        lo, hi = ed[bi], ed[bi + 1]
                        key = _norm_key(lo, hi)
                        xs = bucket_vals.get(key, [])
                        if xs:
                            mu = sum(xs) / len(xs)
                        else:
                            mu = None
                        bucket_mse[key] = mu
                        bucket_counts[key] = len(xs)
                    return {"edges": ed, "bucket_mse": bucket_mse, "bucket_counts": bucket_counts}

                # Compute for TR Basic
                zpb = indiv.get("ZeroProofML-Basic", {})
                if isinstance(zpb, dict) and not zpb.get("near_pole_bucket_mse"):
                    res_file = os.path.join(
                        run_dir, "zeroproof_basic", "tr_rational_basic_results.json"
                    )
                    nb = _compute_tr_buckets(res_file)
                    if nb:
                        zpb["near_pole_bucket_mse"] = nb
                        indiv["ZeroProofML-Basic"] = zpb
                # Compute for TR Full
                zpf = indiv.get("ZeroProofML-Full", {})
                if isinstance(zpf, dict) and not zpf.get("near_pole_bucket_mse"):
                    res_file = os.path.join(
                        run_dir, "zeroproof_full", "zeroproof_full_results.json"
                    )
                    nf = _compute_tr_buckets(res_file)
                    if nf:
                        zpf["near_pole_bucket_mse"] = nf
                        indiv["ZeroProofML-Full"] = zpf
        except Exception as _e:
            # Non-fatal; continue aggregation
            pass

        # Overall MSE collection
        for label in methods:
            res = indiv.get(label, {})
            if not isinstance(res, dict) or not res:
                continue
            if label.startswith("ZeroProofML") and "final_mse" in res:
                agg_overall[label].append(res["final_mse"])
            elif label in ("MLP", "Rational+ε") and "test_metrics" in res:
                tm = res.get("test_metrics", {})
                if isinstance(tm, dict) and "mse" in tm:
                    agg_overall[label].append(tm.get("mse"))

        # Per-bucket MSE
        for label in methods:
            res = indiv.get(label, {})
            if not isinstance(res, dict):
                continue
            nb = res.get("near_pole_bucket_mse", {})
            b_mse = nb.get("bucket_mse", {}) if isinstance(nb, dict) else {}
            if bucket_keys is None and b_mse:
                bucket_keys = list(b_mse.keys())
            for k in bucket_keys or []:
                v = b_mse.get(k)
                if isinstance(v, (int, float)):
                    agg_bucket[label][k].append(v)

        # 2D pole metrics
        for label in methods:
            res = indiv.get(label, {})
            if not isinstance(res, dict):
                continue
            pm = res.get("pole_metrics", {})
            if not isinstance(pm, dict):
                continue
            for k in ("ple", "sign_consistency", "slope_error", "residual_consistency"):
                v = pm.get(k)
                if isinstance(v, (int, float)):
                    agg_pole[label][k].append(v)

    # Build aggregated structure
    # Build aggregated structure (include raw arrays for plotting)
    summary = {
        "pattern": pattern,
        "methods": methods,
        "bucket_edges": edges_cached,
        "overall_mse": {},
        "per_bucket_mse": {},
        "pole_metrics": {},
        "overall_mse_raw": {m: list(xs) for m, xs in agg_overall.items()},
        "per_bucket_mse_raw": {
            m: {k: list(vs) for k, vs in bm.items()} for m, bm in agg_bucket.items()
        },
        "pole_metrics_raw": {
            m: {k: list(vs) for k, vs in mp.items()} for m, mp in agg_pole.items()
        },
    }

    print("Overall MSE (mean±std):")
    for m, xs in agg_overall.items():
        if xs:
            mu, sd = mean_std(xs)
            print(f"  {m:18s}: {mu:.6f} ± {sd:.6f} (n={len(xs)})")
            summary["overall_mse"][m] = {"mean": mu, "std": sd, "n": len(xs)}

    print("\nPer-bucket MSE (mean±std):")
    for m, bm in agg_bucket.items():
        if not bm:
            continue
        print(f"  {m}:")
        summary["per_bucket_mse"][m] = {}
        for k, xs in bm.items():
            mu, sd = mean_std(xs)
            print(f"    {k:16s} {mu:.6f} ± {sd:.6f} (n={len(xs)})")
            summary["per_bucket_mse"][m][k] = {"mean": mu, "std": sd, "n": len(xs)}

    print("\n2D pole metrics (mean±std):")
    for m, mp in agg_pole.items():
        if not mp:
            continue
        print(f"  {m}:")
        summary["pole_metrics"][m] = {}
        for k, xs in mp.items():
            if not xs:
                continue
            mu, sd = mean_std(xs)
            print(f"    {k:20s}: {mu:.6f} ± {sd:.6f} (n={len(xs)})")
            summary["pole_metrics"][m][k] = {"mean": mu, "std": sd, "n": len(xs)}

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate quick parity results across seeds")
    parser.add_argument(
        "--pattern",
        default="results/robotics/quick_s*/comprehensive_comparison.json",
        help="Glob pattern for comprehensive_comparison.json files",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["MLP", "Rational+ε", "ZeroProofML-Basic", "ZeroProofML-Full"],
        help="Methods to aggregate (labels must match JSON)",
    )
    parser.add_argument(
        "--json_out",
        default=None,
        help="Optional path to write aggregated metrics as JSON",
    )
    args = parser.parse_args()

    summary = aggregate(args.pattern, args.methods)
    if summary is None:
        return
    if args.json_out:
        try:
            with open(args.json_out, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"\nAggregated metrics saved to {args.json_out}")
        except Exception as e:
            print(f"Warning: failed to write {args.json_out}: {e}")


if __name__ == "__main__":
    main()
