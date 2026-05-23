#!/usr/bin/env python3
"""
Demo Pipeline -- CS F429 Track A (Unsupervised Hallucination Detection)

Two modes:
  1. LIVE INFERENCE (with --query, --context, --response):
     Loads Qwen2.5-7B, runs dual forward pass, computes all metrics,
     applies CAWC v5 composite, and displays token-level results.

  2. PRE-COMPUTED (no args, or just `python demo_pipeline.py`):
     Loads pre-computed results from results/demo_sample.json.

Usage:
  python demo_pipeline.py --query "..." --context "..." --response "..."
  python demo_pipeline.py   # uses pre-computed demo sample
"""
import argparse, json, sys, os, pickle, time
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).parent / "results"
DEMO_FILE = RESULTS / "demo_sample.json"
TRAIN_CACHE = RESULTS / "train_raw_cache_v5.pkl"
if not TRAIN_CACHE.exists():
    TRAIN_CACHE = RESULTS / "train_raw_cache_v4.pkl"

METRIC_KEYS = ["entropy", "ig", "kl", "cd", "selfcheck", "se"]
DIRECTIONS = {
    "entropy": False, "ig": True, "kl": True,
    "cd": False, "selfcheck": False, "se": False
}


# ================================================================
# UTILITY FUNCTIONS (self-contained, identical to main pipeline)
# ================================================================
def fill_missing(arr, fallback_value):
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, float(fallback_value), dtype=np.float64), 0.0
    out = arr.copy()
    coverage = float(finite.mean())
    if not finite.all():
        out[~finite] = float(np.nanmedian(out[finite]))
    return out, coverage

def percentile_rank(values, sorted_train):
    ranks = np.searchsorted(sorted_train, values, side='right')
    return ranks.astype(np.float64) / max(len(sorted_train), 1)

def window_mean(arr, w):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    hw = w // 2
    padded = np.pad(arr, (hw, hw), mode='edge')
    cs = np.cumsum(padded)
    cs = np.insert(cs, 0, 0.0)
    window_sums = cs[w:] - cs[:len(cs) - w]
    out = window_sums[:n] / w
    for i in range(min(hw, n)):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        out[i] = np.mean(arr[lo:hi])
    for i in range(max(0, n - hw), n):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        out[i] = np.mean(arr[lo:hi])
    return out

def window_max(arr, w):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    hw = w // 2
    out = np.empty(n, dtype=np.float64)
    padded = np.pad(arr, (hw, hw), mode='edge')
    for i in range(n):
        out[i] = np.max(padded[i:i + w])
    return out


def compute_variance_weights(train_results):
    """Derive variance-based weights from train distribution."""
    pools = {mk: [] for mk in METRIC_KEYS}
    for r in train_results:
        scores = r.get("scores_raw_undirected", r.get("scores_raw", r.get("scores", {})))
        n = len(np.asarray(r.get("labels", [])))
        for mk in METRIC_KEYS:
            default = np.full(n, np.nan) if mk == "se" else np.zeros(n)
            arr = np.asarray(scores.get(mk, default), dtype=np.float64)[:n]
            if DIRECTIONS.get(mk, False):
                arr = -arr
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                pools[mk].extend(finite.tolist())

    raw_w = {}
    for mk in METRIC_KEYS:
        vals = np.array(pools[mk])
        raw_w[mk] = float(np.std(vals)) if len(vals) > 10 else 0.0

    total = sum(raw_w.values())
    if total < 1e-10:
        weights = {mk: 1.0 / len(METRIC_KEYS) for mk in METRIC_KEYS}
    else:
        weights = {mk: round(v / total, 4) for mk, v in raw_w.items()}

    MIN_WEIGHT = 0.01
    for mk in METRIC_KEYS:
        if weights.get(mk, 0) < MIN_WEIGHT:
            weights[mk] = MIN_WEIGHT
    w_total = sum(weights.values())
    weights = {mk: round(v / w_total, 4) for mk, v in weights.items()}
    return weights


def compute_train_stats(train_results):
    """Population CDF from train tokens (for percentile-rank normalization)."""
    directed_pools = {mk: [] for mk in METRIC_KEYS}
    for r in train_results:
        scores = r.get("scores_raw_undirected", r.get("scores_raw", r.get("scores", {})))
        n = len(np.asarray(r.get("labels", [])))
        for mk in METRIC_KEYS:
            default = np.full(n, np.nan) if mk == "se" else np.zeros(n)
            arr = np.asarray(scores.get(mk, default), dtype=np.float64)[:n]
            if DIRECTIONS.get(mk, False):
                arr = -arr
            directed_pools[mk].append(arr)
    stats = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for mk in METRIC_KEYS:
            all_vals = np.concatenate(directed_pools[mk])
            stats[mk] = {
                "mean": float(np.nanmean(all_vals)),
                "std": float(np.nanstd(all_vals)),
                "sorted": np.sort(all_vals[np.isfinite(all_vals)]),
            }
            if np.isnan(stats[mk]["mean"]):
                stats[mk]["mean"] = 0.0
            if np.isnan(stats[mk]["std"]):
                stats[mk]["std"] = 0.0
    return stats


def compute_composite_live(raw_scores, train_stats, n, weights):
    """CAWC v5 composite for a single sample."""
    if n < 2:
        return np.full(n, 0.5)

    directed = {}
    for mk in METRIC_KEYS:
        arr = raw_scores[mk][:n].copy()
        arr, _ = fill_missing(arr, train_stats[mk]["mean"])
        if DIRECTIONS.get(mk, False):
            arr = -arr
        directed[mk] = arr

    ranked = {}
    for mk in METRIC_KEYS:
        ranked[mk] = percentile_rank(directed[mk], train_stats[mk]["sorted"])

    smoothed = {}
    for mk in METRIC_KEYS:
        arr = ranked[mk]
        n_ = len(arr)
        w_back = np.zeros(n_, dtype=np.float64)
        w_sym = window_mean(arr, 9)
        for i in range(n_):
            lo, hi = max(0, i - 6), min(n_, i + 3)
            w_back[i] = np.mean(arr[lo:hi])
        smoothed[mk] = 0.30 * window_mean(arr, 5) + 0.35 * w_sym + 0.35 * w_back

    # Weighted linear combination
    z = sum(weights.values()) + 1e-10
    composite = np.zeros(n, dtype=np.float64)
    for mk, w in weights.items():
        composite += (w / z) * smoothed[mk]

    # Interaction terms
    composite += 0.06 * (smoothed["ig"] * smoothed["cd"])
    composite += 0.04 * (smoothed["kl"] * smoothed["ig"])
    composite += 0.04 * window_max(ranked["ig"], 15)
    composite += 0.04 * window_max(ranked["cd"], 15)

    # Gradient features
    grad_ig = np.zeros(n, dtype=np.float64)
    grad_cd = np.zeros(n, dtype=np.float64)
    if n > 1:
        grad_ig[1:] = np.diff(ranked["ig"])
        grad_cd[1:] = np.diff(ranked["cd"])
    composite += 0.06 * window_mean(np.clip(grad_ig, 0, None), 5)
    composite += 0.06 * window_mean(np.clip(grad_cd, 0, None), 5)

    # Forward max-pool
    composite += 0.06 * np.array([np.max(ranked["ig"][i:min(n, i+5)]) for i in range(n)])
    composite += 0.06 * np.array([np.max(ranked["cd"][i:min(n, i+5)]) for i in range(n)])

    # Local excess
    combined = 0.5 * smoothed["ig"] + 0.5 * smoothed["cd"]
    local_base = window_mean(combined, 21)
    composite += 0.05 * np.clip(combined - local_base, 0, None)

    # KL-dominance
    kl_dom = np.clip(ranked["kl"] - ranked["ig"], 0, None)
    composite += 0.04 * window_mean(kl_dom, 7)

    return composite


# ================================================================
# LIVE INFERENCE MODE
# ================================================================
def run_live_inference(query, context, response):
    """Load model, compute metrics, run composite, display results."""
    print("=" * 70)
    print("  LIVE INFERENCE -- Unsupervised Hallucination Detection")
    print("=" * 70)
    print()
    print("Proxy model: Qwen2.5-7B (4-bit NF4 quantized)")
    print("Metrics:     Entropy, IG, KL, CD, SelfCheck")
    print("Weights:     Variance-based (w_i = std_i / sum_std, unsupervised)")
    print()

    # Load train stats for normalization
    print("Loading train statistics for normalization...")
    if not TRAIN_CACHE.exists():
        print(f"ERROR: Train cache not found at {TRAIN_CACHE}")
        print("Run: python run_final_unsupervised.py first.")
        sys.exit(1)

    with open(TRAIN_CACHE, "rb") as f:
        train_data = pickle.load(f)
    train_results = train_data if isinstance(train_data, list) else train_data.get("results", [])
    print(f"  Train samples: {len(train_results)}")

    train_stats = compute_train_stats(train_results)
    weights = compute_variance_weights(train_results)
    print(f"  Weights computed from {len(train_results)} train samples")
    for mk in METRIC_KEYS:
        print(f"    {mk:<12} {weights[mk]:.4f}")
    print()

    # Load model
    print("Loading Qwen2.5-7B (4-bit NF4)...")
    from model_utils import ModelConfig, load_generator_model, dual_forward_pass
    from core_metrics import MetricEvaluator

    cfg = ModelConfig(model_name="Qwen/Qwen2.5-7B", max_gpu_memory_gib=6)
    model, tokenizer = load_generator_model(cfg)
    evaluator = MetricEvaluator(tokenizer, generator_model=model)
    print("  Model loaded successfully.")
    print()

    # Run dual forward pass
    print("Running dual forward pass (with-context vs without-context)...")
    t0 = time.time()
    p_with, p_without, resp_ids = dual_forward_pass(
        model, tokenizer, query, context, response, cfg
    )
    elapsed = time.time() - t0
    n = len(p_with)
    print(f"  {n} response tokens processed in {elapsed:.1f}s")
    print()

    if n < 2:
        print("ERROR: Response too short for meaningful analysis.")
        return

    # Compute token-level metrics
    print("Computing token-level metrics...")
    raw_scores = {mk: np.zeros(n, dtype=np.float64) for mk in METRIC_KEYS}

    for t in range(n):
        metrics = evaluator.token_metrics(p_with[t], p_without[t], int(resp_ids[t]))
        raw_scores["entropy"][t] = metrics["entropy"]
        raw_scores["ig"][t] = metrics["ig"]
        raw_scores["kl"][t] = metrics["kl"]
        raw_scores["cd"][t] = metrics["cd"]
        raw_scores["selfcheck"][t] = metrics["selfcheck"]
        raw_scores["se"][t] = np.nan  # SE requires sampling, skip for demo speed

    # Compute composite
    print("Computing CAWC v5 composite...")
    composite = compute_composite_live(raw_scores, train_stats, n, weights)

    # Decode tokens
    tokens = [tokenizer.decode([int(resp_ids[t])]) for t in range(n)]

    # Display results
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Query:    {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"  Context:  {context[:80]}{'...' if len(context) > 80 else ''}")
    print(f"  Response: {response[:80]}{'...' if len(response) > 80 else ''}")
    print()

    # Summary
    mean_composite = float(np.mean(composite))
    max_composite = float(np.max(composite))
    print(f"  Mean composite score: {mean_composite:.4f}")
    print(f"  Max  composite score: {max_composite:.4f}")
    print()

    # Threshold interpretation
    if mean_composite > 0.7:
        verdict = "HIGH HALLUCINATION RISK"
    elif mean_composite > 0.5:
        verdict = "MODERATE HALLUCINATION RISK"
    else:
        verdict = "LOW HALLUCINATION RISK (likely faithful)"
    print(f"  Verdict: {verdict}")
    print()

    # Token-level detail
    print(f"  Token-level scores ({n} tokens):")
    print(f"  {'Pos':>4} {'Token':<20} {'Composite':>10} {'IG':>8} {'KL':>8} {'CD':>8} {'Flag':>6}")
    print("  " + "-" * 76)

    flagged = 0
    for i in range(n):
        flag = " !!!" if composite[i] > 0.75 else ""
        if flag:
            flagged += 1
        tok_display = repr(tokens[i])[:18]
        print(f"  {i:>4} {tok_display:<20} {composite[i]:>10.4f} "
              f"{raw_scores['ig'][i]:>8.4f} {raw_scores['kl'][i]:>8.4f} "
              f"{raw_scores['cd'][i]:>8.4f}{flag}")

    print()
    print(f"  Flagged tokens (composite > 0.75): {flagged}/{n}")
    print()

    # Mechanistic interpretation
    print("=" * 70)
    print("  MECHANISTIC INTERPRETATION")
    print("=" * 70)
    print()

    # Find the top-3 most suspicious tokens
    top_idx = np.argsort(composite)[-3:][::-1]
    print("  Top-3 most suspicious tokens:")
    for rank, idx in enumerate(top_idx, 1):
        tok = repr(tokens[idx])
        print(f"    #{rank}: pos={idx} token={tok} composite={composite[idx]:.4f}")
        if raw_scores['ig'][idx] > 0:
            print(f"         IG={raw_scores['ig'][idx]:.4f} -> context FAILED to reduce uncertainty")
        if raw_scores['kl'][idx] > 0.5:
            print(f"         KL={raw_scores['kl'][idx]:.4f} -> large distributional shift from context")
        if raw_scores['cd'][idx] > 0:
            print(f"         CD={raw_scores['cd'][idx]:.4f} -> context REDUCED confidence in this token")
    print()

    # Clean up GPU
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================
# PRE-COMPUTED MODE
# ================================================================
def run_precomputed():
    """Display pre-computed demo results from JSON."""
    print("=" * 70)
    print("  Weighted Composite -- Unsupervised Hallucination Detection Demo")
    print("=" * 70)
    print()
    print("Method:  Dual forward pass (with-context vs without-context)")
    print("Proxy:   Qwen2.5-7B (4-bit NF4 quantized)")
    print("Metrics: Entropy, IG, KL, CD, SelfCheck, SE")
    print("Weights: Variance-based from train distribution (w_i = std_i / sum_std)")
    print("         No labels used ANYWHERE in the pipeline.")
    print()

    if not DEMO_FILE.exists():
        print(f"ERROR: Demo data not found at {DEMO_FILE}")
        print("Run:  python run_final_unsupervised.py  first.")
        sys.exit(1)

    with open(DEMO_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = data["n_tokens"]
    labels = data["labels"]
    types = data["types"]
    composite = data["composite"]
    scores = data["scores"]
    meta = data.get("meta", {})
    weights = data.get("weights", {})

    print(f"Generator:     {meta.get('generator', 'unknown')} (RAGTruth dataset sample)")
    print(f"Proxy model:   Qwen2.5-7B (our detector)")
    print(f"Total tokens:  {n}")
    print(f"Hallucinated:  {sum(labels)}")
    print(f"Faithful:      {n - sum(labels)}")
    print()

    # Display weights
    print("Variance-based weights (w_i = std_i / sum_std, unsupervised):")
    for mk, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 40) if w > 0 else "."
        print(f"  {mk:<12} {w:.4f}  {bar}")
    print()

    # Summary statistics
    hall_scores = [composite[i] for i in range(n) if labels[i] == 1]
    faith_scores = [composite[i] for i in range(n) if labels[i] == 0]

    if hall_scores and faith_scores:
        h_mean = sum(hall_scores) / len(hall_scores)
        f_mean = sum(faith_scores) / len(faith_scores)
        print(f"Mean composite -- Hallucinated: {h_mean:.4f} | Faithful: {f_mean:.4f}")
        print(f"Separation gap: {h_mean - f_mean:+.4f} (positive = method works)")
        print()

    # Per-metric breakdown
    print("Per-metric mean scores (hallucinated vs faithful):")
    print(f"  {'Metric':<12} {'Hallucinated':>14} {'Faithful':>14} {'Gap':>10}")
    print("  " + "-" * 52)
    for mk in ["ig", "kl", "cd", "entropy", "selfcheck"]:
        if mk not in scores:
            continue
        h = [scores[mk][i] for i in range(n) if labels[i] == 1]
        f = [scores[mk][i] for i in range(n) if labels[i] == 0]
        if h and f:
            hm = sum(h) / len(h)
            fm = sum(f) / len(f)
            print(f"  {mk:<12} {hm:>14.4f} {fm:>14.4f} {hm - fm:>+10.4f}")
    print()

    # Token-level detail
    show_n = min(60, n)
    print(f"Token-level scores (first {show_n} of {n} tokens):")
    print(f"  {'Pos':>4} {'Label':>6} {'Type':<15} {'Composite':>10} "
          f"{'IG':>8} {'KL':>8} {'CD':>8}")
    print("  " + "-" * 70)
    for i in range(show_n):
        lbl = "HALL" if labels[i] == 1 else "faith"
        tp = types[i] if labels[i] == 1 else "-"
        marker = "  <<<" if labels[i] == 1 else ""
        print(f"  {i:>4} {lbl:>6} {tp:<15} {composite[i]:>10.4f} "
              f"{scores['ig'][i]:>8.4f} {scores['kl'][i]:>8.4f} "
              f"{scores['cd'][i]:>8.4f}{marker}")
    if n > show_n:
        print(f"  ... ({n - show_n} more tokens)")
    print()

    # Interpretation
    print("=" * 70)
    print("  MECHANISTIC INTERPRETATION")
    print("=" * 70)
    print()
    print("  IG  (Information Gain):   High = context FAILED to reduce uncertainty")
    print("                            -> model ignored retrieved context at this token")
    print("  KL  (KL Divergence):      High = large distributional shift due to context")
    print("                            -> context changed predictions without helping")
    print("  CD  (Confidence Drop):    High = context REDUCED confidence in this token")
    print("                            -> context actively disagrees with generation")
    print()
    print("  Hallucinated tokens (<<<) should show HIGHER composite scores")
    print("  than faithful tokens -- confirming the method works.")
    print()

    # Type-level summary if multiple types present
    type_scores = {}
    for i in range(n):
        if labels[i] == 1:
            t = types[i]
            if t not in type_scores:
                type_scores[t] = []
            type_scores[t].append(composite[i])
    if type_scores:
        print("Per-type composite scores:")
        for t, sc in sorted(type_scores.items()):
            print(f"  {t:<20} mean={sum(sc)/len(sc):.4f}  n={len(sc)}")
        print()


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised Hallucination Detection Demo (CAWC v5)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="The question/query")
    parser.add_argument("--context", "-c", type=str, default=None,
                        help="The retrieved context")
    parser.add_argument("--response", "-r", type=str, default=None,
                        help="The generated response to analyze")
    args = parser.parse_args()

    if args.query and args.context and args.response:
        run_live_inference(args.query, args.context, args.response)
    else:
        run_precomputed()


if __name__ == "__main__":
    main()
