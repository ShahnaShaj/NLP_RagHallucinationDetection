import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path(r"c:\Users\LOQ\Desktop\Track A Midsem NLP\NLP Midsem Code")
    cache_path = base_dir / "results" / "scores_cache_v5.pkl"
    
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
        
    results = data.get("results", data) if isinstance(data, dict) else data
    
    mkeys = ["composite", "ig", "kl", "entropy", "cd"]
    positions = [-3, -2, -1, 0, 1]
    pos_labels = ["t-3", "t-2", "t-1", "t", "t+1"]
    
    # 1. Gather all values to compute global mean and std for standardizing
    global_vals = {mk: [] for mk in mkeys}
    for r in results:
        n = len(r.get("labels", []))
        if n < 2: continue
        
        # We need composite as well. It's precomputed in the cache?
        # Actually the cache from process_dataset has "composite".
        if "composite" in r:
            global_vals["composite"].extend(r["composite"][:n])
        
        raw_scores = r.get("scores_raw", {})
        for mk in ["ig", "kl", "entropy", "cd"]:
            if mk in raw_scores:
                global_vals[mk].extend(raw_scores[mk][:n])

    stats = {}
    for mk in mkeys:
        arr = np.array(global_vals[mk], dtype=float)
        arr = arr[np.isfinite(arr)]
        stats[mk] = {"mean": np.mean(arr), "std": np.std(arr) + 1e-10}
        
    # 2. Gather standardized values at relative positions
    bucket = {mk: {p: [] for p in positions} for mk in mkeys}
    
    for r in results:
        labels = r.get("labels", [])
        n = len(labels)
        hidx = np.where(np.array(labels) == 1)[0]
        if len(hidx) == 0: continue
        
        first_hall = int(hidx[0])
        
        raw_scores = r.get("scores_raw", {})
        comp_scores = r.get("composite", [])
        
        for p in positions:
            idx = first_hall + p
            if 0 <= idx < n:
                # Composite
                if idx < len(comp_scores):
                    val = comp_scores[idx]
                    z = (val - stats["composite"]["mean"]) / stats["composite"]["std"]
                    bucket["composite"][p].append(z)
                
                # Others
                for mk in ["ig", "kl", "entropy", "cd"]:
                    if mk in raw_scores and idx < len(raw_scores[mk]):
                        val = raw_scores[mk][idx]
                        z = (val - stats[mk]["mean"]) / stats[mk]["std"]
                        bucket[mk][p].append(z)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = {"composite": "#E63946", "ig": "#457B9D", "kl": "#2A9D8F", "entropy": "#E9C46A", "cd": "#264653"}
    labels_dict = {"composite": "CAWC Composite", "ig": "Info Gain", "kl": "KL Divergence", "entropy": "Token Entropy", "cd": "Context Diff"}
    
    for mk in mkeys:
        means = []
        stds = []
        for p in positions:
            vals = np.array(bucket[mk][p], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
                
        means = np.array(means)
        stds = np.array(stds)
        
        # Save raw means to global dict for CSV
        for i, p in enumerate(positions):
            raw_means_for_csv[mk][p] = means[i]
            
        mn = np.min(means)
        mx = np.max(means)
        if mx > mn:
            means_norm = (means - mn) / (mx - mn)
            stds_norm = stds / (mx - mn) # scale the standard deviation proportionally
        else:
            means_norm = means
            stds_norm = stds
            
        x = np.arange(len(positions))
        ax.plot(x, means_norm, marker='o', label=labels_dict[mk], color=colors[mk], linewidth=2.5, markersize=8)
        ax.fill_between(x, means_norm - stds_norm, means_norm + stds_norm, color=colors[mk], alpha=0.15)
        
    ax.axvline(x=3, color="black", linestyle="--", alpha=0.8, linewidth=2, label="Hallucination Onset (t)")
    
    ax.set_xticks(np.arange(len(positions)))
    ax.set_xticklabels(pos_labels)
    ax.set_xlabel("Position relative to hallucinated token", fontsize=14, fontweight='bold')
    ax.set_ylabel("Normalized Signal Score (\u00b11 s.d.)", fontsize=14, fontweight='bold')
    ax.set_title("Figure 2: Mean signal score at positions t\u22123 to t+1", fontsize=16, pad=15)
    
    # Improve legend and grid
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = base_dir / "results" / "figure_2_temporal_bands.png"
    plt.savefig(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
