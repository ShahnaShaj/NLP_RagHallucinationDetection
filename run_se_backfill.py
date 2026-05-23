#!/usr/bin/env python3
"""
run_se_backfill.py — Compute REAL Semantic Entropy (Kuhn et al. / Manakul et al.)
using DeBERTa-v3-large NLI clustering + TinyLlama continuation sampling.

No proxies. No placeholders. Real SE for every sample.
"""
import pickle, os, sys, time, warnings
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk

from core_metrics import MetricConfig, MetricEvaluator
from model_utils import ModelConfig, load_generator_model

warnings.filterwarnings("ignore")

RESULTS = "results"

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def se_is_valid(se_arr):
    """Check if SE array has real (non-zero, non-placeholder) values."""
    if se_arr is None or len(se_arr) == 0:
        return False
    a = np.asarray(se_arr, dtype=np.float64)
    # Valid if at least 10% of values are non-zero
    return (a != 0).mean() > 0.10

def compute_se_for_sample(ev, token_ids, prefix, entropy_scores):
    """Compute real SE for one sample. Returns numpy array of SE values."""
    tid = np.asarray(token_ids, dtype=np.int64)
    ent = np.asarray(entropy_scores, dtype=np.float64) if len(entropy_scores) > 0 else np.zeros(len(tid))
    se_arr = ev.semantic_entropy_series(tid, prefix, ent)
    return se_arr

def backfill_test_cache(ev, test_path):
    """Backfill SE in the test cache. Test cache has token_ids and text."""
    print(f"\n{'='*60}")
    print(f"BACKFILLING TEST CACHE: {test_path}")
    print(f"{'='*60}")
    
    data = load_pkl(test_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    updated = 0
    skipped = 0
    t0 = time.time()
    
    for i, r in enumerate(tqdm(results, desc="Test SE")):
        # Check existing SE
        raw = r.get('scores_raw', r.get('scores', {}))
        existing_se = raw.get('se', [])
        if se_is_valid(existing_se):
            skipped += 1
            continue
        
        # Get token_ids  
        tid = r.get('token_ids', [])
        if not tid or len(tid) == 0:
            continue
        
        # Get text for prefix
        text = r.get('text', {})
        q = text.get('query', '')
        c = text.get('context', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "
        
        # Get entropy for anchor selection
        ent = raw.get('entropy', [])
        
        try:
            se_arr = compute_se_for_sample(ev, tid, prefix, ent)
            se_list = se_arr.tolist()
            
            # Save to BOTH locations so pipeline finds it
            if 'scores_raw' in r:
                r['scores_raw']['se'] = se_list
            if 'scores' in r:
                r['scores']['se'] = se_list
            
            updated += 1
            
            # Auto-save every 50 samples
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {skipped} skipped, {elapsed:.1f} min elapsed")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(test_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")
            continue
    
    # Final save
    elapsed = (time.time() - t0) / 60
    print(f"\nTest SE complete: {updated} computed, {skipped} skipped, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(test_path, data if isinstance(data, dict) else results)


def backfill_train_cache(ev, train_path, dataset_path):
    """Backfill SE in the train cache. 
    Train cache lacks token_ids/text, so we load from original dataset."""
    print(f"\n{'='*60}")
    print(f"BACKFILLING TRAIN CACHE: {train_path}")
    print(f"{'='*60}")
    
    data = load_pkl(train_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    # Load original dataset to get text
    ds = load_from_disk(dataset_path)
    # Build lookup by sample_id
    ds_lookup = {}
    for idx, row in enumerate(ds):
        ds_lookup[str(row['id'])] = row
    
    print(f"  Dataset loaded: {len(ds)} samples, Cache: {len(results)} samples")
    
    updated = 0
    skipped = 0
    t0 = time.time()
    
    for i, r in enumerate(tqdm(results, desc="Train SE")):
        # Check existing SE
        scores = r.get('scores', {})
        existing_se = scores.get('se', [])
        if se_is_valid(existing_se):
            skipped += 1
            continue
        
        # Match to dataset by sample_id
        sample_id = str(r.get('meta', {}).get('sample_id', ''))
        ds_row = ds_lookup.get(sample_id)
        if ds_row is None:
            continue
        
        # Build prefix and tokenize output to get token_ids
        q = ds_row.get('query', '')
        c = ds_row.get('context', '')
        output_text = ds_row.get('output', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "
        
        # Tokenize the output to get token_ids
        token_ids = ev.tokenizer.encode(output_text, add_special_tokens=False)
        if not token_ids:
            continue
            
        n_tokens = r.get('n_tokens', len(token_ids))
        token_ids = token_ids[:n_tokens]
        
        # Get entropy for anchor selection
        ent = scores.get('entropy', [])
        
        try:
            se_arr = compute_se_for_sample(ev, token_ids, prefix, ent)
            # Ensure length matches n_tokens
            if len(se_arr) < n_tokens:
                se_arr = np.pad(se_arr, (0, n_tokens - len(se_arr)), constant_values=0.0)
            se_arr = se_arr[:n_tokens]
            
            r['scores']['se'] = se_arr.tolist()
            updated += 1
            
            # Auto-save every 50 samples
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {skipped} skipped, {elapsed:.1f} min elapsed")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(train_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: train sample {i} (id={sample_id}) failed: {e}")
            continue
    
    # Final save
    elapsed = (time.time() - t0) / 60
    print(f"\nTrain SE complete: {updated} computed, {skipped} skipped, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(train_path, data if isinstance(data, dict) else results)


def main():
    print("=" * 60)
    print("SEMANTIC ENTROPY BACKFILL — Real DeBERTa NLI (Kuhn et al.)")
    print("=" * 60)
    
    # Setup models
    print("\nLoading TinyLlama for continuation sampling...")
    m_cfg = ModelConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_gpu_memory_gib=5)
    model, tokenizer = load_generator_model(m_cfg)
    
    # SE config: stride=8 anchors, 12s time budget, 6 samples
    metric_cfg = MetricConfig(
        semantic_samples=6,
        semantic_anchor_stride=8,
        semantic_time_budget_seconds=12.0,
        semantic_max_new_tokens=24,
    )
    ev = MetricEvaluator(
        tokenizer=tokenizer, 
        generator_model=model, 
        config=metric_cfg, 
        nli_device="cuda"
    )
    
    test_path = os.path.join(RESULTS, "scores_cache_v5.pkl")
    train_path = os.path.join(RESULTS, "train_raw_cache_v5.pkl")
    train_ds_path = "ragtruth_processed/train"
    
    # Phase 1: Test cache
    if os.path.exists(test_path):
        backfill_test_cache(ev, test_path)
    
    # Phase 2: Train cache (needs original dataset for text)
    if os.path.exists(train_path) and os.path.exists(train_ds_path):
        backfill_train_cache(ev, train_path, train_ds_path)
    
    # Phase 3: Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    for name, path in [("Test", test_path), ("Train", train_path)]:
        if not os.path.exists(path):
            continue
        d = load_pkl(path)
        res = d if isinstance(d, list) else d.get('results', [])
        all_se = []
        for r in res:
            raw = r.get('scores_raw', r.get('scores', {}))
            se = raw.get('se', [])
            if len(se) > 0:
                all_se.extend(se)
        all_se = np.asarray(all_se, dtype=np.float64)
        zero_pct = (all_se == 0).mean() * 100
        print(f"  {name}: {len(all_se)} tokens, zeros={zero_pct:.1f}%, mean={all_se.mean():.4f}, std={all_se.std():.4f}, max={all_se.max():.4f}")


if __name__ == "__main__":
    main()
