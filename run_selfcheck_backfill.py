#!/usr/bin/env python3
"""
run_selfcheck_backfill.py — Compute REAL SelfCheckGPT-NLI (Manakul et al. 2023)

For each sample:
1. Generate N=3 alternative responses using TinyLlama
2. Split original response into sentences
3. For each sentence, check NLI entailment against each alternative using DeBERTa
4. SelfCheck_NLI(sentence) = 1 - mean(entailment_scores)
5. Map sentence scores to token level
"""
import pickle, os, re, time, warnings
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

def split_into_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def selfcheck_is_valid(sc_arr, n_tokens):
    """Check if selfcheck array has real NLI-based values (not just 1-P(tok))."""
    if sc_arr is None or len(sc_arr) == 0:
        return False
    a = np.asarray(sc_arr, dtype=np.float64)
    # The proxy 1-P(tok) has very specific distribution characteristics.
    # Real NLI-based scores tend to have different variance.
    # For safety, we recompute everything.
    return False  # Always recompute

def compute_selfcheck_nli(ev, tokenizer, gen_model, query, context, response_text, n_alternatives=5):
    """
    Real SelfCheckGPT-NLI (Manakul et al. 2023):
    1. Generate N stochastic alternative responses from the LLM (TinyLlama)
    2. For each sentence in the main response, check if each alternative ENTAILS it
    3. selfcheck_score(sentence) = 1 - mean(entailment_scores_across_alternatives)
       High score = sentence NOT supported by alternatives = potential hallucination
    4. Map sentence scores to token level proportionally by character count
    
    Returns token-level selfcheck scores in [0, 1]. Higher = more likely hallucinated.
    """
    # Tokenize response to get token count
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    n_tokens = len(token_ids)
    if n_tokens == 0:
        return np.zeros(0)
    
    # Step 1: Generate N stochastic alternative responses from the LLM
    # Truncate prompt to avoid OOM on long contexts
    ctx_trunc = context[:600] if context else ""
    prompt = f"Question: {query}\nContext: {ctx_trunc}\nAnswer: "
    device = gen_model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    alternatives = []
    with torch.no_grad():
        try:
            outs = gen_model.generate(
                **inputs,
                max_new_tokens=min(n_tokens + 20, 200),
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=n_alternatives,
            )
            for out in outs:
                gen = out[inputs["input_ids"].shape[1]:]
                alt_text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                if alt_text:
                    alternatives.append(alt_text)
        except RuntimeError:  # OOM fallback — generate one at a time
            for _ in range(n_alternatives):
                try:
                    out = gen_model.generate(
                        **inputs,
                        max_new_tokens=min(n_tokens + 20, 200),
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1,
                    )
                    gen = out[0][inputs["input_ids"].shape[1]:]
                    alt_text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    if alt_text:
                        alternatives.append(alt_text)
                except Exception:
                    break
    
    if not alternatives:
        return np.zeros(n_tokens)
    
    # Step 2: Split main response into sentences
    sentences = split_into_sentences(response_text)
    if not sentences:
        return np.full(n_tokens, 0.5)
    
    # Step 3: For each sentence, check NLI entailment against each alternative
    # Pair format: (alternative_as_premise, response_sentence_as_hypothesis)
    # Logic: if alternative entails the sentence → consistent → confident
    pairs = []
    pair_map = []  # (sent_idx, alt_idx)
    for s_idx, sent in enumerate(sentences):
        for a_idx, alt in enumerate(alternatives):
            pairs.append((alt[:800], sent[:400]))  # truncate for DeBERTa
            pair_map.append((s_idx, a_idx))
    
    if not pairs:
        return np.zeros(n_tokens)
    
    # Batch NLI forward pass through DeBERTa
    entailment_scores = ev._nli_entailment_batch(pairs)
    
    # Step 4: Aggregate per-sentence: 1 - mean(entailment) = inconsistency score
    # Count per sentence how many alternatives we have
    sent_counts = np.zeros(len(sentences))
    sentence_scores = np.zeros(len(sentences))
    for k, (s_idx, a_idx) in enumerate(pair_map):
        sentence_scores[s_idx] += entailment_scores[k]
        sent_counts[s_idx] += 1
    
    # Normalize by actual counts and invert: 1 - mean_entailment = inconsistency
    for s_idx in range(len(sentences)):
        cnt = max(int(sent_counts[s_idx]), 1)
        sentence_scores[s_idx] = 1.0 - (sentence_scores[s_idx] / cnt)
    
    # Step 5: Map sentence scores to token level proportional to character count
    token_scores = np.zeros(n_tokens, dtype=np.float64)
    sent_char_lens = [len(s) for s in sentences]
    total_chars = sum(sent_char_lens)
    if total_chars == 0:
        return np.full(n_tokens, float(np.mean(sentence_scores)))
    
    prev = 0
    cumulative = 0
    for s_idx, slen in enumerate(sent_char_lens):
        cumulative += slen
        boundary = min(int(round(cumulative / total_chars * n_tokens)), n_tokens)
        if s_idx < len(sentence_scores):
            token_scores[prev:boundary] = sentence_scores[s_idx]
        prev = boundary
    
    return token_scores


def backfill_test_selfcheck(ev, tokenizer, gen_model, test_path, test_ds_path="ragtruth_processed/test"):
    """Backfill real SelfCheckGPT-NLI in test cache."""
    print(f"\n{'='*60}")
    print(f"SELFCHECKGPT-NLI BACKFILL — TEST CACHE")
    print(f"{'='*60}")
    
    from datasets import load_from_disk
    data = load_pkl(test_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    # Load dataset for text if not in cache entries
    test_ds = load_from_disk(test_ds_path) if os.path.exists(test_ds_path) else None
    
    updated = 0
    t0 = time.time()
    
    for i, r in enumerate(tqdm(results, desc="Test SelfCheck-NLI")):
        # RESUME FROM 1387
        if i < 1387:
            continue
            
        text = r.get('text', {})
        q = text.get('query', '')
        c = text.get('context', '')
        resp = text.get('response', '')
        
        # Fallback to dataset if text not in cache
        if (not resp) and test_ds is not None and i < len(test_ds):
            ds_row = test_ds[i]
            q = ds_row.get('query', '')
            c = ds_row.get('context', '')
            resp = ds_row.get('output', '')
        
        if not resp:
            continue
        
        n_tokens = len(r.get('token_ids', []))
        if n_tokens == 0:
            n_tokens = r.get('n_tokens', 0)
        if n_tokens == 0:
            continue
        
        try:
            sc_scores = compute_selfcheck_nli(ev, tokenizer, gen_model, q, c, resp)
            
            # Ensure length matches
            if len(sc_scores) < n_tokens:
                sc_scores = np.pad(sc_scores, (0, n_tokens - len(sc_scores)), 
                                   constant_values=float(np.mean(sc_scores)) if len(sc_scores) > 0 else 0.0)
            sc_scores = sc_scores[:n_tokens]
            sc_list = sc_scores.tolist()
            
            # Save to both locations
            if 'scores_raw' in r:
                r['scores_raw']['selfcheck'] = sc_list
            if 'scores' in r:
                r['scores']['selfcheck'] = sc_list
            
            updated += 1
            
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {elapsed:.1f} min")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(test_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")
            continue
    
    elapsed = (time.time() - t0) / 60
    print(f"\nTest SelfCheck complete: {updated} computed, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(test_path, data if isinstance(data, dict) else results)


def backfill_train_selfcheck(ev, tokenizer, gen_model, train_path, dataset_path):
    """Backfill real SelfCheckGPT-NLI in train cache using original dataset text."""
    print(f"\n{'='*60}")
    print(f"SELFCHECKGPT-NLI BACKFILL — TRAIN CACHE")
    print(f"{'='*60}")
    
    data = load_pkl(train_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    ds = load_from_disk(dataset_path)
    ds_lookup = {str(row['id']): row for row in ds}
    
    updated = 0
    t0 = time.time()
    
    for i, r in enumerate(tqdm(results, desc="Train SelfCheck-NLI")):
        sample_id = str(r.get('meta', {}).get('sample_id', ''))
        ds_row = ds_lookup.get(sample_id)
        if ds_row is None:
            continue
        
        q = ds_row.get('query', '')
        c = ds_row.get('context', '')
        resp = ds_row.get('output', '')
        
        if not resp:
            continue
        
        n_tokens = r.get('n_tokens', 0)
        if n_tokens == 0:
            continue
        
        try:
            sc_scores = compute_selfcheck_nli(ev, tokenizer, gen_model, q, c, resp)
            
            if len(sc_scores) < n_tokens:
                sc_scores = np.pad(sc_scores, (0, n_tokens - len(sc_scores)),
                                   constant_values=float(np.mean(sc_scores)) if len(sc_scores) > 0 else 0.0)
            sc_scores = sc_scores[:n_tokens]
            
            r['scores']['selfcheck'] = sc_scores.tolist()
            updated += 1
            
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {elapsed:.1f} min")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(train_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: train sample {i} failed: {e}")
            continue
    
    elapsed = (time.time() - t0) / 60
    print(f"\nTrain SelfCheck complete: {updated} computed, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(train_path, data if isinstance(data, dict) else results)


def main():
    print("=" * 60)
    print("SELFCHECKGPT-NLI BACKFILL (Manakul et al. 2023)")
    print("=" * 60)
    
    print("\nLoading TinyLlama + DeBERTa...")
    m_cfg = ModelConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_gpu_memory_gib=5)
    model, tokenizer = load_generator_model(m_cfg)
    
    metric_cfg = MetricConfig()
    ev = MetricEvaluator(
        tokenizer=tokenizer,
        generator_model=model,
        config=metric_cfg,
        nli_device="cuda"
    )
    # Force-load NLI model now
    ev._ensure_nli()
    
    test_path = os.path.join(RESULTS, "scores_cache_v5.pkl")
    train_path = os.path.join(RESULTS, "train_raw_cache_v5.pkl")
    train_ds_path = "ragtruth_processed/train"
    
    if os.path.exists(test_path):
        backfill_test_selfcheck(ev, tokenizer, model, test_path)
    
    if os.path.exists(train_path) and os.path.exists(train_ds_path):
        backfill_train_selfcheck(ev, tokenizer, model, train_path, train_ds_path)
    
    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    for name, path in [("Test", test_path), ("Train", train_path)]:
        if not os.path.exists(path):
            continue
        d = load_pkl(path)
        res = d if isinstance(d, list) else d.get('results', [])
        all_sc = []
        for r in res:
            raw = r.get('scores_raw', r.get('scores', {}))
            sc = raw.get('selfcheck', [])
            if len(sc) > 0:
                all_sc.extend(sc)
        all_sc = np.asarray(all_sc, dtype=np.float64)
        print(f"  {name}: mean={all_sc.mean():.4f}, std={all_sc.std():.4f}, max={all_sc.max():.4f}")


if __name__ == "__main__":
    main()
