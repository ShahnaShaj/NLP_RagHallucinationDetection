# Track A: Dynamic Uncertainty-Aware Attribution (RAG Hallucination Detection)

This repository contains a modular, purely unsupervised research framework designed to detect hallucinations in Retrieval-Augmented Generation (RAG) tasks using the CAWC v5 (Context-Adaptive Weighted Composite) methodology. 

## 🏆 Final Results
The architecture successfully implemented a completely unsupervised methodology (zero supervised sklearn or LogisticRegression classifiers), yielding robust outcomes:
- **Final CAWC v5 Composite AUROC (RAGTruth):** `0.7510`
- **HaluEval Zero-Shot Transfer AUROC:** `0.6558`
- **Architectural Superiority:** CAWC (`0.7510`) beats its own strongest standalone metric (`0.7268`) and basic Equal-Weight averaging (`0.7119`).
- **SOTA Gap Closed:** `>53.8%` (Compared to state-of-the-art supervised models like LUMINA).

## 📊 Methodology Highlights
- Zero supervised parameters.
- **Variance-Based Weighting** ($w_i \propto \sigma_i$) driven entirely by train population distribution logic (zero hyper-parameter tuning).
- Temporal multi-scale windowing (w=3, 5, 7) to detect uncertainty buildup explicitly at $t-1$ prior to token hallucination.
- **Formal Boundary Conditions:** Formally maps and defends failure cases (when exactly proxy metrics collapse during $P_c \approx P_0$).

---

## 🛠 Setup & Dependencies

**Recommended Hardware:** Minimum 8GB VRAM (e.g., NVIDIA RTX 4060). The pipeline runs large proxy LLMs like **Qwen2.5-7B** in 4-bit NF4 quantized mode.

1. **Clone the repository.**
2. **Create a clean python environment (Python 3.9+ recommended).**
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   **Key dependencies:**
   - `torch`, `transformers`, `bitsandbytes` (for model quantization)
   - `accelerate` (for efficient model loading)
   - `datasets` (for loading preprocessed HF formats)
   - `scikit-learn`, `scipy` (for metrics and significance testing)

4. **Datasets location:**
   Datasets should be pre-downloaded HuggingFace Arrow formats and placed in the project root:
   - `ragtruth_processed/train/`
   - `ragtruth_processed/test/`
   - `halueval_spans/train/`

---

## 📂 File Architecture & Technical Details

- **`run_final_unsupervised.py`** 
  The core script of the assignment representing the peak of the system. Runs the purely unsupervised evaluation on cached outputs. Contains CAWC v5 logic, temporal sliding window normalizations (multi-scale), and runs Experiments 1-8. It outputs results and text analysis files into `results/`.
- **`run_pipeline_v2.py`** 
  The primary data-inference script. Queries generation models like **Qwen2.5-7B** and caches token-level context-differential metrics (Entropy, IG, KL). Stores incremental progress to `results/scores_cache*.pkl`.
- **`core_metrics.py`** 
  Houses the mathematical equations and logic. Evaluates the LLM to yield Information Gain (Context Failure), KL Divergence (Distribution shift), Confidence Drop, Semantic Variance, and typical logit-level Entropy.
- **`model_utils.py`** 
  Responsible for safely loading the designated Large Language Model using 4-bit config (`bitsandbytes`) alongside max-memory allocations mapping across GPUs.
- **`data_utils.py`** 
  Data loading, validation, and token alignment tools. Parses dataset spanning and aligns label mappings exactly against the generated text to evaluate accurately which specific tokens are labelled explicitly as hallucinations.
- **`demo_pipeline.py`** 
  A real-time demo terminal script utilizing unseen `(query, context, response)` triplets to parse out and demonstrate uncertainty visually, loading Qwen2.5 dynamically and tracing token heatmaps representing hallucination likelihoods mechanically.
- **`ablation.py`**
  Comprehensive multi-strategy algorithm verifying the composite bounds against 7 fully unsupervised weighting strategies, Random Perturbation checks ($\pm 50\%$), and Leave-One-Out (LOO) marginal contribution assessments.

---

## 🧪 Experiments & Scientific Robustness Summary

The framework executes 8 key configurations + deep architectural proofs to validate hypotheses scientifically, as seen in `results/final_unsupervised_log.txt` and `ablation_analysis.txt`:

1. **Experiments 1 & 2 (AUROC Assessment & Composite Tuning)**
   - Compares raw baseline (Entropy vs SelfCheckGPT) up against context-sensitive variations (Information Gain = `0.7180`, KL Divergence = `0.6719`, Confidence Drop = `0.7268`).
   - Culminates in Unsupervised CAWC v5 achieving **`0.7510`**.

2. **Proof of Robustness (Ablation Analysis)**
   - Analyzed the CAWC pipeline over 7 unique geometric weighting formulas (Variance, Shannon Entropy, SNR, Consensus, Equal, Theory). Peak AUROCs maintained roughly a `~0.02` spread, empirically proving the CAWC *architecture* provides the discriminative power, not weight tuning.
   - Injecting $\pm 50\%$ random noise securely held AUROCs to `0.7450 ± 0.0039`, verifying adversarial robustness.

3. **Leave-One-Out (Marginal Contribution)**
   - We utilized LOO ablation strictly to assess the **marginal discriminative contribution** of each individual sub-metric. By masking each component from all structural interaction terms, we measured the resultant drop in predictive power. Every singular component removal caused a definitive AUROC drop (e.g. removing KL: $\Delta = -0.0368$), cementing their necessity in the composite.

4. **Experiment 3 (Temporal Precedence)**
   - Confirms Hypothesis 1: Analyzes metrics relative to the hallucinated token at slots `t-3` through `t+1`. It concludes that predictive metric separation robustly peaks physically at **`t-1`**, rather than extending loosely "before" onset. This maps the probability tension exactly inside the preceding token block before fabrication.

5. **Formal Boundary Conditions**
   - Proactively formalized the fundamental $P_c \approx P_0$ proxy mapping limitation. When the context-conditioned distribution functionally aligns to the context-free mapping (the model purely ignored the context internally), context-differential metrics collapse. This formal boundary is rigorously factored and acknowledged in the failure reports.

6. **Experiment 4 (Cross-Domain Zero-Shot)**
   - Zero-shot validation test using `HaluEval` test metrics. Achieved an AUROC hold of `0.6558` against new datasets strictly devoid of retraining weights.

7. **Experiment 5 (Type Breakdown)**
   - Calculates specific types of conflict models: Contradictory, Fabricated, and Unsupported hallucinations. Yields a wide **`>0.22` AUROC gap**. Confirms Hypothesis 2: the model reacts distinctively dependent on *why* text is hallucinated.

8. **Experiment 7 & 8 (Failure Case Analysis & SOTA Gap)**
   - Experiment 7 scopes failure configurations manually observed outside standard margins (deeply reliant on boundary constraints discussed above). Experiment 8 confirms an over +53.8% clearance array against supervised benchmarks.

---

## 🚀 How to Reproduce The Results

To cleanly match the **0.7510 AUROC** referenced in logs:

1. **Stage The Inference Pipeline:**
   This calculates logits and raw token bounds. It creates data caches incrementally. *Note: Running this will overwrite caches and invoke GPU requirements.*
   *(Note: Complete caching drops are already mapped securely inside `results/` for reproducible execution!)*
   ```bash
   python run_pipeline_v2.py --base-dir .
   ```

2. **Execute The Full Unsupervised Production Array:**
   This evaluates cached generations entirely autonomously via variance-weighted calculation arrays and creates summary and individual experiment `.csv` / `.txt` files directly inside `results/`.
   ```bash
   python run_final_unsupervised.py
   ```

3. **Run Live Action Evaluator Test:**
   Runs unseen sequences directly into proxy distribution limits over the variance architecture bounds mechanically.
   ```bash
   python demo_pipeline.py --query "Tell me about the Eiffel Tower" \
     --context "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889." \
     --response "The Eiffel Tower is located in Paris, France. It was built in 1995 by architect Frank Lloyd Wright."
   ```
