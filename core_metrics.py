"""
Token-level uncertainty metrics and composite scoring for RAG hallucination analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EPS = 1e-10


@dataclass
class MetricConfig:
    semantic_samples: int = 6
    semantic_max_new_tokens: int = 24
    semantic_temperature: float = 0.8
    semantic_entailment_threshold: float = 0.7
    entropy_percentile_for_semantic: float = 70.0
    # 0 means unlimited anchor positions.
    semantic_max_positions: int = 0
    semantic_anchor_stride: int = 3
    semantic_time_budget_seconds: float = 25.0
    semantic_text_char_limit: int = 500
    default_metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "entropy": 0.15,
            "ig": 0.20,
            "kl": 0.25,
            "cd": 0.15,
            "se": 0.20,
            "selfcheck": 0.05,
        }
    )


class MetricEvaluator:
    """Computes token-level metrics and dynamic composite scores."""

    def __init__(
        self,
        tokenizer,
        generator_model=None,
        config: Optional[MetricConfig] = None,
        nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        nli_device: str = "auto",
    ) -> None:
        self.tokenizer = tokenizer
        self.generator_model = generator_model
        self.config = config or MetricConfig()
        self.nli_model_name = nli_model_name
        self.nli_device = "cuda" if nli_device == "auto" and torch.cuda.is_available() else nli_device

        self._nli_tokenizer = None
        self._nli_model = None
        self._entailment_idx = None

    @staticmethod
    def entropy(prob: np.ndarray) -> float:
        prob = np.clip(prob, EPS, 1.0)
        return float(-np.sum(prob * np.log(prob)))

    @classmethod
    def information_gain(cls, prob_with: np.ndarray, prob_without: np.ndarray) -> float:
        # Low IG indicates context failed to reduce uncertainty.
        return float(cls.entropy(prob_without) - cls.entropy(prob_with))

    @staticmethod
    def kl_divergence(prob_with: np.ndarray, prob_without: np.ndarray) -> float:
        pw = np.clip(prob_with, EPS, 1.0)
        p0 = np.clip(prob_without, EPS, 1.0)
        return float(np.sum(pw * np.log(pw / p0)))

    @staticmethod
    def confidence_drop(prob_with: np.ndarray, prob_without: np.ndarray, token_id: int) -> float:
        # Token-level proxy of P(True) drop when context is introduced.
        token_id = int(token_id)
        return float(prob_without[token_id] - prob_with[token_id])

    @staticmethod
    def selfcheck_score(prob_with: np.ndarray, token_id: int) -> float:
        token_id = int(token_id)
        return float(1.0 - prob_with[token_id])

    @staticmethod
    def context_utilization(prob_with: np.ndarray, prob_without: np.ndarray) -> float:
        ratio = float(np.mean(prob_with) / (np.mean(prob_without) + EPS))
        return float(1.0 / (1.0 + np.exp(-(ratio - 1.0))))

    def token_metrics(
        self,
        prob_with: np.ndarray,
        prob_without: np.ndarray,
        token_id: int,
    ) -> Dict[str, float]:
        return {
            "entropy": self.entropy(prob_with),
            "ig": self.information_gain(prob_with, prob_without),
            "kl": self.kl_divergence(prob_with, prob_without),
            "cd": self.confidence_drop(prob_with, prob_without, token_id),
            "selfcheck": self.selfcheck_score(prob_with, token_id),
            "u": self.context_utilization(prob_with, prob_without),
        }

    @staticmethod
    def minmax_scale(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            return np.zeros_like(values, dtype=np.float64)
        return (values - vmin) / (vmax - vmin + EPS)

    def _ensure_nli(self) -> None:
        if self._nli_model is not None:
            return

        self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        nli_dtype = torch.float16 if str(self.nli_device).startswith("cuda") else torch.float32
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name,
            torch_dtype=nli_dtype,
        )
        self._nli_model.eval()
        self._nli_model.to(self.nli_device)

        label_to_id = {k.lower(): v for k, v in self._nli_model.config.label2id.items()}
        entailment_keys = [k for k in label_to_id if "entail" in k]
        if not entailment_keys:
            raise RuntimeError(f"No entailment label found in NLI model labels: {label_to_id}")
        self._entailment_idx = label_to_id[entailment_keys[0]]

    def _nli_entailment(self, premise: str, hypothesis: str) -> float:
        self._ensure_nli()
        premise = premise[: self.config.semantic_text_char_limit]
        hypothesis = hypothesis[: self.config.semantic_text_char_limit]
        batch = self._nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        batch = {k: v.to(self.nli_device) for k, v in batch.items()}
        use_amp = str(self.nli_device).startswith("cuda")
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self._nli_model(**batch).logits[0]
            else:
                logits = self._nli_model(**batch).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return float(probs[self._entailment_idx].detach().cpu().item())

    def _nli_entailment_batch(self, pairs: List[tuple]) -> List[float]:
        """Batch NLI: process all (premise, hypothesis) pairs in one forward pass."""
        self._ensure_nli()
        if not pairs:
            return []
        premises = [p[: self.config.semantic_text_char_limit] for p, _ in pairs]
        hypotheses = [h[: self.config.semantic_text_char_limit] for _, h in pairs]
        batch = self._nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=True,
        )
        batch = {k: v.to(self.nli_device) for k, v in batch.items()}
        use_amp = str(self.nli_device).startswith("cuda")
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self._nli_model(**batch).logits
            else:
                logits = self._nli_model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
        return [float(probs[i, self._entailment_idx].item()) for i in range(len(pairs))]

    @staticmethod
    def _cluster_entropy_from_adjacency(adj: np.ndarray) -> float:
        n = adj.shape[0]
        visited = [False] * n
        cluster_sizes: List[int] = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                neighbors = np.where(adj[node])[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(int(nb))
            cluster_sizes.append(size)

        probs = np.array(cluster_sizes, dtype=np.float64) / max(sum(cluster_sizes), 1)
        probs = np.clip(probs, EPS, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def semantic_entropy_from_samples(self, samples: List[str]) -> float:
        """Compute semantic entropy via DeBERTa NLI clustering (Kuhn et al.).
        Uses batched NLI for all pairs in a single forward pass."""
        if len(samples) <= 1:
            return 0.0

        n = len(samples)
        adj = np.eye(n, dtype=bool)
        threshold = self.config.semantic_entailment_threshold

        # Build all (i,j) and (j,i) pairs for batched NLI
        pairs_ij = []
        pair_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_ij.append((samples[i], samples[j]))
                pairs_ij.append((samples[j], samples[i]))
                pair_indices.append((i, j))

        # Single batched forward pass through DeBERTa
        scores = self._nli_entailment_batch(pairs_ij)

        # Parse results: scores come in (i->j, j->i) pairs
        for k, (i, j) in enumerate(pair_indices):
            e_ij = scores[2 * k]
            e_ji = scores[2 * k + 1]
            if min(e_ij, e_ji) >= threshold:
                adj[i, j] = True
                adj[j, i] = True

        return self._cluster_entropy_from_adjacency(adj)

    def sample_local_continuations(self, prefix_text: str) -> List[str]:
        if self.generator_model is None or self.tokenizer is None:
            return []

        device = self.generator_model.get_input_embeddings().weight.device
        inputs = self.tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sampled = []

        with torch.no_grad():
            # Batch generation: 2 rounds of 3 to stay within 8GB VRAM
            for _ in range(2):
                outs = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=self.config.semantic_max_new_tokens,
                    do_sample=True,
                    temperature=self.config.semantic_temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=min(3, self.config.semantic_samples),
                )
                for out in outs:
                    gen = out[inputs["input_ids"].shape[1] :]
                    sampled.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
                if len(sampled) >= self.config.semantic_samples:
                    break
            sampled = sampled[:self.config.semantic_samples]

        return sampled

    def semantic_entropy_series(
        self,
        response_token_ids: np.ndarray,
        base_prompt: str,
        entropy_scores: np.ndarray,
    ) -> np.ndarray:
        n = int(len(response_token_ids))
        se = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return se

        # Build anchor positions for local continuation sampling.
        # We evaluate SE on anchors and interpolate to get dense token-level values.
        stride = max(int(self.config.semantic_anchor_stride), 1)
        anchors = list(range(stride - 1, n, stride))
        if not anchors or anchors[-1] != n - 1:
            anchors.append(n - 1)

        # Add high-entropy anchors to improve local sensitivity at uncertain regions.
        if len(entropy_scores) == n and np.isfinite(entropy_scores).any():
            q = np.nanpercentile(entropy_scores, self.config.entropy_percentile_for_semantic)
            hi = np.where(entropy_scores >= q)[0].tolist()
            # Keep every 3rd high-entropy token to control runtime.
            anchors.extend(hi[::3])

        anchors = sorted(set(int(a) for a in anchors if 0 <= a < n))
        if self.config.semantic_max_positions and self.config.semantic_max_positions > 0:
            if len(anchors) > self.config.semantic_max_positions:
                sel = np.linspace(0, len(anchors) - 1, self.config.semantic_max_positions, dtype=int)
                anchors = [anchors[i] for i in sel]
                if anchors[-1] != n - 1:
                    anchors[-1] = n - 1

        start_ts = time.time()
        for anchor in anchors:
            if (time.time() - start_ts) > self.config.semantic_time_budget_seconds:
                break

            # Generate prefix up to current anchor token.
            end_idx = int(anchor) + 1
            prefix = self.tokenizer.decode(response_token_ids[:end_idx], skip_special_tokens=True)
            prompt = f"{base_prompt}{prefix}"

            try:
                samples = self.sample_local_continuations(prompt)
                if samples:
                    se[anchor] = self.semantic_entropy_from_samples(samples)
            except Exception:
                # Keep NaN if local SE computation failed.
                continue

        known = np.where(np.isfinite(se))[0]
        if len(known) == 0:
            return se

        if len(known) == 1:
            se[:] = float(se[known[0]])
            return se

        # Dense interpolation from anchor estimates.
        full_x = np.arange(n, dtype=np.int32)
        se = np.interp(full_x, known, se[known]).astype(np.float64)

        return se

    def dynamic_composite(
        self,
        score_arrays: Dict[str, np.ndarray],
        context_utilization: np.ndarray,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        metric_weights = metric_weights or self.config.default_metric_weights
        n = len(context_utilization)
        out = np.zeros(n, dtype=np.float64)

        for i in range(n):
            u = float(np.clip(context_utilization[i], 0.0, 1.0))
            dyn_weights = {
                "ig": metric_weights.get("ig", 0.0) * u,
                "kl": metric_weights.get("kl", 0.0) * (0.5 + 0.5 * u),
                "entropy": metric_weights.get("entropy", 0.0) * (1.0 - u),
                "cd": metric_weights.get("cd", 0.0) * (1.0 - 0.5 * u),
                "se": metric_weights.get("se", 0.0),
                "selfcheck": metric_weights.get("selfcheck", 0.0) * (1.0 - u),
            }
            z = sum(dyn_weights.values()) + EPS
            out[i] = (
                dyn_weights["ig"] * score_arrays["ig"][i]
                + dyn_weights["kl"] * score_arrays["kl"][i]
                + dyn_weights["entropy"] * score_arrays["entropy"][i]
                + dyn_weights["cd"] * score_arrays["cd"][i]
                + dyn_weights["se"] * score_arrays["se"][i]
                + dyn_weights["selfcheck"] * score_arrays["selfcheck"][i]
            ) / z

        return out
