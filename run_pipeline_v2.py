"""
Track A pipeline: Dynamic Uncertainty-Aware Attribution for hallucination detection.
"""
from __future__ import annotations

import atexit
import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from core_metrics import MetricConfig, MetricEvaluator
from data_utils import (
    align_spans_to_tokens,
    extract_halueval_fields,
    extract_ragtruth_fields,
    inspect_datasets,
    load_halueval_train,
    load_ragtruth_splits,
    parse_halueval_labels,
    parse_ragtruth_labels,
)
from model_utils import ModelConfig, dual_forward_pass, load_generator_model


@dataclass
class AnalyzerConfig:
    base_dir: str
    results_dir: str = "results"
    cache_every: int = 50
    checkpoint_seconds: int = 120
    direction_samples: int = 100
    train_weight_samples: int = 200
    rag_cache_name: str = "scores_cache_v5.pkl"
    halu_cache_name: str = "halueval_scores_cache_v5.pkl"
    direction_file: str = "metric_directions_v4.json"
    weights_file: str = "composite_weights_v4.json"
    inspection_file: str = "dataset_inspection.txt"
    semantic_stride: int = 8
    compute_semantic_for_calibration: bool = False
    compute_semantic_for_halueval: bool = False
    quality_se_mode: bool = False
    force_recompute: bool = False
    progress_every: int = 5
    progress_seconds: int = 30
    run_lock_file: str = "pipeline_v4.lock"


class RAGAnalyzer:
    """End-to-end analyzer for Track A experiments."""

    def __init__(
        self,
        analyzer_cfg: AnalyzerConfig,
        model_cfg: Optional[ModelConfig] = None,
        metric_cfg: Optional[MetricConfig] = None,
    ) -> None:
        self.cfg = analyzer_cfg
        self.model_cfg = model_cfg or ModelConfig()
        self.metric_cfg = metric_cfg or MetricConfig()

        self.base_dir = Path(self.cfg.base_dir)
        self.results_dir = self.base_dir / self.cfg.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.metric_evaluator: Optional[MetricEvaluator] = None

        self.metric_keys = ["entropy", "ig", "kl", "cd", "se", "selfcheck"]
        self.direction_flags: Dict[str, bool] = {k: False for k in self.metric_keys}
        self.metric_weights = dict(self.metric_cfg.default_metric_weights)

        self._lock_file_path = self.results_dir / self.cfg.run_lock_file
        self._lock_active = False

        if self.cfg.quality_se_mode:
            # Dense semantic entropy across all splits for high-quality SE caches.
            self.cfg.semantic_stride = 1
            self.cfg.compute_semantic_for_calibration = True
            self.cfg.compute_semantic_for_halueval = True

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False

    def _release_run_lock(self) -> None:
        if not self._lock_active:
            return
        try:
            if self._lock_file_path.exists():
                self._lock_file_path.unlink()
        finally:
            self._lock_active = False

    def _acquire_run_lock(self) -> None:
        if self._lock_file_path.exists():
            try:
                existing = json.loads(self._lock_file_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

            other_pid = int(existing.get("pid", -1))
            if self._pid_is_alive(other_pid):
                started = existing.get("started_at", "unknown")
                raise RuntimeError(
                    f"Another pipeline run is active (pid={other_pid}, started_at={started}). "
                    "Stop it first or remove stale lock file."
                )
            try:
                self._lock_file_path.unlink()
            except OSError:
                pass

        payload = {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_dir": str(self.base_dir),
        }
        self._lock_file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._lock_active = True
        atexit.register(self._release_run_lock)

    @staticmethod
    def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        mask = np.isfinite(scores)
        labels = labels[mask]
        scores = scores[mask]
        if labels.size == 0 or len(np.unique(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, scores))

    def setup_models(self) -> None:
        self.model, self.tokenizer = load_generator_model(self.model_cfg)
        model_device = self.model.get_input_embeddings().weight.device
        if self.model_cfg.enforce_cuda and model_device.type != "cuda":
            raise RuntimeError(f"Generator model expected on CUDA but loaded on {model_device}.")

        nli_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric_evaluator = MetricEvaluator(
            tokenizer=self.tokenizer,
            generator_model=self.model,
            config=self.metric_cfg,
            nli_device=nli_device,
        )
        tqdm.write(f"[setup] generator_device={model_device}, nli_device={nli_device}")

    def run_dataset_inspection(self) -> Path:
        out_path = self.results_dir / self.cfg.inspection_file
        inspect_datasets(str(self.base_dir), str(out_path))
        return out_path

    def _build_prompt_prefix(self, query: str, context: str) -> str:
        return f"Question: {query}\nContext: {context}\nAnswer: "

    def _extract_sample_parts(self, sample: Dict, dataset_name: str):
        if dataset_name.startswith("ragtruth"):
            query, context, response = extract_ragtruth_fields(sample)
            spans = parse_ragtruth_labels(sample)
            meta = {
                "sample_id": str(sample.get("id", "")),
                "task_type": str(sample.get("task_type", "unknown")),
                "generator": str(sample.get("model", "unknown")),
                "quality": str(sample.get("quality", "unknown")),
            }
            return query, context, response, spans, meta

        if dataset_name.startswith("halueval"):
            query, context, response = extract_halueval_fields(sample)
            spans = parse_halueval_labels(sample)
            meta = {
                "sample_id": "",
                "task_type": str(sample.get("task_type", "unknown")),
                "generator": str(sample.get("dataset", "halueval")),
                "quality": "unknown",
            }
            return query, context, response, spans, meta

        query = str(sample.get("query", "")).strip()
        context = str(sample.get("context", "")).strip()
        response = str(sample.get("response", "")).strip()
        spans = sample.get("spans", [])
        meta = {
            "sample_id": str(sample.get("id", "custom")),
            "task_type": str(sample.get("task_type", "custom")),
            "generator": str(sample.get("model", "custom")),
            "quality": "unknown",
        }
        return query, context, response, spans, meta

    def _score_sample(
        self,
        sample: Dict,
        dataset_name: str,
        compute_semantic: bool,
        use_direction: bool,
    ) -> Optional[Dict]:
        if self.metric_evaluator is None:
            raise RuntimeError("Models are not initialized. Call setup_models() first.")

        query, context, response, spans, meta = self._extract_sample_parts(sample, dataset_name)
        if not response or not context:
            return None

        aligned_token_ids, labels, types = align_spans_to_tokens(self.tokenizer, response, spans)
        p_with, p_without, response_ids = dual_forward_pass(
            model=self.model,
            tokenizer=self.tokenizer,
            query=query,
            context=context,
            response=response,
            cfg=self.model_cfg,
        )

        n = min(len(response_ids), len(labels), len(p_with), len(p_without), len(aligned_token_ids))
        if n < 2:
            return None

        raw_scores: Dict[str, List[float]] = {k: [] for k in ["entropy", "ig", "kl", "cd", "selfcheck", "u"]}
        for i in range(n):
            m = self.metric_evaluator.token_metrics(
                prob_with=p_with[i],
                prob_without=p_without[i],
                token_id=int(response_ids[i]),
            )
            for key in raw_scores:
                raw_scores[key].append(m[key])

        entropy_arr = np.asarray(raw_scores["entropy"], dtype=np.float64)
        se_arr = np.full(n, np.nan, dtype=np.float64)
        if compute_semantic:
            prefix = self._build_prompt_prefix(query=query, context=context)
            se_arr = self.metric_evaluator.semantic_entropy_series(
                response_token_ids=np.asarray(response_ids[:n], dtype=np.int64),
                base_prompt=prefix,
                entropy_scores=entropy_arr,
            )

        raw_np = {
            "entropy": entropy_arr,
            "ig": np.asarray(raw_scores["ig"], dtype=np.float64),
            "kl": np.asarray(raw_scores["kl"], dtype=np.float64),
            "cd": np.asarray(raw_scores["cd"], dtype=np.float64),
            "se": np.asarray(se_arr, dtype=np.float64),
            "selfcheck": np.asarray(raw_scores["selfcheck"], dtype=np.float64),
            "u": np.asarray(raw_scores["u"], dtype=np.float64),
        }
        meta = dict(meta)
        meta["se_coverage"] = float(np.isfinite(raw_np["se"]).mean())

        if not use_direction:
            return {
                "labels": labels[:n].astype(int).tolist(),
                "types": types[:n].tolist(),
                "token_ids": np.asarray(response_ids[:n], dtype=np.int64).tolist(),
                "raw_scores": {k: raw_np[k].tolist() for k in self.metric_keys},
                "u": raw_np["u"].tolist(),
                "meta": meta,
            }

        directed = {}
        scaled = {}
        for key in self.metric_keys:
            arr = raw_np[key].copy()
            if self.direction_flags.get(key, False):
                arr = -arr
            directed[key] = arr

            finite = np.isfinite(arr)
            if not finite.any():
                # Neutral signal when metric is missing (not a proxy).
                scaled[key] = np.zeros_like(arr, dtype=np.float64)
            else:
                arr_for_scale = arr.copy()
                if not finite.all():
                    arr_for_scale[~finite] = float(np.nanmedian(arr_for_scale[finite]))
                scaled[key] = MetricEvaluator.minmax_scale(arr_for_scale)

        composite = self.metric_evaluator.dynamic_composite(
            score_arrays=scaled,
            context_utilization=raw_np["u"],
            metric_weights=self.metric_weights,
        )

        token_text = self.tokenizer.convert_ids_to_tokens(np.asarray(response_ids[:n], dtype=np.int64).tolist())

        return {
            "dataset": dataset_name,
            "meta": meta,
            "labels": labels[:n].astype(int).tolist(),
            "types": types[:n].tolist(),
            "token_ids": np.asarray(response_ids[:n], dtype=np.int64).tolist(),
            "tokens": token_text,
            "scores": {k: scaled[k].tolist() for k in self.metric_keys},
            "scores_raw": {k: directed[k].tolist() for k in self.metric_keys},
            "scores_raw_undirected": {k: raw_np[k].tolist() for k in self.metric_keys},
            "context_utilization": raw_np["u"].tolist(),
            "composite": composite.tolist(),
            "text": {
                "query": query,
                "context": context,
                "response": response,
            },
        }

    def _load_cache(self, cache_path: Path):
        if not cache_path.exists():
            return {"results": [], "processed_indices": []}

        with open(cache_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, list):
            return {"results": payload, "processed_indices": list(range(len(payload)))}

        payload.setdefault("results", [])
        payload.setdefault("processed_indices", [])
        return payload

    def _save_cache(self, cache_path: Path, payload: Dict) -> None:
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)

    def _emit_progress(self, dataset_name: str, processed: int, total: int, kept: int, start_ts: float) -> None:
        elapsed = max(time.time() - start_ts, 1e-6)
        rate = processed / elapsed
        remaining = max(total - processed, 0)
        eta = remaining / max(rate, 1e-6)
        msg = (
            f"[{dataset_name}] processed={processed}/{total}, kept={kept}, "
            f"rate={rate:.2f} samples/s, eta={eta/60.0:.1f}m"
        )
        tqdm.write(msg)

    def run_metric_direction_test(self, calibration_ds) -> Dict[str, Dict[str, float]]:
        details = {}
        score_store = {k: [] for k in self.metric_keys}
        labels_store = []

        used = 0
        for i, sample in enumerate(tqdm(calibration_ds, desc="Direction test", total=min(self.cfg.direction_samples, len(calibration_ds)))):
            if used >= self.cfg.direction_samples:
                break
            out = self._score_sample(
                sample=sample,
                dataset_name="ragtruth_train",
                # Keep calibration lightweight; semantic entropy remains enabled in full scoring.
                compute_semantic=self.cfg.compute_semantic_for_calibration,
                use_direction=False,
            )
            if out is None:
                continue

            labels_store.extend(out["labels"])
            for key in self.metric_keys:
                score_store[key].extend(out["raw_scores"][key])
            used += 1

        labels = np.asarray(labels_store, dtype=np.int32)

        for key in self.metric_keys:
            vals = np.asarray(score_store[key], dtype=np.float64)
            raw_auc = self._safe_auroc(labels, vals)
            neg_auc = self._safe_auroc(labels, -vals)
            should_negate = neg_auc > raw_auc

            # Critical direction fix requirement.
            if key in {"ig", "cd"}:
                should_negate = True

            self.direction_flags[key] = bool(should_negate)
            chosen_auc = neg_auc if should_negate else raw_auc
            details[key] = {
                "raw_auroc": raw_auc,
                "negated_auroc": neg_auc,
                "negate": bool(should_negate),
                "chosen_auroc": float(chosen_auc),
            }

        out_path = self.results_dir / self.cfg.direction_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)
        return details

    def calibrate_unsupervised_weights(self, train_ds) -> Dict[str, float]:
        pool = {k: [] for k in self.metric_keys}
        used = 0

        for i, sample in enumerate(tqdm(train_ds, desc="Unsupervised weight calibration", total=min(self.cfg.train_weight_samples, len(train_ds)))):
            if used >= self.cfg.train_weight_samples:
                break

            out = self._score_sample(
                sample=sample,
                dataset_name="ragtruth_train",
                # Weight calibration uses robust distribution spreads from fast metrics.
                compute_semantic=self.cfg.compute_semantic_for_calibration,
                use_direction=True,
            )
            if out is None:
                continue

            for key in self.metric_keys:
                pool[key].extend(out["scores"][key])
            used += 1

        spreads = {}
        for key in self.metric_keys:
            arr = np.asarray(pool[key], dtype=np.float64)
            if arr.size == 0:
                spreads[key] = 1e-3
            else:
                q75 = np.percentile(arr, 75)
                q25 = np.percentile(arr, 25)
                spreads[key] = max(float(q75 - q25), 1e-3)

        total = sum(spreads.values())
        self.metric_weights = {k: v / total for k, v in spreads.items()}

        out_path = self.results_dir / self.cfg.weights_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_weights, f, indent=2)
        return self.metric_weights

    def process_dataset(self, ds, dataset_name: str, cache_name: str, compute_semantic: bool) -> List[Dict]:
        cache_path = self.results_dir / cache_name
        cache_payload = self._load_cache(cache_path)

        if self.cfg.force_recompute:
            results = []
            processed = set()
        else:
            results = cache_payload.get("results", [])
            processed = set(cache_payload.get("processed_indices", []))

        newly_processed = 0
        last_checkpoint_ts = time.time()
        last_log_ts = time.time()
        start_ts = time.time()
        total = len(ds)

        for idx in tqdm(range(len(ds)), desc=f"Processing {dataset_name}"):
            if idx in processed:
                continue

            sample = ds[idx]
            semantic_for_sample = False
            if compute_semantic:
                if self.cfg.semantic_stride <= 1:
                    semantic_for_sample = True
                else:
                    semantic_for_sample = (idx % self.cfg.semantic_stride == 0)

            try:
                out = self._score_sample(
                    sample=sample,
                    dataset_name=dataset_name,
                    compute_semantic=semantic_for_sample,
                    use_direction=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                tqdm.write(f"[{dataset_name}] sample idx={idx} failed: {exc}")
                out = None

            if out is not None:
                results.append(out)

            processed.add(idx)
            newly_processed += 1
            processed_count = len(processed)

            if (
                (newly_processed % max(self.cfg.progress_every, 1) == 0)
                or (time.time() - last_log_ts >= self.cfg.progress_seconds)
            ):
                self._emit_progress(
                    dataset_name=dataset_name,
                    processed=processed_count,
                    total=total,
                    kept=len(results),
                    start_ts=start_ts,
                )
                last_log_ts = time.time()

            if (
                newly_processed >= self.cfg.cache_every
                or (time.time() - last_checkpoint_ts) >= self.cfg.checkpoint_seconds
            ):
                cache_payload = {
                    "results": results,
                    "processed_indices": sorted(processed),
                    "direction_flags": self.direction_flags,
                    "metric_weights": self.metric_weights,
                    "updated_at": time.time(),
                }
                self._save_cache(cache_path, cache_payload)
                tqdm.write(
                    f"[{dataset_name}] checkpoint saved: kept={len(results)}, processed={len(processed)}"
                )
                newly_processed = 0
                last_checkpoint_ts = time.time()

        cache_payload = {
            "results": results,
            "processed_indices": sorted(processed),
            "direction_flags": self.direction_flags,
            "metric_weights": self.metric_weights,
            "updated_at": time.time(),
        }
        self._save_cache(cache_path, cache_payload)
        self._emit_progress(
            dataset_name=dataset_name,
            processed=len(processed),
            total=total,
            kept=len(results),
            start_ts=start_ts,
        )
        return results

    def score_triplet(self, query: str, context: str, response: str) -> Dict:
        sample = {
            "query": query,
            "context": context,
            "response": response,
            "spans": [],
            "id": "demo",
            "task_type": "demo",
            "model": "qwen2.5-7b",
        }
        out = self._score_sample(
            sample=sample,
            dataset_name="custom_demo",
            compute_semantic=True,
            use_direction=True,
        )
        if out is None:
            raise RuntimeError("Unable to score provided triplet. Check text length and content.")
        return out

    def run(self) -> Dict[str, int]:
        self._acquire_run_lock()
        try:
            self.run_dataset_inspection()
            self.setup_models()

            rag_train, rag_test = load_ragtruth_splits(str(self.base_dir))
            halu_train = load_halueval_train(str(self.base_dir))

            self.run_metric_direction_test(rag_train)
            self.calibrate_unsupervised_weights(rag_train)

            # Generate train cache for distribution thresholding
            train_results = self.process_dataset(
                ds=rag_train,
                dataset_name="ragtruth_train",
                cache_name="train_raw_cache_v5.pkl",
                compute_semantic=False, # We backfill this later
            )

            rag_results = self.process_dataset(
                ds=rag_test,
                dataset_name="ragtruth_test",
                cache_name=self.cfg.rag_cache_name,
                compute_semantic=True,
            )
            halu_results = self.process_dataset(
                ds=halu_train,
                dataset_name="halueval_train",
                cache_name=self.cfg.halu_cache_name,
                compute_semantic=self.cfg.compute_semantic_for_halueval,
            )

            summary = {
                "ragtruth_samples_scored": len(rag_results),
                "halueval_samples_scored": len(halu_results),
            }
            with open(self.results_dir / "pipeline_run_summary_v4.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return summary
        finally:
            self._release_run_lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track A Dynamic Uncertainty-Aware Attribution pipeline.")
    parser.add_argument("--base-dir", type=str, default=".", help="Project root directory")
    parser.add_argument("--direction-samples", type=int, default=100, help="Samples for metric direction validation")
    parser.add_argument("--train-weight-samples", type=int, default=200, help="Samples for unsupervised weight calibration")
    parser.add_argument("--semantic-stride", type=int, default=8, help="Compute semantic entropy every N samples")
    parser.add_argument("--semantic-for-halueval", action="store_true", help="Compute semantic entropy for HaluEval split")
    parser.add_argument("--semantic-for-calibration", action="store_true", help="Compute semantic entropy during direction/weight calibration")
    parser.add_argument("--quality-se-mode", action="store_true", help="Dense SE mode (stride=1 and semantic on for all splits)")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cache processed indices and recompute samples")
    parser.add_argument("--cache-every", type=int, default=50, help="Checkpoint every N newly processed samples")
    parser.add_argument("--checkpoint-seconds", type=int, default=120, help="Checkpoint after this many seconds even if N is not reached")
    parser.add_argument("--progress-every", type=int, default=5, help="Print heartbeat every N newly processed samples")
    parser.add_argument("--progress-seconds", type=int, default=30, help="Print heartbeat after this many seconds without logs")
    parser.add_argument("--semantic-time-budget", type=float, default=25.0, help="Max seconds budget for semantic entropy per sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    analyzer_cfg = AnalyzerConfig(
        base_dir=args.base_dir,
        cache_every=args.cache_every,
        checkpoint_seconds=args.checkpoint_seconds,
        direction_samples=args.direction_samples,
        train_weight_samples=args.train_weight_samples,
        semantic_stride=args.semantic_stride,
        compute_semantic_for_calibration=args.semantic_for_calibration,
        compute_semantic_for_halueval=args.semantic_for_halueval,
        quality_se_mode=args.quality_se_mode,
        force_recompute=args.force_recompute,
        progress_every=args.progress_every,
        progress_seconds=args.progress_seconds,
    )

    model_cfg = ModelConfig(model_name="Qwen/Qwen2.5-7B", max_gpu_memory_gib=6)
    metric_cfg = MetricConfig(semantic_time_budget_seconds=args.semantic_time_budget)

    analyzer = RAGAnalyzer(analyzer_cfg=analyzer_cfg, model_cfg=model_cfg, metric_cfg=metric_cfg)
    run_summary = analyzer.run()
    print(json.dumps(run_summary, indent=2))
