"""
Dataset loading, schema inspection, and token-label alignment utilities.
"""
from __future__ import annotations

import json
import os
import pprint
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, load_from_disk


RAGTRUTH_FIELDS = {
    "id": "id",
    "query": "query",
    "context": "context",
    "output": "output",
    "task_type": "task_type",
    "quality": "quality",
    "model": "model",
    "temperature": "temperature",
    "hallucination_labels": "hallucination_labels",
    "hallucination_labels_processed": "hallucination_labels_processed",
    "input_str": "input_str",
}

HALUEVAL_FIELDS = {
    "prompt": "prompt",
    "answer": "answer",
    "labels": "labels",
    "split": "split",
    "task_type": "task_type",
    "dataset": "dataset",
    "language": "language",
}

TYPE_MAP = {
    "evident conflict": "contradictory",
    "subtle conflict": "contradictory",
    "conflict": "contradictory",
    "evident baseless info": "fabricated",
    "baseless info": "fabricated",
    "subtle baseless info": "unsupported",
    "unsupported": "unsupported",
    "fabricated": "fabricated",
    "contradictory": "contradictory",
}


def load_ragtruth_splits(base_dir: str) -> Tuple[Dataset, Dataset]:
    train_path = os.path.join(base_dir, "ragtruth_processed", "train")
    test_path = os.path.join(base_dir, "ragtruth_processed", "test")
    train_ds = load_from_disk(train_path)
    test_ds = load_from_disk(test_path)
    return train_ds, test_ds


def load_halueval_train(base_dir: str) -> Dataset:
    train_path = os.path.join(base_dir, "halueval_spans", "train")
    return load_from_disk(train_path)


def inspect_datasets(base_dir: str, output_path: str) -> None:
    """Writes field names and one full sample from each required split."""
    rag_train, rag_test = load_ragtruth_splits(base_dir)
    halu_train = load_halueval_train(base_dir)

    sections = []
    for name, path, ds in [
        ("ragtruth_train", os.path.join("ragtruth_processed", "train"), rag_train),
        ("ragtruth_test", os.path.join("ragtruth_processed", "test"), rag_test),
        ("halueval_train", os.path.join("halueval_spans", "train"), halu_train),
    ]:
        sections.append(f"=== {name} ===")
        sections.append(f"path: {path}")
        sections.append("fields:")
        sections.append(json.dumps(list(ds.features.keys()), indent=2, ensure_ascii=False))
        sections.append("sample:")
        sections.append(pprint.pformat(ds[0] if len(ds) > 0 else {}, width=140, compact=False, sort_dicts=False))
        sections.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


def _map_type(raw_label: str) -> str:
    if raw_label is None:
        return "unsupported"
    return TYPE_MAP.get(str(raw_label).strip().lower(), "unsupported")


def parse_ragtruth_labels(sample: Dict) -> List[Dict]:
    raw = sample.get(RAGTRUTH_FIELDS["hallucination_labels"], "[]")
    if not raw or raw == "[]":
        return []

    try:
        labels = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return []

    parsed = []
    if not isinstance(labels, list):
        return parsed

    for item in labels:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        mapped_type = _map_type(item.get("label_type", item.get("label", "unsupported")))
        parsed.append(
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "text": str(item.get("text", "")),
                "label_type": str(item.get("label_type", "unknown")),
                "mapped_type": mapped_type,
            }
        )
    return parsed


def parse_halueval_labels(sample: Dict) -> List[Dict]:
    labels = sample.get(HALUEVAL_FIELDS["labels"], [])
    if labels is None:
        return []

    parsed = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        mapped_type = _map_type(item.get("label", item.get("label_type", "unsupported")))
        parsed.append(
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "text": "",
                "label_type": str(item.get("label", "unknown")),
                "mapped_type": mapped_type,
            }
        )
    return parsed


def align_spans_to_tokens(tokenizer, text: str, spans: List[Dict]):
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )
    offsets = encoded["offset_mapping"]
    token_ids = np.array(encoded["input_ids"], dtype=np.int64)

    labels = np.zeros(len(token_ids), dtype=np.int32)
    types = np.array(["faithful"] * len(token_ids), dtype=object)

    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        mapped_type = span.get("mapped_type", "unsupported")
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < end and tok_end > start:
                labels[i] = 1
                types[i] = mapped_type

    return token_ids, labels, types


def extract_ragtruth_fields(sample: Dict) -> Tuple[str, str, str]:
    query = str(sample.get(RAGTRUTH_FIELDS["query"], "")).strip()
    context = str(sample.get(RAGTRUTH_FIELDS["context"], "")).strip()
    output = str(sample.get(RAGTRUTH_FIELDS["output"], "")).strip()
    return query, context, output


def extract_halueval_fields(sample: Dict) -> Tuple[str, str, str]:
    prompt = str(sample.get(HALUEVAL_FIELDS["prompt"], "")).strip()
    answer = str(sample.get(HALUEVAL_FIELDS["answer"], "")).strip()

    parts = prompt.split("\n", 1)
    query = parts[0].strip() if parts else ""
    context = parts[1].strip() if len(parts) > 1 else prompt
    return query, context, answer
