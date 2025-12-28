import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm


HF_REPO = "wandb/RAGTruth-processed"
SPLIT = "train"
LIMIT = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

RAW_OUT = RAW_DIR / "ragtruth_raw.jsonl"
PROCESSED_OUT = PROCESSED_DIR / "ragtruth_processed.jsonl"
SAMPLE_OUT = SAMPLES_DIR / "ragtruth_sample.jsonl"
SAMPLE_N = 50

for p in [RAW_DIR, PROCESSED_DIR, SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def pick_first_present(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def join_context(ctx):
    if ctx is None:
        return ""
    if isinstance(ctx, str):
        return ctx.strip()

    if isinstance(ctx, list):
        parts = []
        for item in ctx:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "passage", "chunk"):
                    if key in item and item[key]:
                        parts.append(safe_str(item[key]).strip())
                        break
                else:
                    parts.append(safe_str(item).strip())
            else:
                parts.append(safe_str(item).strip())
        parts = [p for p in parts if p]
        return "\n\n---\n\n".join(parts)

    return safe_str(ctx).strip()


def normalize_row(row, idx):
    rid = pick_first_present(row, ("id", "example_id", "uid", "qid"))
    example_id = safe_str(rid) if rid is not None else f"ragtruth_{idx}"

    question = pick_first_present(row, ("question", "query", "prompt", "instruction"))
    if question is None:
        question = pick_first_present(row, ("input", "source", "document"))
    question = safe_str(question).strip() if question is not None else "[MISSING_QUESTION]"

    context = pick_first_present(row, ("context", "contexts", "retrieved_context", "evidence", "passages"))
    context = join_context(context)

    answer = pick_first_present(row, ("answer", "output", "response", "generation", "model_output"))
    answer = safe_str(answer).strip() if answer is not None else ""

    task = pick_first_present(row, ("task", "task_type", "dataset", "source_dataset", "benchmark"))
    task = safe_str(task).strip() if task is not None else None

    label = pick_first_present(
        row,
        (
            "label",
            "hallucination_labels_processed",
            "human_label",
            "faithfulness",
            "groundedness",
            "is_hallucination",
        ),
    )
    label = safe_str(label).strip() if label is not None else None

    normalized = {
        "example_id": example_id,
        "task": task,
        "question": question,
        "context": context,
        "answer": answer,
        "label": label,
        "meta": row,
    }
    return normalized


def main():
    ds = load_dataset(HF_REPO, split=SPLIT)

    raw_rows = []
    for i, row in enumerate(tqdm(ds)):
        if LIMIT is not None and i >= LIMIT:
            break
        raw_rows.append(dict(row))

    write_jsonl(RAW_OUT, raw_rows)

    processed_rows = []
    sample_rows = []

    n_total = len(ds) if LIMIT is None else min(LIMIT, len(ds))
    for i in tqdm(range(n_total)):
        row = dict(ds[i])
        norm = normalize_row(row, i)
        processed_rows.append(norm)
        if len(sample_rows) < SAMPLE_N:
            sample_rows.append(norm)

    write_jsonl(PROCESSED_OUT, processed_rows)
    write_jsonl(SAMPLE_OUT, sample_rows)


if __name__ == "__main__":
    main()
