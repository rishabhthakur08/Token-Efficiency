#!/usr/bin/env python3
"""Build and validate the unified prompt-by-model analysis dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("coding", "math", "science")
TOKEN_FILENAMES = {dataset: f"{dataset}-full.json" for dataset in DATASETS}


class ValidationError(ValueError):
    """Raised when source data cannot be aligned safely."""


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not read valid JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValidationError(
                            f"Invalid JSON in {path} at line {line_number}: {exc}"
                        ) from exc
    except OSError as exc:
        raise ValidationError(f"Could not read {path}: {exc}") from exc
    return rows


def discover_models(token_counts_dir: Path) -> list[Path]:
    if not token_counts_dir.is_dir():
        raise ValidationError(f"Token-count directory does not exist: {token_counts_dir}")
    models = sorted(path for path in token_counts_dir.iterdir() if path.is_dir())
    if not models:
        raise ValidationError(f"No model directories found in {token_counts_dir}")
    return models


def validate_prompts(prompts: list[dict[str, Any]], dataset: str) -> None:
    missing = [index for index, row in enumerate(prompts) if "id" not in row or "problem" not in row]
    if missing:
        raise ValidationError(
            f"{dataset}: prompts missing required id/problem fields at indices {missing[:5]}"
        )
    ids = [row["id"] for row in prompts]
    if len(ids) != len(set(ids)):
        raise ValidationError(f"{dataset}: prompt IDs must be unique")


def validate_features(
    features: Any, dataset: str, prompt_count: int
) -> list[dict[str, Any]]:
    if not isinstance(features, list):
        raise ValidationError(f"{dataset}: feature file must contain a JSON list")
    if len(features) != prompt_count:
        raise ValidationError(
            f"{dataset}: feature count {len(features)} does not match prompt count {prompt_count}"
        )
    indices = [row.get("index") for row in features]
    if indices != list(range(prompt_count)):
        raise ValidationError(f"{dataset}: feature indices must be sequential from 0")
    if any(row.get("dataset") != dataset for row in features):
        raise ValidationError(f"{dataset}: feature dataset labels are not aligned")
    return features


def validate_tokens(payload: Any, model: str, dataset: str, prompt_count: int) -> list[int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
        raise ValidationError(f"{model}/{dataset}: token file must contain a tokens list")
    tokens = payload["tokens"]
    if len(tokens) != prompt_count:
        raise ValidationError(
            f"{model}/{dataset}: token count {len(tokens)} does not match prompt count "
            f"{prompt_count}"
        )
    invalid = [
        index
        for index, value in enumerate(tokens)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0
    ]
    if invalid:
        raise ValidationError(
            f"{model}/{dataset}: tokens must be nonnegative integers; invalid indices "
            f"{invalid[:5]}"
        )
    return tokens


def build_analysis_dataset(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    prompts_dir = root / "data" / "prompts"
    features_dir = root / "data" / "prompt_features"
    token_counts_dir = root / "data" / "token_counts"
    models = discover_models(token_counts_dir)

    dataset_sources: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for dataset in DATASETS:
        prompts = load_jsonl(prompts_dir / f"{dataset}-full.jsonl")
        validate_prompts(prompts, dataset)
        features = validate_features(
            load_json(features_dir / f"{dataset}_features.json"), dataset, len(prompts)
        )
        dataset_sources[dataset] = (prompts, features)

    records: list[dict[str, Any]] = []
    combinations: list[dict[str, Any]] = []
    for model_dir in models:
        model = model_dir.name
        for dataset in DATASETS:
            prompts, features = dataset_sources[dataset]
            token_path = model_dir / dataset / TOKEN_FILENAMES[dataset]
            if not token_path.is_file():
                raise ValidationError(f"Missing expected token file: {token_path}")
            tokens = validate_tokens(load_json(token_path), model, dataset, len(prompts))
            combinations.append({"model": model, "dataset": dataset, "rows": len(tokens)})

            for index, (prompt, feature, output_tokens) in enumerate(
                zip(prompts, features, tokens, strict=True)
            ):
                feature_values = {
                    key: value for key, value in feature.items() if key not in {"index", "dataset"}
                }
                records.append(
                    {
                        "prompt_id": prompt["id"],
                        "prompt_index": index,
                        "dataset": dataset,
                        "model": model,
                        "output_tokens": output_tokens,
                        "problem": prompt["problem"],
                        **feature_values,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(
        ["dataset", "prompt_index", "model"], kind="stable", ignore_index=True
    )
    unique_prompts = frame[["dataset", "prompt_id"]].drop_duplicates().shape[0]
    report = {
        "status": "valid",
        "models": [path.name for path in models],
        "model_count": len(models),
        "datasets": list(DATASETS),
        "prompt_count": unique_prompts,
        "row_count": len(frame),
        "model_dataset_combinations": combinations,
    }
    return frame, report


def write_outputs(
    frame: pd.DataFrame, report: dict[str, Any], output_dir: Path, write_csv: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "analysis_dataset.parquet", index=False)
    if write_csv:
        frame.to_csv(output_dir / "analysis_dataset.csv", index=False)
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Output directory (default: data/processed).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip the human-readable CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        frame, report = build_analysis_dataset()
        write_outputs(frame, report, args.output_dir, write_csv=not args.no_csv)
    except ValidationError as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc
    print(
        f"Wrote {report['row_count']:,} rows for {report['model_count']} models and "
        f"{report['prompt_count']:,} prompts to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
