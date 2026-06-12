import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_processing.build_analysis_dataset import (
    ValidationError,
    build_analysis_dataset,
    write_outputs,
)


DATASETS = ("coding", "math", "science")


def make_repo(root: Path) -> Path:
    for dataset in DATASETS:
        prompt_dir = root / "data" / "prompts"
        feature_dir = root / "data" / "prompt_features"
        token_dir = root / "data" / "token_counts" / "model-a" / dataset
        prompt_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)
        token_dir.mkdir(parents=True, exist_ok=True)

        prompts = [
            {"id": f"{dataset}-0", "problem": "First problem", "answer": "ignored"},
            {"id": f"{dataset}-1", "problem": "Second problem", "metadata": {"ignored": True}},
        ]
        with (prompt_dir / f"{dataset}-full.jsonl").open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                handle.write(json.dumps(prompt) + "\n")
        (feature_dir / f"{dataset}_features.json").write_text(
            json.dumps(
                [
                    {"index": 0, "dataset": dataset, "word_count": 2},
                    {"index": 1, "dataset": dataset, "word_count": 2},
                ]
            ),
            encoding="utf-8",
        )
        (token_dir / f"{dataset}-full.json").write_text(
            json.dumps({"tokens": [10, 20]}), encoding="utf-8"
        )
    return root


def test_builds_sorted_long_format_and_excludes_answers(tmp_path: Path) -> None:
    frame, report = build_analysis_dataset(make_repo(tmp_path))

    assert len(frame) == 6
    assert report["model_count"] == 1
    assert report["prompt_count"] == 6
    assert "answer" not in frame.columns
    assert "metadata" not in frame.columns
    assert frame.iloc[0][["dataset", "prompt_index", "model"]].tolist() == [
        "coding",
        0,
        "model-a",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "Missing expected token file"),
        ("mismatch", "does not match prompt count"),
        ("invalid", "nonnegative integers"),
        ("duplicate", "prompt IDs must be unique"),
    ],
)
def test_validation_failures(tmp_path: Path, mutation: str, message: str) -> None:
    root = make_repo(tmp_path)
    token_path = root / "data" / "token_counts" / "model-a" / "coding" / "coding-full.json"
    prompt_path = root / "data" / "prompts" / "coding-full.jsonl"
    if mutation == "missing":
        token_path.unlink()
    elif mutation == "mismatch":
        token_path.write_text(json.dumps({"tokens": [10]}), encoding="utf-8")
    elif mutation == "invalid":
        token_path.write_text(json.dumps({"tokens": [10, -1]}), encoding="utf-8")
    else:
        prompts = [json.loads(line) for line in prompt_path.read_text().splitlines()]
        prompts[1]["id"] = prompts[0]["id"]
        prompt_path.write_text("\n".join(map(json.dumps, prompts)) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        build_analysis_dataset(root)


def test_parquet_and_csv_match(tmp_path: Path) -> None:
    frame, report = build_analysis_dataset(make_repo(tmp_path / "repo"))
    output_dir = tmp_path / "outputs"
    write_outputs(frame, report, output_dir, write_csv=True)

    parquet = pd.read_parquet(output_dir / "analysis_dataset.parquet")
    csv = pd.read_csv(output_dir / "analysis_dataset.csv")
    pd.testing.assert_frame_equal(parquet, csv, check_dtype=False)
