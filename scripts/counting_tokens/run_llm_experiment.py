#!/usr/bin/env python3
"""Run prompt-token experiments for one model folder.

The per-model scripts in this repository import this module and pass a blank
API key placeholder plus provider/model settings. Results are written in the
existing format: {"tokens": [ ... ]}.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "coding/mbpp": ROOT / "data" / "prompts" / "coding" / "mbpp.jsonl",
    "coding/coding-full": ROOT / "data" / "prompts" / "coding" / "coding-full.jsonl",
    "math/math-full": ROOT / "data" / "prompts" / "math" / "math-full.jsonl",
    "science/science-full": ROOT / "data" / "prompts" / "science" / "science-full.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_for(row: dict[str, Any], dataset_name: str) -> str:
    if dataset_name.startswith("coding/"):
        tests = "\n".join(row.get("test_list", []))
        return (
            "Solve the following programming task. Return only the final Python "
            "solution.\n\n"
            f"Task:\n{row['text']}\n\n"
            f"Tests:\n{tests}"
        )
    return (
        "Answer the following problem. Return the final answer clearly.\n\n"
        f"{row['problem']}"
    )


def read_existing_tokens(path: Path) -> list[int]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("tokens", []))


def write_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump({"tokens": tokens}, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def token_count_from_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }
    data = post_json(
        base_url.rstrip("/") + "/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        payload,
    )
    usage = data.get("usage", {})
    return int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("total_tokens", 0)
    )


def token_count_from_anthropic(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> int:
    payload = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = post_json(
        base_url.rstrip("/") + "/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        payload,
    )
    usage = data.get("usage", {})
    return int(usage.get("output_tokens") or usage.get("total_tokens", 0))


def token_count_from_gemini(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> int:
    url = (
        base_url.rstrip("/")
        + f"/v1beta/models/{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = post_json(url, {}, payload)
    usage = data.get("usageMetadata", {})
    return int(
        usage.get("candidatesTokenCount")
        or usage.get("totalTokenCount")
        or 0
    )


def count_tokens(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> int:
    if not api_key:
        raise ValueError("Set API_KEY in this model's run_experiment.py before running.")
    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if provider == "anthropic":
        return token_count_from_anthropic(**kwargs)
    if provider == "gemini":
        return token_count_from_gemini(**kwargs)
    return token_count_from_openai_compatible(**kwargs)


def run_experiment(
    *,
    experiment_dir: Path,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    datasets: list[str],
    max_output_tokens: int,
    sleep_seconds: float,
    limit: int | None,
) -> None:
    for dataset_name in datasets:
        rows = load_jsonl(DATASETS[dataset_name])
        if limit is not None:
            rows = rows[:limit]

        output_path = experiment_dir / f"{dataset_name}.json"
        tokens = read_existing_tokens(output_path)
        if len(tokens) > len(rows):
            tokens = tokens[: len(rows)]

        print(f"{experiment_dir.name}: {dataset_name} ({len(tokens)}/{len(rows)} done)")
        for row in rows[len(tokens) :]:
            prompt = prompt_for(row, dataset_name)
            tokens.append(
                count_tokens(
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                )
            )
            write_tokens(output_path, tokens)
            print(f"  wrote {len(tokens)}/{len(rows)} -> {output_path}")
            if sleep_seconds:
                time.sleep(sleep_seconds)


def parse_args(default_datasets: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to run. Repeat for multiple datasets. Defaults to all.",
    )
    parser.add_argument("--limit", type=int, help="Only run the first N prompts.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between calls.")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    args = parser.parse_args()
    args.datasets = args.dataset or default_datasets
    return args

