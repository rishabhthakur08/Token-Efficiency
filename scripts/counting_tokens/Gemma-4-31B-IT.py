#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_llm_experiment import parse_args, run_experiment

API_KEY = ""
PROVIDER = "openai-compatible"
BASE_URL = ""
MODEL = "gemma-4-31b-it"
DATASETS = ["coding/mbpp", "coding/coding-full", "math/math-full", "science/science-full"]

if __name__ == "__main__":
    args = parse_args(DATASETS)
    run_experiment(
        experiment_dir=Path(__file__).resolve().parent,
        provider=PROVIDER,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        datasets=args.datasets,
        max_output_tokens=args.max_output_tokens,
        sleep_seconds=args.sleep,
        limit=args.limit,
    )
