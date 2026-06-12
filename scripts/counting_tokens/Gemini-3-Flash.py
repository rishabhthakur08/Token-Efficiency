#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents))
from run_llm_experiment import parse_args, run_experiment

API_KEY = ""
PROVIDER = "gemini"
BASE_URL = "https://generativelanguage.googleapis.com"
MODEL = "gemini-3-flash"
DATASETS = ["coding", "math", "science"]

if __name__ == "__main__":
    args = parse_args(DATASETS)
    run_experiment(
        experiment_dir=Path(__file__).resolve().parents[2] / "data" / "token_counts" / "Gemini-3-Flash",
        provider=PROVIDER,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        datasets=args.datasets,
        max_output_tokens=args.max_output_tokens,
        sleep_seconds=args.sleep,
        limit=args.limit,
    )
