#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.features import extract_features

DATASETS = {
    "math":    ROOT / "data" / "prompts" / "math"    / "math-full.jsonl",
    "coding":  ROOT / "data" / "prompts" / "coding"  / "coding-full.jsonl",
    "mbpp":    ROOT / "data" / "prompts" / "coding"  / "mbpp.jsonl",
    "science": ROOT / "data" / "prompts" / "science" / "science-full.jsonl",
}

OUTPUT_DIR = ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    for name, path in DATASETS.items():
        print(f"processing {name}...")
        rows = load_jsonl(path)

        results = []
        for i, row in enumerate(rows):
            features = extract_features(row, name)
            features["index"] = i
            features["dataset"] = name
            results.append(features)

        output_path = OUTPUT_DIR / f"{name}_features.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"  saved {len(results)} records -> {output_path}")

    print("\ndone.")


if __name__ == "__main__":
    main()
