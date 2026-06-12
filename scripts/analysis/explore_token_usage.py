#!/usr/bin/env python3
"""Generate descriptive tables and figures for output-token usage."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "prompt_id",
    "prompt_index",
    "dataset",
    "model",
    "output_tokens",
    "problem",
}
SUMMARY_STATS = [
    "count",
    "mean",
    "median",
    "std",
    "min",
    "p25",
    "p75",
    "p90",
    "p95",
    "p99",
    "max",
]
OUTLIER_COLUMNS = [
    "prompt_id",
    "prompt_index",
    "dataset",
    "model",
    "output_tokens",
    "lower_bound",
    "upper_bound",
    "problem",
]


class AnalysisError(ValueError):
    """Raised when the analysis input cannot be analyzed safely."""


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise AnalysisError(f"Input dataset does not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise AnalysisError(f"Could not read Parquet dataset {path}: {exc}") from exc

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise AnalysisError(f"Input dataset is missing required columns: {missing}")
    if frame.empty:
        raise AnalysisError("Input dataset contains no rows")
    if frame["output_tokens"].isna().any() or (frame["output_tokens"] < 0).any():
        raise AnalysisError("output_tokens must contain nonnegative values")

    return frame.sort_values(
        ["dataset", "prompt_index", "model"], kind="stable", ignore_index=True
    )


def summarize_tokens(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if groups:
        grouped = frame.groupby(groups, sort=True, observed=True)["output_tokens"]
        summary = grouped.agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            p99=lambda values: values.quantile(0.99),
            max="max",
        ).reset_index()
    else:
        values = frame["output_tokens"]
        summary = pd.DataFrame(
            [
                {
                    "count": values.count(),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "p90": values.quantile(0.90),
                    "p95": values.quantile(0.95),
                    "p99": values.quantile(0.99),
                    "max": values.max(),
                }
            ]
        )
    return summary[[*groups, *SUMMARY_STATS]]


def feature_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    candidates = [
        column
        for column in numeric.columns
        if column not in {"prompt_index", "output_tokens"}
    ]
    rows = []
    for dataset in sorted(frame["dataset"].unique()):
        subset = frame.loc[frame["dataset"] == dataset]
        for feature in candidates:
            if subset[feature].nunique(dropna=False) <= 1:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "pearson_correlation": subset[feature].corr(subset["output_tokens"]),
                    "spearman_correlation": subset[feature].corr(
                        subset["output_tokens"], method="spearman"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "spearman_correlation", "feature"],
        ascending=[True, False, True],
        kind="stable",
        ignore_index=True,
    )


def detect_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(["model", "dataset"], sort=True, observed=True):
        q1 = group["output_tokens"].quantile(0.25)
        q3 = group["output_tokens"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = group.loc[
            (group["output_tokens"] < lower_bound) | (group["output_tokens"] > upper_bound)
        ].copy()
        outliers["lower_bound"] = lower_bound
        outliers["upper_bound"] = upper_bound
        parts.append(outliers)

    if not parts:
        return pd.DataFrame(columns=OUTLIER_COLUMNS)
    return (
        pd.concat(parts, ignore_index=True)[OUTLIER_COLUMNS]
        .sort_values(
            ["dataset", "model", "output_tokens", "prompt_index"],
            ascending=[True, True, False, True],
            kind="stable",
            ignore_index=True,
        )
    )


def write_tables(frame: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "token_summary_overall": summarize_tokens(frame, []),
        "token_summary_by_model": summarize_tokens(frame, ["model"]),
        "token_summary_by_dataset": summarize_tokens(frame, ["dataset"]),
        "token_summary_by_model_dataset": summarize_tokens(frame, ["model", "dataset"]),
        "feature_correlations": feature_correlations(frame),
        "outliers": detect_outliers(frame),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_distributions(frame: pd.DataFrame, path: Path) -> None:
    grid = sns.displot(
        data=frame,
        x="output_tokens",
        col="dataset",
        col_order=sorted(frame["dataset"].unique()),
        kind="hist",
        bins=45,
        log_scale=True,
        facet_kws={"sharex": True, "sharey": False},
        height=4,
        aspect=1.05,
        color="#326B8C",
    )
    grid.set_axis_labels("Output tokens (log scale)", "Responses")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Output-Token Distributions by Dataset", y=1.05, fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def plot_model_medians(summary: pd.DataFrame, path: Path) -> None:
    order = summary.groupby("model")["median"].median().sort_values().index
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=summary,
        y="model",
        x="median",
        hue="dataset",
        order=order,
        hue_order=sorted(summary["dataset"].unique()),
        palette="colorblind",
    )
    plt.xscale("log")
    plt.xlabel("Median output tokens (log scale)")
    plt.ylabel("Model")
    plt.title("Median Output-Token Usage by Model and Dataset")
    save_figure(path)


def plot_model_dataset_heatmap(summary: pd.DataFrame, path: Path) -> None:
    matrix = summary.pivot(index="model", columns="dataset", values="median")
    matrix = matrix.loc[matrix.median(axis=1).sort_values().index]
    plt.figure(figsize=(9, 9))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Median tokens"})
    plt.xlabel("Dataset")
    plt.ylabel("Model")
    plt.title("Median Output Tokens by Model and Dataset")
    save_figure(path)


def plot_feature_correlations(correlations: pd.DataFrame, path: Path) -> None:
    matrix = correlations.pivot(
        index="feature", columns="dataset", values="spearman_correlation"
    )
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(9, max(7, len(matrix) * 0.38)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman correlation"},
    )
    plt.xlabel("Dataset")
    plt.ylabel("Feature")
    plt.title("Prompt-Feature Correlations with Output Tokens")
    save_figure(path)


def generate_figures(
    frame: pd.DataFrame, tables: dict[str, pd.DataFrame], figures_dir: Path
) -> None:
    sns.set_theme(style="whitegrid")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_distributions(frame, figures_dir / "token_distributions_by_dataset.png")
    plot_model_medians(
        tables["token_summary_by_model_dataset"],
        figures_dir / "median_tokens_by_model_dataset.png",
    )
    plot_model_dataset_heatmap(
        tables["token_summary_by_model_dataset"],
        figures_dir / "model_dataset_token_heatmap.png",
    )
    plot_feature_correlations(
        tables["feature_correlations"],
        figures_dir / "feature_correlation_heatmap.png",
    )


def run_analysis(input_path: Path, output_dir: Path, figures_dir: Path) -> None:
    frame = load_dataset(input_path)
    tables = write_tables(frame, output_dir)
    generate_figures(frame, tables, figures_dir)
    print(
        f"Analyzed {len(frame):,} rows; wrote {len(tables)} tables to {output_dir} "
        f"and 4 figures to {figures_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "analysis_dataset.parquet",
        help="Input analysis dataset in Parquet format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "analysis",
        help="Directory for generated CSV summary tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_analysis(args.input, args.output_dir, args.figures_dir)
    except AnalysisError as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc


if __name__ == "__main__":
    main()
